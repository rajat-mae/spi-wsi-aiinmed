#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CONCH Baseline B + Model C training script
==========================================

This script trains:

1) Baseline B
   - frozen CONCH image encoder
   - small supervised probe head
   - predicts pathway scores from patch PNGs

2) Model C
   - CONCH image encoder with lightweight residual adapters
   - same supervised probe head
   - predicts pathway scores from patch PNGs

Expected inputs
---------------
A patch folder containing PNG images named with spot/barcode IDs, e.g.
    AAACAAGTATCTCCCA-1.png
or filenames that contain the barcode somewhere in the stem, e.g.
    spot_AAACAAGTATCTCCCA-1_x123_y456.png

A hallmark pathway CSV in wide format like:
    first column  = pathway names
    other columns = spot/barcode IDs
"""

import os
import re
import sys
import json
import copy
import argparse
import random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def normalize_pathway_name(x: str) -> str:
    return str(x).strip().upper()


def load_hallmark_wide_csv(pathway_csv: str, selected_pathways: Optional[List[str]] = None) -> pd.DataFrame:
    raw = pd.read_csv(pathway_csv, index_col=0)
    raw.index = [normalize_pathway_name(x) for x in raw.index]
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.apply(lambda col: pd.to_numeric(col, errors="coerce"))

    if selected_pathways is not None and len(selected_pathways) > 0:
        selected = [normalize_pathway_name(x) for x in selected_pathways]
        keep = [p for p in selected if p in raw.index]
        if len(keep) == 0:
            raise ValueError(f"None of the selected pathways were found in CSV: {selected[:10]}")
        raw = raw.loc[keep]

    df = raw.T.copy()
    df.index.name = "barcode"
    df.reset_index(inplace=True)
    return df


BARCODE_REGEX = re.compile(r"([A-Za-z0-9]+-\d+)")


def extract_barcode_from_filename(filename: str, valid_barcodes: set) -> Optional[str]:
    stem = Path(filename).stem
    if stem in valid_barcodes:
        return stem
    m = BARCODE_REGEX.search(stem)
    if m:
        bc = m.group(1)
        if bc in valid_barcodes:
            return bc
    for vb in valid_barcodes:
        if vb in stem:
            return vb
    return None


def build_manifest_from_pngs(
    patch_dir: str,
    pathway_csv: str,
    sample_id: str,
    selected_pathways: Optional[List[str]] = None,
    patch_glob: str = "*.png",
) -> Tuple[pd.DataFrame, List[str]]:
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"Patch folder not found: {patch_dir}")
    if not os.path.isfile(pathway_csv):
        raise FileNotFoundError(f"Pathway CSV not found: {pathway_csv}")

    pathway_df = load_hallmark_wide_csv(pathway_csv, selected_pathways=selected_pathways)
    target_cols = [c for c in pathway_df.columns if c != "barcode"]
    if len(target_cols) == 0:
        raise RuntimeError(f"No pathway columns found in {pathway_csv}")

    barcode_to_targets = pathway_df.set_index("barcode").to_dict(orient="index")
    valid_barcodes = set(barcode_to_targets.keys())

    png_paths = sorted(Path(patch_dir).glob(patch_glob))
    if len(png_paths) == 0:
        raise RuntimeError(f"No PNG files found in {patch_dir} with pattern {patch_glob}")

    rows = []
    matched = 0
    unmatched = 0
    for p in png_paths:
        barcode = extract_barcode_from_filename(p.name, valid_barcodes)
        if barcode is None:
            unmatched += 1
            continue
        rec = {
            "sample_id": sample_id,
            "image_path": str(p),
            "barcode": barcode,
        }
        rec.update(barcode_to_targets[barcode])
        rows.append(rec)
        matched += 1

    print(f"[{sample_id}] matched PNGs: {matched} | unmatched PNGs: {unmatched}")

    if len(rows) == 0:
        raise RuntimeError("No PNG patches matched barcode columns in the CSV.")

    manifest = pd.DataFrame(rows)
    keep_cols = ["sample_id", "image_path", "barcode"] + target_cols
    manifest = manifest[keep_cols].copy()
    for c in target_cols:
        manifest[c] = pd.to_numeric(manifest[c], errors="coerce")
    manifest = manifest.dropna(subset=target_cols).reset_index(drop=True)
    return manifest, target_cols


def split_manifest(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if df["sample_id"].nunique() >= 3:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        train_val_idx, test_idx = next(gss1.split(df, groups=df["sample_id"]))
        train_val = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        rel_val_frac = val_frac / (1.0 - test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val_frac, random_state=seed)
        train_idx, val_idx = next(gss2.split(train_val, groups=train_val["sample_id"]))
        train_df = train_val.iloc[train_idx].reset_index(drop=True)
        val_df = train_val.iloc[val_idx].reset_index(drop=True)
        split_mode = "group_by_sample"
    else:
        train_val, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
        rel_val_frac = val_frac / (1.0 - test_frac)
        train_df, val_df = train_test_split(train_val, test_size=rel_val_frac, random_state=seed)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        split_mode = "random_patch_split"
    return train_df, val_df, test_df, split_mode


class PNGPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: List[str], preprocess):
        self.df = df.reset_index(drop=True).copy()
        self.target_cols = target_cols
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.preprocess(img)
        y = torch.tensor(row[self.target_cols].values.astype(np.float32))
        meta = {
            "sample_id": row["sample_id"],
            "barcode": row["barcode"],
            "image_path": row["image_path"],
        }
        return x, y, meta


def collate_meta(batch):
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y, list(metas)


class ArrayDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def regression_metrics(y_true_z: np.ndarray, y_pred_z: np.ndarray, target_cols_scaled: List[str], scaler: StandardScaler):
    y_true = scaler.inverse_transform(y_true_z)
    y_pred = scaler.inverse_transform(y_pred_z)
    base_names = [c.replace("__z", "") for c in target_cols_scaled]
    rows = []
    for j, name in enumerate(base_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        try:
            pcc = pearsonr(yt, yp)[0]
        except Exception:
            pcc = np.nan
        try:
            spr = spearmanr(yt, yp).correlation
        except Exception:
            spr = np.nan
        mse = float(np.mean((yt - yp) ** 2))
        mae = float(np.mean(np.abs(yt - yp)))
        rows.append({
            "target": name,
            "pearson_r": pcc,
            "spearman_rho": spr,
            "mse": mse,
            "mae": mae,
        })
    metrics_df = pd.DataFrame(rows)
    summary = {
        "mean_pearson_r": float(np.nanmean(metrics_df["pearson_r"])),
        "mean_spearman_rho": float(np.nanmean(metrics_df["spearman_rho"])),
        "mean_mse": float(np.nanmean(metrics_df["mse"])),
        "mean_mae": float(np.nanmean(metrics_df["mae"])),
    }
    return metrics_df, summary


class EarlyStopper:
    def __init__(self, patience=5, mode="min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.best_state = None
        self.bad_epochs = 0

    def step(self, value, model):
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "min" and value < self.best:
            improved = True
        elif self.mode == "max" and value > self.best:
            improved = True

        if improved:
            self.best = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class ProbeHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 64, init_scale: float = 1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(self.norm(x))))


class BlockWithAdapter(nn.Module):
    def __init__(self, block: nn.Module, dim: int, bottleneck: int, init_scale: float):
        super().__init__()
        self.block = block
        self.adapter = ResidualAdapter(dim, bottleneck=bottleneck, init_scale=init_scale)

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if isinstance(out, tuple):
            x0 = self.adapter(out[0])
            return (x0, *out[1:])
        return self.adapter(out)


def infer_block_hidden_dim(block: nn.Module) -> int:
    for attr in ["norm1", "norm", "ln_1", "ln1", "pre_norm"]:
        if hasattr(block, attr):
            mod = getattr(block, attr)
            if hasattr(mod, "normalized_shape"):
                ns = mod.normalized_shape
                if isinstance(ns, (tuple, list)):
                    return int(ns[0])
                return int(ns)
    for mod in block.modules():
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)
    raise RuntimeError(f"Could not infer hidden dim for block type: {block.__class__.__name__}")


def get_module_by_path(root: nn.Module, path: str):
    cur = root
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def list_candidate_block_containers(root: nn.Module):
    candidates = []

    def score_container(name: str, container: nn.Module):
        try:
            n = len(container)
        except Exception:
            return None
        if n == 0:
            return None
        first = container[0]
        cls = first.__class__.__name__.lower()
        score = 0
        if "block" in cls:
            score += 5
        if "residualattentionblock" in cls:
            score += 5
        if "transformerblock" in cls:
            score += 5
        lname = name.lower()
        for key in ["blocks", "resblocks", "transformer", "trunk", "visual", "encoder"]:
            if key in lname:
                score += 2
        score += min(n, 64) / 64.0
        n_sub = sum(1 for _ in first.modules())
        if n_sub > 5:
            score += 1
        return {
            "name": name,
            "container": container,
            "length": n,
            "first_cls": first.__class__.__name__,
            "score": score,
        }

    for name, mod in root.named_modules():
        if isinstance(mod, (nn.ModuleList, nn.Sequential)):
            cand = score_container(name, mod)
            if cand is not None:
                candidates.append(cand)
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return candidates


def find_transformer_block_container(conch_model: nn.Module):
    root = conch_model.visual if hasattr(conch_model, "visual") else conch_model
    explicit_paths = [
        "transformer.resblocks",
        "trunk.blocks",
        "blocks",
        "resblocks",
        "visual.trunk.blocks",
        "visual.blocks",
        "visual.transformer.resblocks",
    ]
    for p in explicit_paths:
        try:
            mod = get_module_by_path(root, p)
            if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) > 0:
                print(f"[adapter] using explicit block path: {p} | n_blocks={len(mod)} | first={mod[0].__class__.__name__}")
                return p, mod
        except Exception:
            pass
    candidates = list_candidate_block_containers(root)
    if len(candidates) == 0:
        print("[adapter] No candidate containers found. Top-level visual children:")
        for name, mod in root.named_children():
            print(f"  - {name}: {mod.__class__.__name__}")
        raise RuntimeError("Could not find a transformer block container inside the CONCH visual encoder.")
    print("[adapter] top candidate containers:")
    for c in candidates[:10]:
        print(f"  - {c['name']} | len={c['length']} | first={c['first_cls']} | score={c['score']:.2f}")
    best = candidates[0]
    print(f"[adapter] selected: {best['name']} | n_blocks={best['length']} | first={best['first_cls']}")
    return best["name"], best["container"]


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _module_device(mod: nn.Module):
    for p in mod.parameters(recurse=True):
        return p.device
    for b in mod.buffers(recurse=True):
        return b.device
    return torch.device("cpu")


def inject_adapters_into_last_blocks(conch_model: nn.Module, last_n: int, bottleneck: int, init_scale: float):
    path, block_container = find_transformer_block_container(conch_model)
    n = len(block_container)
    if last_n > n:
        raise ValueError(f"Requested last_n={last_n}, but the container has only {n} blocks.")
    selected_indices = list(range(n - last_n, n))
    adapter_modules = []
    for idx in selected_indices:
        old_block = block_container[idx]
        dim = infer_block_hidden_dim(old_block)
        block_device = _module_device(old_block)
        wrapped = BlockWithAdapter(old_block, dim=dim, bottleneck=bottleneck, init_scale=init_scale).to(block_device)
        block_container[idx] = wrapped
        adapter_modules.append(wrapped.adapter)
        print(f"[adapter] wrapped block {idx}/{n-1} | type={old_block.__class__.__name__} | dim={dim} | device={block_device}")
    return path, selected_indices, adapter_modules


def load_conch(conch_repo: str, conch_checkpoint: str, hf_token: str, device):
    if conch_repo and os.path.isdir(conch_repo) and conch_repo not in sys.path:
        sys.path.insert(0, conch_repo)
    from conch.open_clip_custom import create_model_from_pretrained
    kwargs = {}
    if isinstance(conch_checkpoint, str) and conch_checkpoint.startswith("hf_hub:") and hf_token:
        kwargs["hf_auth_token"] = hf_token
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path=conch_checkpoint,
        **kwargs,
    )
    model = model.to(device)
    return model, preprocess


@torch.no_grad()
def extract_embeddings(encoder_model, loader, device, amp=True):
    encoder_model.eval()
    all_feats, all_y, all_meta = [], [], []
    autocast_enabled = amp and (device.type == "cuda")
    for x, y, meta in tqdm(loader, desc="Extract embeddings", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            feats = encoder_model.encode_image(x, proj_contrast=False, normalize=False)
        feats = feats.float().cpu()
        all_feats.append(feats)
        all_y.append(y.cpu())
        all_meta.extend(meta)
    X = torch.cat(all_feats, dim=0).numpy()
    Y = torch.cat(all_y, dim=0).numpy()
    meta_df = pd.DataFrame(all_meta)
    return X, Y, meta_df


def train_probe_head(train_loader, val_loader, in_dim, out_dim, device, cfg):
    model = ProbeHead(in_dim=in_dim, out_dim=out_dim, hidden_dim=cfg["probe_hidden_dim"], dropout=cfg["probe_dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["probe_lr"], weight_decay=cfg["probe_wd"])
    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=cfg["probe_patience"], mode="min")
    history = []
    for epoch in range(1, cfg["probe_epochs"] + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Baseline B] epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")
        if stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break
    model.load_state_dict(stopper.best_state)
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_probe_head(model, loader, device):
    model.eval()
    preds, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).float().cpu().numpy()
        preds.append(pred)
        ys.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)


class ConchAdapterRegressor(nn.Module):
    def __init__(self, conch_model, feature_dim: int, out_dim: int, probe_hidden_dim: int, probe_dropout: float):
        super().__init__()
        self.conch = conch_model
        self.head = ProbeHead(in_dim=feature_dim, out_dim=out_dim, hidden_dim=probe_hidden_dim, dropout=probe_dropout)

    def forward(self, x):
        feats = self.conch.encode_image(x, proj_contrast=False, normalize=False)
        return self.head(feats)


def run_epoch_modelC(model, loader, device, optimizer=None, train=False, amp=True):
    criterion = nn.MSELoss()
    losses = []
    preds_all, ys_all = [], []
    model.train(train)
    autocast_enabled = amp and (device.type == "cuda")
    for x, y, meta in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            pred = model(x)
            loss = criterion(pred, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(loss.detach().item())
        preds_all.append(pred.detach().float().cpu().numpy())
        ys_all.append(y.detach().float().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(preds_all, axis=0), np.concatenate(ys_all, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train CONCH Baseline B and Model C from PNG patches + hallmark CSV.")
    parser.add_argument("--patch_dir", type=str, required=True)
    parser.add_argument("--pathway_csv", type=str, required=True)
    parser.add_argument("--sample_id", type=str, default="sample001")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--patch_glob", type=str, default="*.png")
    parser.add_argument("--selected_pathways", nargs="*", default=None)
    parser.add_argument("--conch_repo", type=str, default="./CONCH")
    parser.add_argument("--conch_checkpoint", type=str, default="hf_hub:MahmoodLab/conch")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--probe_hidden_dim", type=int, default=256)
    parser.add_argument("--probe_dropout", type=float, default=0.10)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_wd", type=float, default=1e-4)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_patience", type=int, default=7)
    parser.add_argument("--adapter_last_n_blocks", type=int, default=2)
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--adapter_init_scale", type=float, default=1e-3)
    parser.add_argument("--adapter_lr", type=float, default=3e-4)
    parser.add_argument("--adapter_wd", type=float, default=1e-4)
    parser.add_argument("--adapter_epochs", type=int, default=50)
    parser.add_argument("--adapter_patience", type=int, default=5)
    parser.add_argument("--print_adapter_candidates", action="store_true")
    args = parser.parse_args()

    args.amp = False if args.no_amp else True
    os.makedirs(args.work_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.work_dir, "run_config.json"))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    def set_worker_seed(worker_id: int):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    manifest, target_cols = build_manifest_from_pngs(
        patch_dir=args.patch_dir,
        pathway_csv=args.pathway_csv,
        sample_id=args.sample_id,
        selected_pathways=args.selected_pathways,
        patch_glob=args.patch_glob,
    )
    print("Manifest shape:", manifest.shape)
    print("Num targets:", len(target_cols))
    print("Targets (first 10):", target_cols[:10])
    print(manifest.head())
    manifest.to_csv(os.path.join(args.work_dir, "patch_manifest.csv"), index=False)
    save_json(target_cols, os.path.join(args.work_dir, "target_names.json"))

    train_df, val_df, test_df, split_mode = split_manifest(manifest, args.val_frac, args.test_frac, args.seed)
    print("Split mode:", split_mode)
    print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)

    target_scaler = StandardScaler()
    target_scaler.fit(train_df[target_cols].values)
    for df_ in (train_df, val_df, test_df):
        scaled = target_scaler.transform(df_[target_cols].values)
        for i, c in enumerate(target_cols):
            df_[f"{c}__z"] = scaled[:, i]
    scaled_target_cols = [f"{c}__z" for c in target_cols]
    save_json({"target_names": target_cols, "mean": target_scaler.mean_.tolist(), "scale": target_scaler.scale_.tolist()}, os.path.join(args.work_dir, "target_scaler_stats.json"))

    conch_model, conch_preprocess = load_conch(args.conch_repo, args.conch_checkpoint, args.hf_token, device)
    conch_model.eval()

    train_patch_ds = PNGPatchDataset(train_df, scaled_target_cols, conch_preprocess)
    val_patch_ds = PNGPatchDataset(val_df, scaled_target_cols, conch_preprocess)
    test_patch_ds = PNGPatchDataset(test_df, scaled_target_cols, conch_preprocess)

    train_patch_loader = DataLoader(train_patch_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=set_worker_seed, generator=g, collate_fn=collate_meta)
    val_patch_loader = DataLoader(val_patch_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=set_worker_seed, generator=g, collate_fn=collate_meta)
    test_patch_loader = DataLoader(test_patch_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=set_worker_seed, generator=g, collate_fn=collate_meta)

    print("\n================ Baseline B ================\n")
    for p in conch_model.parameters():
        p.requires_grad = False
    conch_model.eval()

    X_train, Y_train, _ = extract_embeddings(conch_model, train_patch_loader, device, amp=args.amp)
    X_val, Y_val, _ = extract_embeddings(conch_model, val_patch_loader, device, amp=args.amp)
    X_test, Y_test, _ = extract_embeddings(conch_model, test_patch_loader, device, amp=args.amp)
    np.savez_compressed(os.path.join(args.work_dir, "frozen_conch_embeddings.npz"), X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test)

    train_feat_loader = DataLoader(ArrayDataset(X_train, Y_train), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_feat_loader = DataLoader(ArrayDataset(X_val, Y_val), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_feat_loader = DataLoader(ArrayDataset(X_test, Y_test), batch_size=args.batch_size, shuffle=False, num_workers=0)
    probe_cfg = {"probe_hidden_dim": args.probe_hidden_dim, "probe_dropout": args.probe_dropout, "probe_lr": args.probe_lr, "probe_wd": args.probe_wd, "probe_epochs": args.probe_epochs, "probe_patience": args.probe_patience}
    probe_model_B, baseline_history = train_probe_head(train_feat_loader, val_feat_loader, in_dim=X_train.shape[1], out_dim=Y_train.shape[1], device=device, cfg=probe_cfg)
    baseline_pred_z, baseline_true_z = predict_probe_head(probe_model_B, test_feat_loader, device)
    baseline_metrics_df, baseline_summary = regression_metrics(baseline_true_z, baseline_pred_z, scaled_target_cols, target_scaler)
    print("Baseline B summary:")
    print(json.dumps(baseline_summary, indent=2))
    baseline_history.to_csv(os.path.join(args.work_dir, "baselineB_history.csv"), index=False)
    baseline_metrics_df.to_csv(os.path.join(args.work_dir, "baselineB_test_metrics.csv"), index=False)
    torch.save(probe_model_B.state_dict(), os.path.join(args.work_dir, "baselineB_probe_head.pt"))

    print("\n================ Model C ================\n")
    conch_model_C, _ = load_conch(args.conch_repo, args.conch_checkpoint, args.hf_token, device)
    freeze_all(conch_model_C)

    if args.print_adapter_candidates:
        root = conch_model_C.visual if hasattr(conch_model_C, "visual") else conch_model_C
        print("[adapter] top-level visual children:")
        for name, mod in root.named_children():
            print(f"  - {name}: {mod.__class__.__name__}")
        print("[adapter] candidate containers:")
        for c in list_candidate_block_containers(root)[:20]:
            print(f"  - {c['name']} | len={c['length']} | first={c['first_cls']} | score={c['score']:.2f}")

    adapter_block_path, adapter_block_ids, adapter_modules = inject_adapters_into_last_blocks(conch_model_C, last_n=args.adapter_last_n_blocks, bottleneck=args.adapter_bottleneck, init_scale=args.adapter_init_scale)
    print("[adapter] final selected path:", adapter_block_path)
    save_json({"adapter_block_path": adapter_block_path, "adapter_block_ids": adapter_block_ids, "adapter_last_n_blocks": args.adapter_last_n_blocks, "adapter_bottleneck": args.adapter_bottleneck, "adapter_init_scale": args.adapter_init_scale}, os.path.join(args.work_dir, "adapter_config.json"))

    x0, _, _ = next(iter(train_patch_loader))
    with torch.no_grad():
        feat0 = conch_model_C.encode_image(x0[:2].to(device), proj_contrast=False, normalize=False)
    feature_dim = int(feat0.shape[-1])
    print("Feature dim:", feature_dim)

    modelC = ConchAdapterRegressor(conch_model=conch_model_C.to(device), feature_dim=feature_dim, out_dim=len(scaled_target_cols), probe_hidden_dim=args.probe_hidden_dim, probe_dropout=args.probe_dropout).to(device)
    for p in modelC.parameters():
        p.requires_grad = False
    for m in adapter_modules:
        for p in m.parameters():
            p.requires_grad = True
    for p in modelC.head.parameters():
        p.requires_grad = True
    print("Model C total params:", f"{count_total_params(modelC):,}")
    print("Model C trainable params:", f"{count_trainable_params(modelC):,}")

    optimizerC = torch.optim.AdamW([p for p in modelC.parameters() if p.requires_grad], lr=args.adapter_lr, weight_decay=args.adapter_wd)
    stopperC = EarlyStopper(patience=args.adapter_patience, mode="min")
    historyC = []
    for epoch in range(1, args.adapter_epochs + 1):
        train_loss, _, _ = run_epoch_modelC(modelC, train_patch_loader, device, optimizer=optimizerC, train=True, amp=args.amp)
        val_loss, _, _ = run_epoch_modelC(modelC, val_patch_loader, device, optimizer=None, train=False, amp=args.amp)
        historyC.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[Model C] epoch {epoch:02d} | train {train_loss:.5f} | val {val_loss:.5f}")
        if stopperC.step(val_loss, modelC):
            print("Early stopping triggered.")
            break
    modelC.load_state_dict(stopperC.best_state)
    _, test_pred_z_C, test_true_z_C = run_epoch_modelC(modelC, test_patch_loader, device, optimizer=None, train=False, amp=args.amp)
    modelC_metrics_df, modelC_summary = regression_metrics(test_true_z_C, test_pred_z_C, scaled_target_cols, target_scaler)
    print("Model C summary:")
    print(json.dumps(modelC_summary, indent=2))
    pd.DataFrame(historyC).to_csv(os.path.join(args.work_dir, "modelC_history.csv"), index=False)
    modelC_metrics_df.to_csv(os.path.join(args.work_dir, "modelC_test_metrics.csv"), index=False)
    torch.save(modelC.state_dict(), os.path.join(args.work_dir, "modelC_adapters_probe.pt"))

    comparison = baseline_metrics_df.merge(modelC_metrics_df, on="target", suffixes=("_baselineB", "_modelC"))
    comparison["delta_pearson_r"] = comparison["pearson_r_modelC"] - comparison["pearson_r_baselineB"]
    comparison["delta_spearman_rho"] = comparison["spearman_rho_modelC"] - comparison["spearman_rho_baselineB"]
    comparison["delta_mse"] = comparison["mse_modelC"] - comparison["mse_baselineB"]
    summary_rows = pd.DataFrame([
        {"model": "Baseline B", **baseline_summary},
        {"model": "Model C", **modelC_summary},
    ])
    summary_rows.to_csv(os.path.join(args.work_dir, "summary_comparison.csv"), index=False)
    comparison.to_csv(os.path.join(args.work_dir, "per_pathway_comparison.csv"), index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(baseline_history["epoch"], baseline_history["train_loss"], label="Baseline B train")
    plt.plot(baseline_history["epoch"], baseline_history["val_loss"], label="Baseline B val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Baseline B training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, "baselineB_history.png"), dpi=160)
    plt.close()

    histC = pd.DataFrame(historyC)
    plt.figure(figsize=(8, 4))
    plt.plot(histC["epoch"], histC["train_loss"], label="Model C train")
    plt.plot(histC["epoch"], histC["val_loss"], label="Model C val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Model C training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, "modelC_history.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(max(12, len(comparison) * 0.25), 5))
    x = np.arange(len(comparison))
    w = 0.38
    plt.bar(x - w / 2, comparison["pearson_r_baselineB"], width=w, label="Baseline B")
    plt.bar(x + w / 2, comparison["pearson_r_modelC"], width=w, label="Model C")
    plt.xticks(x, comparison["target"], rotation=90)
    plt.ylabel("Pearson r")
    plt.title("Per-pathway test performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.work_dir, "per_pathway_pearson_comparison.png"), dpi=160)
    plt.close()

    print("\nDone.")
    print(f"Results written to: {args.work_dir}")
    print("Saved files:")
    print(" - patch_manifest.csv")
    print(" - target_names.json")
    print(" - target_scaler_stats.json")
    print(" - adapter_config.json")
    print(" - frozen_conch_embeddings.npz")
    print(" - baselineB_probe_head.pt")
    print(" - modelC_adapters_probe.pt")
    print(" - baselineB_history.csv / .png")
    print(" - modelC_history.csv / .png")
    print(" - baselineB_test_metrics.csv")
    print(" - modelC_test_metrics.csv")
    print(" - summary_comparison.csv")
    print(" - per_pathway_comparison.csv")


if __name__ == "__main__":
    main()
