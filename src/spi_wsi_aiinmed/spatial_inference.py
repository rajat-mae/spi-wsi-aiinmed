#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Notebook-first spatial pathway inference with repeated LLM robustness.

This module keeps the structure of the original notebook workflow:
- pathway score CSV is read with pd.read_csv(..., index_col=0).T;
- images are matched by patient/sample ID;
- Claude generates pathology prompts per pathway;
- optional PubMed/critic pass can rewrite prompts;
- CONCH image/text embeddings localise pathway-associated prompts spatially;
- N independent LLM runs are saved separately;
- robustness metrics quantify run-to-run variability.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .keys import ask_user_for_keys, require_key
from .prompt_robustness import analyze_replicate_robustness

Image.MAX_IMAGE_PIXELS = None


@dataclass
class SpatialInferenceConfig:
    scores_csv: str
    image_folder: str
    out_dir: str = "spi_wsi_outputs"

    cancer_context: str = "oesophagus cancer"
    n_runs: int = 5
    iterations: int = 1
    tile_size: int = 800
    num_prompts: int = 5
    top_fraction: float = 0.20
    max_patients: Optional[int] = None
    max_pathways: Optional[int] = None

    anthropic_model: str = "claude-3-7-sonnet-20250219"
    anthropic_api_key: str = ""
    hf_token: str = ""
    entrez_email: str = ""
    temperature: float = 0.8
    max_tokens: int = 768

    conch_repo: str = "CONCH"
    conch_checkpoint: str = "hf_hub:MahmoodLab/conch"
    device: str = "auto"

    skip_pubmed: bool = True
    skip_critic: bool = True
    max_articles: int = 5
    interactive_pathology_feedback: bool = True

    save_per_pathway_heatmaps: bool = True
    image_extensions: Tuple[str, ...] = (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".png", ".jpg", ".jpeg")


def _optional_imports():
    missing = []
    try:
        import torch  # noqa
    except Exception:
        missing.append("torch")
    try:
        import matplotlib  # noqa
    except Exception:
        missing.append("matplotlib")
    if missing:
        raise RuntimeError("Missing required packages: " + ", ".join(missing))


def make_anthropic_client(api_key: str):
    import anthropic
    if hasattr(anthropic, "Anthropic"):
        return anthropic.Anthropic(api_key=api_key)
    return anthropic.Client(api_key=api_key)


def call_claude(client, *, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    try:
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.content
        if isinstance(content, list) and content:
            return getattr(content[0], "text", str(content[0])).strip()
        return str(content).strip()
    except Exception as e:
        print(f"Claude call failed: {e}")
        return ""


def load_conch_model(cfg: SpatialInferenceConfig):
    import torch
    if cfg.conch_repo and Path(cfg.conch_repo).exists() and str(Path(cfg.conch_repo).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(cfg.conch_repo).resolve()))
    try:
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
    except Exception as e:
        raise RuntimeError(
            "Could not import CONCH. Clone/install CONCH and set cfg.conch_repo correctly. "
            "Example: git clone https://github.com/mahmoodlab/CONCH.git"
        ) from e

    kwargs = {}
    token = cfg.hf_token or os.environ.get("HF_TOKEN", "")
    if token:
        kwargs["hf_auth_token"] = token
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        cfg.conch_checkpoint,
        **kwargs,
    )
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    model = model.to(device).eval()
    tokenizer = get_tokenizer()
    return model, preprocess, tokenizer, tokenize, device


def load_scores(scores_csv: str, max_patients: Optional[int], max_pathways: Optional[int]) -> pd.DataFrame:
    # Matches the original notebook convention: rows=pathways, cols=patient/sample IDs.
    df = pd.read_csv(scores_csv, index_col=0).T
    if max_patients is not None:
        df = df.iloc[:max_patients]
    if max_pathways is not None:
        df = df.iloc[:, :max_pathways]
    return df


def find_image_for_sample(image_folder: str, sample_id: str, extensions: Sequence[str]) -> Optional[Path]:
    folder = Path(image_folder)
    for ext in extensions:
        p = folder / f"{sample_id}{ext}"
        if p.exists():
            return p
    # Fallback: case-insensitive prefix match.
    sample_lower = str(sample_id).lower()
    for p in folder.iterdir() if folder.exists() else []:
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".vsi"}:
            if p.stem.lower().startswith(sample_lower):
                return p
    return None


def tile_and_embed_image(image_path: Path, *, tile_size: int, model, preprocess, device):
    import torch
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    tiles, coords = [], []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tiles.append(Image.fromarray(arr[y:y + tile_size, x:x + tile_size]))
            coords.append((x, y))
    if not tiles:
        raise RuntimeError(f"No tiles generated for {image_path}. Reduce tile_size={tile_size}.")
    embs = []
    with torch.no_grad():
        for t in tiles:
            inp = preprocess(t).unsqueeze(0).to(device)
            e = model.encode_image(inp, proj_contrast=False, normalize=False)
            e = e.detach().float().cpu().numpy().reshape(-1)
            embs.append(e)
    embs = np.vstack(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-8, None)
    return embs, coords


def generate_pathway_prompts(
    *,
    client,
    model_name: str,
    pathway: str,
    score: float,
    cancer_context: str,
    num_prompts: int,
    max_tokens: int,
    temperature: float,
    pathology_feedback: Optional[List[str]] = None,
    negative_prompts: Optional[List[str]] = None,
) -> List[str]:
    feedback_block = ""
    if pathology_feedback:
        feedback_block = "\nPathology feedback from previous iteration:\n" + "\n".join(f"- {x}" for x in pathology_feedback) + "\n"
    penalty_block = ""
    if negative_prompts:
        penalty_block = "\nAvoid these previously low-alignment prompts:\n" + "\n".join(f"- {x}" for x in negative_prompts[:10]) + "\n"

    prompt = f"""
You are a biomedical pathology assistant generating prompts for a pathology foundation model to spatially localise pathway activity on H&E whole-slide images.

Cancer context: {cancer_context}
Pathway: {pathway}
Normalised enrichment/pathway score: {score:.3f}

Instructions:
1. Generate {num_prompts} concise but morphologically informative prompts.
2. Keep each prompt one sentence.
3. Describe H&E-visible features, tumour architecture, immune/stromal patterns, necrosis, proliferation, invasion, and microenvironment cues when biologically appropriate.
4. Include relevant cell-type and pathway-associated biological cues, but do not make unsupported diagnostic claims.
5. Output only a numbered list from 1 to {num_prompts}.
{penalty_block}
{feedback_block}
""".strip()
    text = call_claude(client, model=model_name, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    out: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[\.)]\s*(.+)$", line.strip())
        if m:
            out.append(m.group(1).strip())
    if not out:
        for line in text.splitlines():
            line = line.strip(" -\t")
            if len(line) > 20:
                out.append(line)
    return out[:num_prompts]


def search_pubmed(query: str, *, max_articles: int, email: str) -> str:
    if not email:
        return "PubMed skipped because ENTREZ_EMAIL is missing."
    try:
        from Bio import Entrez
        Entrez.email = email
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles, sort="relevance", retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        if not ids:
            return "No articles found."
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
        text = handle.read()
        handle.close()
        return str(text)[:5000]
    except Exception as e:
        return f"PubMed lookup failed: {e}"


def evaluate_prompt_with_critic(
    *,
    client,
    model_name: str,
    pathway: str,
    score: float,
    prompt_text: str,
    abstracts: str,
    cancer_context: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    critic_prompt = f"""
You are a senior pathology professor reviewing a generated H&E prompt.
Cancer context: {cancer_context}
Pathway: {pathway}; score: {score:.3f}
Prompt: {prompt_text}
PubMed context: {abstracts}

Return only JSON with keys: critique, rewritten_prompt, category, score.
category must be one of: Tumour Instability, Tumour Aggressiveness, Tumour Suppression, Microenvironment, Other.
score must be integer 1-5.
""".strip()
    text = call_claude(client, model=model_name, prompt=critic_prompt, max_tokens=max_tokens, temperature=temperature)
    try:
        obj = json.loads(text)
        return {
            "Critique": obj.get("critique", ""),
            "Rewritten Prompt": obj.get("rewritten_prompt", prompt_text),
            "Category": obj.get("category", "Other"),
            "Score": int(obj.get("score", 3)),
        }
    except Exception:
        return {"Critique": "", "Rewritten Prompt": prompt_text, "Category": "Other", "Score": 3}


def build_prompts_for_sample(
    *,
    sample_id: str,
    score_row: pd.Series,
    client,
    cfg: SpatialInferenceConfig,
    pathology_feedback: List[str],
    negative_prompts_by_pathway: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    rows = []
    for pathway, score in score_row.items():
        if pd.isna(score):
            continue
        negative_prompts = None
        if negative_prompts_by_pathway:
            negative_prompts = negative_prompts_by_pathway.get(str(pathway), None)
        prompts = generate_pathway_prompts(
            client=client,
            model_name=cfg.anthropic_model,
            pathway=str(pathway),
            score=float(score),
            cancer_context=cfg.cancer_context,
            num_prompts=cfg.num_prompts,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            pathology_feedback=pathology_feedback,
            negative_prompts=negative_prompts,
        )
        for i, p in enumerate(prompts, 1):
            rec = {
                "sample_id": sample_id,
                "Pathway": str(pathway),
                "PathwayScore": float(score),
                "Prompt_ID": i,
                "Original Prompt": p,
                "Prompt": p,
                "Category": "Unreviewed",
                "Critique": "",
                "Score": np.nan,
            }
            if not cfg.skip_critic:
                abstracts = ""
                if not cfg.skip_pubmed:
                    abstracts = search_pubmed(p, max_articles=cfg.max_articles, email=cfg.entrez_email)
                eval_rec = evaluate_prompt_with_critic(
                    client=client,
                    model_name=cfg.anthropic_model,
                    pathway=str(pathway),
                    score=float(score),
                    prompt_text=p,
                    abstracts=abstracts,
                    cancer_context=cfg.cancer_context,
                    max_tokens=cfg.max_tokens,
                    temperature=max(0.0, min(cfg.temperature, 0.3)),
                )
                rec.update(eval_rec)
                rec["Prompt"] = rec.get("Rewritten Prompt", p) or p
            rows.append(rec)
    return pd.DataFrame(rows)


def encode_text_prompts(prompt_df: pd.DataFrame, *, model, tokenizer, tokenize_fn, device):
    import torch
    texts = prompt_df["Prompt"].fillna("").astype(str).tolist()
    tok = tokenize_fn(texts=texts, tokenizer=tokenizer).to(device)
    with torch.no_grad():
        e = model.encode_text(tok).detach().float().cpu().numpy()
    e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-8, None)
    return e


def compute_tile_prompt_cosine(
    *,
    tile_embs: np.ndarray,
    coords: List[Tuple[int, int]],
    prompt_df: pd.DataFrame,
    model,
    tokenizer,
    tokenize_fn,
    device,
) -> pd.DataFrame:
    text_embs = encode_text_prompts(prompt_df, model=model, tokenizer=tokenizer, tokenize_fn=tokenize_fn, device=device)
    sim = tile_embs @ text_embs.T
    recs = []
    for ti, (x, y) in enumerate(coords):
        for pj, p_row in prompt_df.reset_index(drop=True).iterrows():
            rec = p_row.to_dict()
            rec.update({
                "Tile_Index": int(ti),
                "x": int(x),
                "y": int(y),
                "Cosine_Similarity": float(sim[ti, pj]),
            })
            recs.append(rec)
    return pd.DataFrame(recs)


def summarize_tile_pathway_scores(cos_df: pd.DataFrame, *, top_fraction: float) -> pd.DataFrame:
    # Mean of positive cosine scores per tile/pathway; this is the spatial pathway score.
    tmp = cos_df.copy()
    tmp["positive_cosine"] = tmp["Cosine_Similarity"].clip(lower=0)
    score = tmp.groupby(["sample_id", "Tile_Index", "x", "y", "Pathway"], dropna=False).agg(
        value=("positive_cosine", "mean"),
        max_value=("positive_cosine", "max"),
        mean_raw_cosine=("Cosine_Similarity", "mean"),
        n_prompts=("Prompt", "count"),
        Prompt=("Prompt", lambda s: " || ".join(pd.Series(s).dropna().astype(str).head(3))),
        Category=("Category", lambda s: pd.Series(s).dropna().astype(str).mode().iat[0] if len(pd.Series(s).dropna()) else ""),
    ).reset_index()
    score["selected"] = False
    for _, idx in score.groupby(["sample_id", "Pathway"], dropna=False).groups.items():
        vals = score.loc[idx, "value"]
        if vals.notna().sum() == 0:
            continue
        cutoff = vals.quantile(1 - top_fraction)
        score.loc[idx, "selected"] = vals >= cutoff
    return score


def save_tile_map(image_path: Path, coords: List[Tuple[int, int]], outpath: Path, tile_size: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    img = np.array(Image.open(image_path).convert("RGB"))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")
    for idx, (x, y) in enumerate(coords):
        rect = patches.Rectangle((x, y), tile_size, tile_size, linewidth=1.2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x + tile_size / 2, y + tile_size / 2, str(idx), ha="center", va="center", fontsize=7,
                color="white", bbox=dict(facecolor="black", alpha=0.55, boxstyle="round"))
    fig.savefig(outpath, bbox_inches="tight", dpi=180)
    plt.close(fig)


def save_pathway_heatmap(image_path: Path, coords: List[Tuple[int, int]], tile_scores: pd.DataFrame, outpath: Path, tile_size: int, title: str = "") -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    img = np.array(Image.open(image_path).convert("RGB"))
    vals = pd.to_numeric(tile_scores["value"], errors="coerce")
    vmax = float(vals.max()) if vals.notna().any() else 1.0
    if vmax <= 0:
        vmax = 1.0
    cmap = cm.get_cmap("RdYlGn_r")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    for _, r in tile_scores.iterrows():
        ti = int(r["Tile_Index"])
        if ti >= len(coords):
            continue
        x, y = coords[ti]
        norm = max(0.0, min(1.0, float(r["value"]) / vmax)) if pd.notna(r["value"]) else 0.0
        rect = patches.Rectangle((x, y), tile_size, tile_size, facecolor=cmap(norm), linewidth=0, alpha=0.65)
        ax.add_patch(rect)
    fig.savefig(outpath, bbox_inches="tight", dpi=180)
    plt.close(fig)


def run_single_replicate(
    *,
    cfg: SpatialInferenceConfig,
    run_id: str,
    scores_df: pd.DataFrame,
    client,
    conch_bundle,
    run_dir: Path,
) -> Dict[str, Path]:
    model, preprocess, tokenizer, tokenize_fn, device = conch_bundle
    run_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = run_dir / "heatmaps"
    tilemap_dir = run_dir / "tilemaps"
    heatmap_dir.mkdir(exist_ok=True)
    tilemap_dir.mkdir(exist_ok=True)

    all_prompts, all_cos, all_scores = [], [], []
    feedback_by_sample: Dict[str, List[str]] = {}

    for sample_id, row in scores_df.iterrows():
        print(f"\n[{run_id}] Sample: {sample_id}")
        img_path = find_image_for_sample(cfg.image_folder, str(sample_id), cfg.image_extensions)
        if img_path is None:
            print(f"  No image found for sample {sample_id}; skipping.")
            continue
        tile_embs, coords = tile_and_embed_image(img_path, tile_size=cfg.tile_size, model=model, preprocess=preprocess, device=device)
        save_tile_map(img_path, coords, tilemap_dir / f"{sample_id}_tile_map.png", cfg.tile_size)

        feedback_by_sample.setdefault(str(sample_id), [])
        negative_prompts_by_pathway: Dict[str, List[str]] = {}
        last_prompt_df = pd.DataFrame()
        last_cos_df = pd.DataFrame()
        last_score_df = pd.DataFrame()

        for it in range(cfg.iterations):
            print(f"  Iteration {it + 1}/{cfg.iterations}")
            prompt_df = build_prompts_for_sample(
                sample_id=str(sample_id),
                score_row=row,
                client=client,
                cfg=cfg,
                pathology_feedback=feedback_by_sample[str(sample_id)],
                negative_prompts_by_pathway=negative_prompts_by_pathway,
            )
            if prompt_df.empty:
                print("  No prompts generated; skipping sample.")
                break
            prompt_df["run_id"] = run_id
            prompt_df["iteration"] = it + 1
            cos_df = compute_tile_prompt_cosine(
                tile_embs=tile_embs,
                coords=coords,
                prompt_df=prompt_df,
                model=model,
                tokenizer=tokenizer,
                tokenize_fn=tokenize_fn,
                device=device,
            )
            cos_df["run_id"] = run_id
            cos_df["iteration"] = it + 1
            score_df = summarize_tile_pathway_scores(cos_df, top_fraction=cfg.top_fraction)
            score_df["run_id"] = run_id
            score_df["iteration"] = it + 1

            # Prepare low-alignment prompts for next iteration.
            low = cos_df[cos_df["Cosine_Similarity"] < 0]
            negative_prompts_by_pathway = {
                str(pw): grp["Prompt"].dropna().astype(str).drop_duplicates().head(10).tolist()
                for pw, grp in low.groupby("Pathway")
            }

            last_prompt_df, last_cos_df, last_score_df = prompt_df, cos_df, score_df

            if cfg.save_per_pathway_heatmaps:
                for pathway, sdf in score_df.groupby("Pathway"):
                    safe_pw = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pathway))[:80]
                    save_pathway_heatmap(
                        img_path, coords, sdf, heatmap_dir / f"{sample_id}_iter{it + 1}_{safe_pw}_heatmap.png", cfg.tile_size,
                        title=f"{sample_id}: {pathway}"
                    )

            if cfg.iterations > 1 and cfg.interactive_pathology_feedback and it < cfg.iterations - 1:
                print("\nPathology feedback can guide the next iteration for this sample.")
                fb = input(f"Enter pathology guidance for {sample_id} after iteration {it + 1}, or press Enter to skip:\n> ").strip()
                if fb:
                    feedback_by_sample[str(sample_id)].append(fb)

        if not last_prompt_df.empty:
            all_prompts.append(last_prompt_df)
        if not last_cos_df.empty:
            all_cos.append(last_cos_df)
        if not last_score_df.empty:
            all_scores.append(last_score_df)

    paths: Dict[str, Path] = {}
    if all_prompts:
        prompts = pd.concat(all_prompts, ignore_index=True)
        paths["prompts_csv"] = run_dir / "generated_prompts.csv"
        prompts.to_csv(paths["prompts_csv"], index=False)
    if all_cos:
        cos = pd.concat(all_cos, ignore_index=True)
        paths["cosine_csv"] = run_dir / "tile_prompt_cosine.csv"
        cos.to_csv(paths["cosine_csv"], index=False)
    if all_scores:
        scores = pd.concat(all_scores, ignore_index=True)
        paths["tile_scores_csv"] = run_dir / "tile_pathway_scores.csv"
        scores.to_csv(paths["tile_scores_csv"], index=False)
    return paths


def run_spatial_inference_replicates(cfg: SpatialInferenceConfig) -> Dict[str, object]:
    _optional_imports()
    cfg.anthropic_api_key = require_key("ANTHROPIC_API_KEY", cfg.anthropic_api_key)
    cfg.hf_token = cfg.hf_token or os.environ.get("HF_TOKEN", "")
    cfg.entrez_email = cfg.entrez_email or os.environ.get("ENTREZ_EMAIL", "")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    scores_df = load_scores(cfg.scores_csv, cfg.max_patients, cfg.max_pathways)
    client = make_anthropic_client(cfg.anthropic_api_key)
    conch_bundle = load_conch_model(cfg)

    run_paths = []
    for i in range(cfg.n_runs):
        run_id = f"run_{i:03d}"
        run_dir = out / run_id
        print(f"\n========== Starting independent LLM replicate {run_id} ==========")
        paths = run_single_replicate(cfg=cfg, run_id=run_id, scores_df=scores_df, client=client, conch_bundle=conch_bundle, run_dir=run_dir)
        run_paths.append({"run_id": run_id, **{k: str(v) for k, v in paths.items()}})

    manifest = {"config": asdict(cfg), "runs": run_paths}
    (out / "replicate_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    score_files = [rp["tile_scores_csv"] for rp in run_paths if "tile_scores_csv" in rp]
    if len(score_files) >= 2:
        rob = analyze_replicate_robustness(
            inputs=score_files,
            out_dir=str(out / "robustness"),
            run_col="run_id",
            sample_col="sample_id",
            spatial_col="Tile_Index",
            pathway_col="Pathway",
            value_col="value",
            selected_col="selected",
            prompt_col="Prompt",
            top_fraction=None,
            make_plots=True,
        )
        manifest["robustness"] = rob
        (out / "replicate_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    else:
        print("Robustness analysis skipped: fewer than two successful replicate score files.")
    return manifest


def ask_keys_then_run(cfg: SpatialInferenceConfig) -> Dict[str, object]:
    """Notebook-only convenience function: asks user for keys, then runs."""
    ask_user_for_keys(default_entrez_email=cfg.entrez_email)
    cfg.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    cfg.hf_token = os.environ.get("HF_TOKEN", "")
    cfg.entrez_email = os.environ.get("ENTREZ_EMAIL", cfg.entrez_email)
    return run_spatial_inference_replicates(cfg)


def main_cli(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run repeated LLM spatial pathway inference and robustness analysis.")
    ap.add_argument("--scores_csv", required=True)
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--out_dir", default="spi_wsi_outputs")
    ap.add_argument("--cancer_context", default="oesophagus cancer")
    ap.add_argument("--n_runs", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--tile_size", type=int, default=800)
    ap.add_argument("--num_prompts", type=int, default=5)
    ap.add_argument("--top_fraction", type=float, default=0.20)
    ap.add_argument("--max_patients", type=int, default=None)
    ap.add_argument("--max_pathways", type=int, default=None)
    ap.add_argument("--anthropic_model", default="claude-3-7-sonnet-20250219")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--conch_repo", default="CONCH")
    ap.add_argument("--conch_checkpoint", default="hf_hub:MahmoodLab/conch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip_pubmed", action="store_true")
    ap.add_argument("--skip_critic", action="store_true")
    ap.add_argument("--no_interactive_feedback", action="store_true")
    ap.add_argument("--prompt_for_keys", action="store_true", help="Ask for API keys interactively if missing.")
    args = ap.parse_args(argv)

    if args.prompt_for_keys:
        ask_user_for_keys()
    cfg = SpatialInferenceConfig(
        scores_csv=args.scores_csv,
        image_folder=args.image_folder,
        out_dir=args.out_dir,
        cancer_context=args.cancer_context,
        n_runs=args.n_runs,
        iterations=args.iterations,
        tile_size=args.tile_size,
        num_prompts=args.num_prompts,
        top_fraction=args.top_fraction,
        max_patients=args.max_patients,
        max_pathways=args.max_pathways,
        anthropic_model=args.anthropic_model,
        temperature=args.temperature,
        conch_repo=args.conch_repo,
        conch_checkpoint=args.conch_checkpoint,
        device=args.device,
        skip_pubmed=args.skip_pubmed,
        skip_critic=args.skip_critic,
        interactive_pathology_feedback=not args.no_interactive_feedback,
    )
    manifest = run_spatial_inference_replicates(cfg)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main_cli()
