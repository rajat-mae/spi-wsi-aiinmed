#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robustness analysis for repeated stochastic LLM spatial-prompt runs.

The expected input is either:
1) one combined CSV containing multiple run IDs, or
2) a glob/list of per-run CSV files.

Rows should represent the same spatial unit across runs, usually:
    run_id, sample_id, Tile_Index, Pathway, value, selected, Prompt

The analyzer computes pairwise spatial correlation and selected-region overlap
per sample/pathway, then creates summary tables and consistency maps.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

RUN_RE = re.compile(r"(?:^|[_\-])run[_\-]?(\d+)(?:[_\-.]|$)", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suf == ".json":
        return pd.read_json(path)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table: {path}")


def infer_run_id(path: Path, fallback: int) -> str:
    m = RUN_RE.search(path.stem)
    return f"run_{int(m.group(1)):03d}" if m else f"run_{fallback:03d}"


def load_replicate_tables(inputs: Sequence[str], run_col: str = "run_id") -> pd.DataFrame:
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if any(ch in item for ch in "*?[]"):
            paths.extend([Path(x) for x in sorted(glob.glob(item))])
        else:
            paths.append(p)
    if not paths:
        raise FileNotFoundError("No replicate files matched inputs.")

    frames = []
    for i, p in enumerate(paths):
        if not p.exists():
            raise FileNotFoundError(p)
        df = _read_table(p)
        if run_col not in df.columns:
            df[run_col] = infer_run_id(p, i)
        df["source_file"] = str(p)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out[run_col] = out[run_col].astype(str)
    return out


def parse_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    truthy = {"1", "true", "yes", "y", "selected", "positive", "pos", "foreground", "fg"}
    return s.astype(str).str.strip().str.lower().isin(truthy)


def tokenize_text(x: object) -> set[str]:
    if pd.isna(x):
        return set()
    return {t.lower() for t in TOKEN_RE.findall(str(x))}


def jaccard(a: set, b: set) -> float:
    u = a | b
    return np.nan if not u else len(a & b) / len(u)


def dice(a: set, b: set) -> float:
    d = len(a) + len(b)
    return np.nan if d == 0 else 2.0 * len(a & b) / d


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa, bb = a[mask], b[mask]
    if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
        return np.nan
    return float(pearsonr(aa, bb)[0])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa, bb = a[mask], b[mask]
    if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
        return np.nan
    return float(spearmanr(aa, bb).correlation)


def add_selected_by_top_fraction(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    value_col: str,
    top_fraction: float,
    selected_col: str = "selected",
) -> pd.DataFrame:
    """Derive a selected mask by marking the top fraction per run/sample/pathway."""
    out = df.copy()
    out[selected_col] = False
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1].")
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        sub = out.loc[idx]
        vals = pd.to_numeric(sub[value_col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        cutoff = vals.quantile(1.0 - top_fraction)
        out.loc[idx, selected_col] = vals >= cutoff
    return out


def normalise_for_analysis(
    df: pd.DataFrame,
    *,
    run_col: str,
    sample_col: str,
    spatial_col: str,
    pathway_col: str,
    value_col: str,
    selected_col: Optional[str],
    prompt_col: Optional[str],
    top_fraction: Optional[float],
) -> pd.DataFrame:
    required = [run_col, sample_col, spatial_col, pathway_col, value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    if selected_col and selected_col in out.columns:
        out["__selected__"] = parse_bool(out[selected_col])
    elif top_fraction is not None:
        out = add_selected_by_top_fraction(
            out,
            group_cols=[run_col, sample_col, pathway_col],
            value_col=value_col,
            top_fraction=top_fraction,
            selected_col="__selected__",
        )
    else:
        out["__selected__"] = pd.NA

    if prompt_col and prompt_col in out.columns:
        out["__prompt__"] = out[prompt_col].fillna("").astype(str)
    else:
        out["__prompt__"] = ""

    keep = [run_col, sample_col, spatial_col, pathway_col, value_col, "__selected__", "__prompt__"]
    if "x" in out.columns:
        keep.append("x")
    if "y" in out.columns:
        keep.append("y")
    out = out[keep].drop_duplicates(
        subset=[run_col, sample_col, spatial_col, pathway_col], keep="last"
    )
    return out


def compute_pairwise_metrics(
    df: pd.DataFrame,
    *,
    run_col: str = "run_id",
    sample_col: str = "sample_id",
    spatial_col: str = "Tile_Index",
    pathway_col: str = "Pathway",
    value_col: str = "value",
) -> pd.DataFrame:
    rows = []
    for (sample_id, pathway), sdf in df.groupby([sample_col, pathway_col], dropna=False):
        runs = sorted(sdf[run_col].unique())
        for ra, rb in itertools.combinations(runs, 2):
            a = sdf[sdf[run_col] == ra].set_index(spatial_col)
            b = sdf[sdf[run_col] == rb].set_index(spatial_col)
            common = a.index.intersection(b.index)
            union = a.index.union(b.index)
            av = a.loc[common, value_col].to_numpy(float) if len(common) else np.array([])
            bv = b.loc[common, value_col].to_numpy(float) if len(common) else np.array([])

            sa = set(a.index[a["__selected__"].fillna(False).astype(bool)]) if "__selected__" in a else set()
            sb = set(b.index[b["__selected__"].fillna(False).astype(bool)]) if "__selected__" in b else set()

            # Row-wise prompt token jaccard over common spatial IDs
            row_jaccs = []
            if "__prompt__" in a and "__prompt__" in b:
                for loc in common:
                    row_jaccs.append(jaccard(tokenize_text(a.loc[loc, "__prompt__"]), tokenize_text(b.loc[loc, "__prompt__"])))
            corpus_a = tokenize_text(" ".join(a.get("__prompt__", pd.Series(dtype=str)).astype(str).tolist()))
            corpus_b = tokenize_text(" ".join(b.get("__prompt__", pd.Series(dtype=str)).astype(str).tolist()))

            rows.append({
                "sample_id": sample_id,
                "Pathway": pathway,
                "run_a": ra,
                "run_b": rb,
                "n_common_locations": int(len(common)),
                "n_union_locations": int(len(union)),
                "spatial_pearson_r": safe_pearson(av, bv),
                "spatial_spearman_rho": safe_spearman(av, bv),
                "selected_n_a": int(len(sa)),
                "selected_n_b": int(len(sb)),
                "selected_intersection_n": int(len(sa & sb)),
                "selected_jaccard": jaccard(sa, sb),
                "selected_dice": dice(sa, sb),
                "overlap_fraction_a": len(sa & sb) / len(sa) if sa else np.nan,
                "overlap_fraction_b": len(sa & sb) / len(sb) if sb else np.nan,
                "mean_row_prompt_token_jaccard": float(np.nanmean(row_jaccs)) if row_jaccs else np.nan,
                "corpus_prompt_token_jaccard": jaccard(corpus_a, corpus_b),
            })
    return pd.DataFrame(rows)


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spatial_pearson_r", "spatial_spearman_rho", "selected_jaccard", "selected_dice",
        "overlap_fraction_a", "overlap_fraction_b", "mean_row_prompt_token_jaccard",
        "corpus_prompt_token_jaccard",
    ]
    rows = []
    groups = list(pairwise.groupby(["sample_id", "Pathway"], dropna=False))
    for (sample_id, pathway), sdf in groups:
        row = {"sample_id": sample_id, "Pathway": pathway, "n_run_pairs": int(len(sdf))}
        for m in metrics:
            v = pd.to_numeric(sdf.get(m, pd.Series(dtype=float)), errors="coerce")
            row[f"{m}_mean"] = float(v.mean()) if v.notna().any() else np.nan
            row[f"{m}_std"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else np.nan
            row[f"{m}_min"] = float(v.min()) if v.notna().any() else np.nan
            row[f"{m}_max"] = float(v.max()) if v.notna().any() else np.nan
        rows.append(row)
    if len(pairwise):
        row = {"sample_id": "__overall__", "Pathway": "__overall__", "n_run_pairs": int(len(pairwise))}
        for m in metrics:
            v = pd.to_numeric(pairwise.get(m, pd.Series(dtype=float)), errors="coerce")
            row[f"{m}_mean"] = float(v.mean()) if v.notna().any() else np.nan
            row[f"{m}_std"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else np.nan
            row[f"{m}_min"] = float(v.min()) if v.notna().any() else np.nan
            row[f"{m}_max"] = float(v.max()) if v.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_consistency_map(
    df: pd.DataFrame,
    *,
    run_col: str,
    sample_col: str,
    spatial_col: str,
    pathway_col: str,
    value_col: str,
) -> pd.DataFrame:
    key = [sample_col, pathway_col, spatial_col]
    g = df.groupby(key, dropna=False)
    out = g.agg(
        n_runs=(run_col, "nunique"),
        value_mean=(value_col, "mean"),
        value_std=(value_col, "std"),
        value_min=(value_col, "min"),
        value_max=(value_col, "max"),
        n_unique_prompt_texts=("__prompt__", pd.Series.nunique),
    ).reset_index()
    denom = out["value_mean"].abs().replace(0, np.nan)
    out["value_cv"] = out["value_std"] / denom
    if "__selected__" in df.columns and df["__selected__"].notna().any():
        tmp = df.copy()
        tmp["__selected_float__"] = tmp["__selected__"].fillna(False).astype(float)
        sel = tmp.groupby(key, dropna=False)["__selected_float__"].agg(
            selection_frequency="mean", selected_count="sum"
        ).reset_index()
        out = out.merge(sel, on=key, how="left")
    return out


def save_plots(pairwise: pd.DataFrame, consistency: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    specs = [
        (pairwise, "spatial_pearson_r", "Pairwise spatial Pearson r", "pairwise_spatial_pearson_hist.png"),
        (pairwise, "spatial_spearman_rho", "Pairwise spatial Spearman rho", "pairwise_spatial_spearman_hist.png"),
        (pairwise, "selected_jaccard", "Pairwise selected-tile Jaccard", "pairwise_selected_jaccard_hist.png"),
        (pairwise, "selected_dice", "Pairwise selected-tile Dice", "pairwise_selected_dice_hist.png"),
        (consistency, "selection_frequency", "Selection frequency across LLM runs", "selection_frequency_hist.png"),
        (consistency, "value_std", "Spatial score standard deviation across LLM runs", "value_std_hist.png"),
    ]
    for table, col, title, fname in specs:
        if col not in table.columns:
            continue
        v = pd.to_numeric(table[col], errors="coerce").dropna()
        if v.empty:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(v, bins=min(30, max(5, int(np.sqrt(len(v))))))
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()


def analyze_replicate_robustness(
    *,
    inputs: Sequence[str],
    out_dir: str,
    run_col: str = "run_id",
    sample_col: str = "sample_id",
    spatial_col: str = "Tile_Index",
    pathway_col: str = "Pathway",
    value_col: str = "value",
    selected_col: Optional[str] = "selected",
    prompt_col: Optional[str] = "Prompt",
    top_fraction: Optional[float] = None,
    make_plots: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_replicate_tables(inputs, run_col=run_col)
    df = normalise_for_analysis(
        raw,
        run_col=run_col,
        sample_col=sample_col,
        spatial_col=spatial_col,
        pathway_col=pathway_col,
        value_col=value_col,
        selected_col=selected_col,
        prompt_col=prompt_col,
        top_fraction=top_fraction,
    )
    if df[run_col].nunique() < 2:
        raise ValueError("At least two independent LLM runs are required.")
    pairwise = compute_pairwise_metrics(df, run_col=run_col, sample_col=sample_col, spatial_col=spatial_col, pathway_col=pathway_col, value_col=value_col)
    summary = summarize_pairwise(pairwise)
    consistency = build_consistency_map(df, run_col=run_col, sample_col=sample_col, spatial_col=spatial_col, pathway_col=pathway_col, value_col=value_col)

    df.to_csv(out / "normalised_replicate_tile_scores.csv", index=False)
    pairwise.to_csv(out / "pairwise_replicate_metrics.csv", index=False)
    summary.to_csv(out / "summary_replicate_metrics.csv", index=False)
    consistency.to_csv(out / "prompt_consistency_map.csv", index=False)
    manifest = {
        "n_rows": int(len(df)),
        "n_runs": int(df[run_col].nunique()),
        "runs": sorted(df[run_col].unique().tolist()),
        "out_dir": str(out),
        "metrics": ["spatial_pearson_r", "spatial_spearman_rho", "selected_jaccard", "selected_dice"],
    }
    (out / "prompt_robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if make_plots:
        save_plots(pairwise, consistency, out)
    return manifest


def main_cli(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Analyze robustness across repeated stochastic LLM prompt runs.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_col", default="run_id")
    ap.add_argument("--sample_col", default="sample_id")
    ap.add_argument("--spatial_col", default="Tile_Index")
    ap.add_argument("--pathway_col", default="Pathway")
    ap.add_argument("--value_col", default="value")
    ap.add_argument("--selected_col", default="selected")
    ap.add_argument("--prompt_col", default="Prompt")
    ap.add_argument("--top_fraction", type=float, default=None)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args(argv)
    manifest = analyze_replicate_robustness(
        inputs=args.inputs,
        out_dir=args.out_dir,
        run_col=args.run_col,
        sample_col=args.sample_col,
        spatial_col=args.spatial_col,
        pathway_col=args.pathway_col,
        value_col=args.value_col,
        selected_col=args.selected_col or None,
        prompt_col=args.prompt_col or None,
        top_fraction=args.top_fraction,
        make_plots=not args.no_plots,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main_cli()
