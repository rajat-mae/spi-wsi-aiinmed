#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Specificity testing for CONCH/LLM spatial pathway prompts.

The input is usually `tile_prompt_cosine.csv` produced by spatial inference.
For each tile and intended pathway, the script compares the pathway's prompt
similarity to decoy prompt similarities from other pathways on the same tile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def run_specificity_test(
    *,
    cosine_csv: str,
    out_dir: str,
    sample_col: str = "sample_id",
    spatial_col: str = "Tile_Index",
    pathway_col: str = "Pathway",
    value_col: str = "Cosine_Similarity",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cosine_csv)
    for c in [sample_col, spatial_col, pathway_col, value_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    pathway_tile = df.groupby([sample_col, spatial_col, pathway_col], dropna=False)[value_col].agg(
        target_mean="mean", target_max="max", n_target_prompts="count"
    ).reset_index()

    rows = []
    for (sample_id, tile_id), sub in pathway_tile.groupby([sample_col, spatial_col], dropna=False):
        all_vals = sub[[pathway_col, "target_mean"]].copy()
        for _, r in sub.iterrows():
            pathway = r[pathway_col]
            decoy = all_vals[all_vals[pathway_col] != pathway]["target_mean"]
            decoy_mean = float(decoy.mean()) if len(decoy) else np.nan
            decoy_max = float(decoy.max()) if len(decoy) else np.nan
            target = float(r["target_mean"])
            rank = int((all_vals["target_mean"] > target).sum() + 1)
            rows.append({
                sample_col: sample_id,
                spatial_col: tile_id,
                pathway_col: pathway,
                "target_mean_cosine": target,
                "target_max_cosine": float(r["target_max"]),
                "decoy_mean_cosine": decoy_mean,
                "decoy_max_cosine": decoy_max,
                "specificity_margin_mean": target - decoy_mean if np.isfinite(decoy_mean) else np.nan,
                "specificity_margin_max_decoy": target - decoy_max if np.isfinite(decoy_max) else np.nan,
                "pathway_rank_on_tile": rank,
                "n_pathways_on_tile": int(len(all_vals)),
                "rank_percentile": 1.0 - ((rank - 1) / max(1, len(all_vals) - 1)) if len(all_vals) > 1 else np.nan,
                "n_target_prompts": int(r["n_target_prompts"]),
            })
    per_tile = pd.DataFrame(rows)
    summary = per_tile.groupby([sample_col, pathway_col], dropna=False).agg(
        n_tiles=(spatial_col, "count"),
        mean_specificity_margin=("specificity_margin_mean", "mean"),
        median_specificity_margin=("specificity_margin_mean", "median"),
        mean_rank_percentile=("rank_percentile", "mean"),
        frac_top_rank=("pathway_rank_on_tile", lambda s: float((pd.Series(s) == 1).mean())),
    ).reset_index()
    overall = pd.DataFrame([{
        sample_col: "__overall__",
        pathway_col: "__overall__",
        "n_tiles": int(len(per_tile)),
        "mean_specificity_margin": float(per_tile["specificity_margin_mean"].mean()),
        "median_specificity_margin": float(per_tile["specificity_margin_mean"].median()),
        "mean_rank_percentile": float(per_tile["rank_percentile"].mean()),
        "frac_top_rank": float((per_tile["pathway_rank_on_tile"] == 1).mean()),
    }])
    summary = pd.concat([summary, overall], ignore_index=True)

    per_tile.to_csv(out / "specificity_by_tile_pathway.csv", index=False)
    summary.to_csv(out / "specificity_summary.csv", index=False)
    manifest = {"input": cosine_csv, "out_dir": str(out), "n_rows": int(len(per_tile))}
    (out / "specificity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main_cli(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Test specificity of spatial pathway prompts against decoy pathways.")
    ap.add_argument("--cosine_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_col", default="sample_id")
    ap.add_argument("--spatial_col", default="Tile_Index")
    ap.add_argument("--pathway_col", default="Pathway")
    ap.add_argument("--value_col", default="Cosine_Similarity")
    args = ap.parse_args(argv)
    manifest = run_specificity_test(
        cosine_csv=args.cosine_csv,
        out_dir=args.out_dir,
        sample_col=args.sample_col,
        spatial_col=args.spatial_col,
        pathway_col=args.pathway_col,
        value_col=args.value_col,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main_cli()
