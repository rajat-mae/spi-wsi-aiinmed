"""Example: analyse already generated replicate tile_pathway_scores.csv files."""
from spi_wsi_aiinmed.prompt_robustness import analyze_replicate_robustness

manifest = analyze_replicate_robustness(
    inputs=["runs/spatial_inference_demo/run_*/tile_pathway_scores.csv"],
    out_dir="runs/spatial_inference_demo/robustness_manual",
    run_col="run_id",
    sample_col="sample_id",
    spatial_col="Tile_Index",
    pathway_col="Pathway",
    value_col="value",
    selected_col="selected",
    prompt_col="Prompt",
)
print(manifest)
