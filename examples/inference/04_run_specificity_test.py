"""Example: test whether prompts are pathway-specific rather than generic."""
from spi_wsi_aiinmed.specificity_testing import run_specificity_test

manifest = run_specificity_test(
    cosine_csv="runs/spatial_inference_demo/run_000/tile_prompt_cosine.csv",
    out_dir="runs/spatial_inference_demo/run_000/specificity",
)
print(manifest)
