"""Example: repeated LLM spatial pathway inference from CSV + image folder.
Run from repo root after installing the package in editable mode.
"""
from spi_wsi_aiinmed.keys import ask_user_for_keys
from spi_wsi_aiinmed.spatial_inference import SpatialInferenceConfig, run_spatial_inference_replicates

# Option A only: ask the user for keys at runtime. Nothing is saved to disk.
ask_user_for_keys(default_entrez_email="your.email@example.com")

cfg = SpatialInferenceConfig(
    scores_csv="examples/data/hallmark_patient_scores.csv",
    image_folder="examples/data/histology_images",
    out_dir="runs/spatial_inference_demo",
    cancer_context="oesophagus cancer",
    n_runs=5,
    iterations=1,
    tile_size=800,
    num_prompts=5,
    temperature=0.8,
    top_fraction=0.20,
    conch_repo="CONCH",
    conch_checkpoint="hf_hub:MahmoodLab/conch",
    skip_pubmed=True,
    skip_critic=True,
    max_patients=1,
    max_pathways=3,
)

manifest = run_spatial_inference_replicates(cfg)
print(manifest)
