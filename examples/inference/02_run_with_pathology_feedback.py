"""Example: run more than one iteration and enter pathology feedback after iteration 1."""
from spi_wsi_aiinmed.keys import ask_user_for_keys
from spi_wsi_aiinmed.spatial_inference import SpatialInferenceConfig, run_spatial_inference_replicates

ask_user_for_keys(default_entrez_email="your.email@example.com")

cfg = SpatialInferenceConfig(
    scores_csv="examples/data/hallmark_patient_scores.csv",
    image_folder="examples/data/histology_images",
    out_dir="runs/spatial_inference_feedback_demo",
    cancer_context="oesophagus cancer",
    n_runs=3,
    iterations=2,
    interactive_pathology_feedback=True,
    tile_size=800,
    num_prompts=5,
    temperature=0.8,
    top_fraction=0.20,
    conch_repo="CONCH",
    skip_pubmed=True,
    skip_critic=True,
    max_patients=1,
    max_pathways=3,
)

run_spatial_inference_replicates(cfg)
