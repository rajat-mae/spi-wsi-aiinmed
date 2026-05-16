# SPI-WSI-AIinMED

Reusable tools for H&E whole-slide image pathway-prompt inference, stochastic LLM robustness analysis, CONCH adapter/probe fine-tuning, and prompt specificity testing.

The repository is designed for notebook-first computational pathology experiments where pathway scores, whole-slide/patch images, and LLM-generated pathology prompts are used to spatially infer pathway-associated regions on H&E images.

## What this package does

1. **Spatial pathway inference from user inputs**
   - Reads a pathway-score CSV.
   - Reads matching histology images from a local folder.
   - Uses an LLM to generate pathway-specific pathology prompts.
   - Uses CONCH image/text embeddings to score image tiles against generated prompts.
   - Saves tile maps, pathway heatmaps, prompt tables, cosine tables, and spatial pathway score tables.

2. **Explicit stochastic LLM robustness testing**
   - Runs the same input through the LLM `N` independent times.
   - Saves each run separately as `run_000`, `run_001`, `run_002`, etc.
   - Quantifies run-to-run variability using:
     - spatial Pearson correlation,
     - spatial Spearman correlation,
     - selected-tile Jaccard overlap,
     - selected-tile Dice overlap,
     - prompt-token overlap,
     - selection-frequency maps.

3. **Human/pathology feedback after iteration 1**
   - If `iterations > 1`, the package can ask for pathology feedback after each iteration.
   - The feedback is injected into the next prompt-generation iteration.
   - This keeps the human-in-the-loop structure while preserving reproducibility through saved per-run outputs.

4. **CONCH fine-tuning**
   - Includes an adapter/probe training workflow for CONCH patch images and hallmark pathway labels.

5. **Prompt specificity testing**
   - Tests whether pathway prompts behave specifically for their intended pathway compared with other pathway prompts on the same tile.

## Repository layout

```text
spi-wsi-aiinmed/
  src/spi_wsi_aiinmed/
    spatial_inference.py          # repeated LLM inference + spatial pathway maps
    prompt_robustness.py          # overlap/sensitivity metrics across LLM runs
    specificity_testing.py        # pathway specificity testing
    train_conch_patch_hallmark.py # CONCH baseline/adapters fine-tuning
    keys.py                       # notebook-safe key entry
  examples/
    inference/                    # separate runnable inference examples
    notebooks/                    # original example notebooks, copied as provided
    data/                         # data instructions only
  docs/
  tests/
  pyproject.toml
  requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/spi-wsi-aiinmed.git
cd spi-wsi-aiinmed
```

### 2. Create a Python environment

```bash
conda create -n spi-wsi-aiinmed python=3.10 -y
conda activate spi-wsi-aiinmed
python -m pip install --upgrade pip
pip install -e ".[notebooks]"
```

### 3. Install CONCH

Clone CONCH next to this repository or provide its path in the config:

```bash
git clone https://github.com/mahmoodlab/CONCH.git
pip install -e CONCH
```

## API keys: Option A only

The example workflow asks the user for keys at runtime. Keys are not stored in notebooks, scripts, JSON files, or the repository.

In a notebook or Python script:

```python
from spi_wsi_aiinmed.keys import ask_user_for_keys
ask_user_for_keys(default_entrez_email="your.email@example.com")
```

You will be asked for:

```text
ANTHROPIC_API_KEY   required for Claude prompt generation
HF_TOKEN            optional/required depending on CONCH access
ENTREZ_EMAIL        optional if PubMed critic mode is used
```

## Download example data

Large image data are not stored in GitHub. Download the public example data here:

https://drive.google.com/file/d/1Hz-rnqrIV9KZvy3lxO_R5Awv2GaB6OKM/view

Place the downloaded data under:

```text
examples/data/
```

Recommended structure:

```text
examples/data/
  hallmark_patient_scores.csv
  histology_images/
    SAMPLE001.tif
    SAMPLE002.tif
  sample001/
    patches/
      AAACAAGTATCTCCCA-1.png
      AAACACCAATAACTGC-1.png
    hallmark_pathways.csv
```

## Input format for spatial pathway inference

The score CSV follows the current notebook convention:

```python
scores_df = pd.read_csv(SCORES_CSV, index_col=0).T
```

Therefore, the CSV should look like:

```text
Pathway,SAMPLE001,SAMPLE002
HALLMARK_E2F_TARGETS,0.42,-0.15
HALLMARK_G2M_CHECKPOINT,0.61,0.20
HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION,-0.10,0.55
```

Histology images should be named by sample ID:

```text
examples/data/histology_images/SAMPLE001.tif
examples/data/histology_images/SAMPLE002.tif
```

Supported image extensions include `.tif`, `.tiff`, `.ome.tif`, `.ome.tiff`, `.png`, `.jpg`, and `.jpeg`.

## Run spatial pathway inference with repeated LLM runs

Python example:

```python
from spi_wsi_aiinmed.keys import ask_user_for_keys
from spi_wsi_aiinmed.spatial_inference import SpatialInferenceConfig, run_spatial_inference_replicates

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
```

Command-line equivalent:

```bash
spi-infer-spatial-pathways \
  --scores_csv examples/data/hallmark_patient_scores.csv \
  --image_folder examples/data/histology_images \
  --out_dir runs/spatial_inference_demo \
  --cancer_context "oesophagus cancer" \
  --n_runs 5 \
  --iterations 1 \
  --tile_size 800 \
  --num_prompts 5 \
  --temperature 0.8 \
  --top_fraction 0.20 \
  --conch_repo CONCH \
  --skip_pubmed \
  --skip_critic \
  --max_patients 1 \
  --max_pathways 3 \
  --prompt_for_keys
```

## Run with pathology feedback after iteration 1

Set `iterations > 1` and keep `interactive_pathology_feedback=True`:

```python
cfg.iterations = 2
cfg.interactive_pathology_feedback = True
manifest = run_spatial_inference_replicates(cfg)
```

After iteration 1, the notebook/terminal asks:

```text
Enter pathology guidance for SAMPLE001 after iteration 1, or press Enter to skip:
>
```

That feedback is used in the next LLM prompt-generation iteration.

## Output structure

```text
runs/spatial_inference_demo/
  config.json
  replicate_manifest.json
  run_000/
    generated_prompts.csv
    tile_prompt_cosine.csv
    tile_pathway_scores.csv
    tilemaps/
    heatmaps/
  run_001/
    ...
  robustness/
    normalised_replicate_tile_scores.csv
    pairwise_replicate_metrics.csv
    summary_replicate_metrics.csv
    prompt_consistency_map.csv
    prompt_robustness_manifest.json
    pairwise_spatial_pearson_hist.png
    pairwise_selected_jaccard_hist.png
    selection_frequency_hist.png
    value_std_hist.png
```

## Interpreting robustness outputs

Use:

```text
robustness/summary_replicate_metrics.csv
```

Key columns:

```text
spatial_pearson_r_mean       stability of spatial score maps across LLM runs
spatial_spearman_rho_mean    stability of tile ranking across LLM runs
selected_jaccard_mean        overlap of selected top pathway tiles
selected_dice_mean           Dice overlap of selected top pathway tiles
mean_row_prompt_token_jaccard_mean   local prompt wording overlap
corpus_prompt_token_jaccard_mean     overall wording overlap
```

Use:

```text
robustness/prompt_consistency_map.csv
```

to identify unstable sample/pathway/tile combinations using:

```text
selection_frequency
value_std
value_cv
n_unique_prompt_texts
```

## Run specificity testing

After inference, run:

```bash
spi-test-specificity \
  --cosine_csv runs/spatial_inference_demo/run_000/tile_prompt_cosine.csv \
  --out_dir runs/spatial_inference_demo/run_000/specificity
```

Outputs:

```text
specificity_by_tile_pathway.csv
specificity_summary.csv
specificity_manifest.json
```

Interpretation:

```text
specificity_margin_mean > 0     intended pathway prompts score higher than decoy pathway prompts
mean_rank_percentile near 1     intended pathway ranks near the top on the tile
frac_top_rank high              pathway prompt is most specific for many tiles
```

## CONCH fine-tuning

The fine-tuning command trains:

1. frozen CONCH encoder + supervised probe head;
2. CONCH encoder with lightweight residual adapters + supervised probe head.

Expected inputs:

```text
examples/data/sample001/patches/
  AAACAAGTATCTCCCA-1.png
  AAACACCAATAACTGC-1.png
examples/data/sample001/hallmark_pathways.csv
```

Run:

```bash
spi-train-conch \
  --patch_dir examples/data/sample001/patches \
  --pathway_csv examples/data/sample001/hallmark_pathways.csv \
  --sample_id sample001 \
  --work_dir runs/conch_finetune_sample001 \
  --conch_repo CONCH \
  --conch_checkpoint hf_hub:MahmoodLab/conch \
  --batch_size 8 \
  --num_workers 0 \
  --no_amp
```

## Example notebooks

The original example notebooks are included as provided:

```text
examples/notebooks/Agent_SPI_WSI_AIinMED.ipynb
examples/notebooks/Agent_SPI_WSI_AIinMed_ST_validation_HEST_Prostate.ipynb
```

A minimal notebook using the runtime key prompt is also provided:

```text
examples/inference/notebook_spatial_inference_option_a.ipynb
```

## Safety and privacy

Do not commit:

```text
API keys
patient data
private WSI files
large TIFF/SVS/OME-TIFF data
model checkpoints
run outputs
```

The `.gitignore` excludes common private and large-output file types by default.

## Citation

If this package supports a manuscript, cite the repository and the underlying model/tool dependencies used in your experiment, including CONCH and the LLM provider.
