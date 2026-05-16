"""Example command for adapter/probe fine-tuning of CONCH on patch PNGs + hallmark CSV.
This file prints the command; run it manually after preparing real data.
"""
cmd = r"""
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
"""
print(cmd)
