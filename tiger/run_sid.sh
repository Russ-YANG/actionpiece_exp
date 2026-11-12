#!/usr/bin/env bash
# GRID step 3b: generate semantic IDs from an RQ-VAE checkpoint.
set -euo pipefail
DIM="${1:?usage: run_sid.sh <dim> <ckpt_path>}"
CKPT="${2:?usage: run_sid.sh <dim> <ckpt_path>}"
GRID_DIR="${GRID_DIR:-$HOME/science/GRID}"
cd "$GRID_DIR"
export PYTHONUNBUFFERED=1
./.venv/bin/python -m src.inference \
  experiment=rqvae_inference_flat \
  data_dir=data/amazon_data/beauty \
  embedding_path="$GRID_DIR/qwen_emb/d${DIM}/pickle/merged_predictions_tensor.pt" \
  embedding_dim="$DIM" num_hierarchies=3 codebook_width=256 \
  ckpt_path="$CKPT"
