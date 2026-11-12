#!/usr/bin/env bash
# GRID step 3a: train the RQ-VAE (3 layers x 256) on the Qwen embeddings of one width.
set -euo pipefail
DIM="${1:?usage: run_rqvae.sh <dim>}"
GRID_DIR="${GRID_DIR:-$HOME/science/GRID}"
cd "$GRID_DIR"
export PYTHONUNBUFFERED=1
./.venv/bin/python -m src.train \
  experiment=rqvae_train_flat \
  data_dir=data/amazon_data/beauty \
  embedding_path="$GRID_DIR/qwen_emb/d${DIM}/pickle/merged_predictions_tensor.pt" \
  embedding_dim="$DIM" num_hierarchies=3 codebook_width=256
