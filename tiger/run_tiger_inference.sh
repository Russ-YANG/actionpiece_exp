#!/usr/bin/env bash
# GRID step 4b: generate recommendations with a trained TIGER checkpoint.
set -euo pipefail
DIM="${1:?usage: run_tiger_inference.sh <dim> <semantic_id_path> <ckpt_path>}"
SID="${2:?}"
CKPT="${3:?}"
GRID_DIR="${GRID_DIR:-$HOME/science/GRID}"
cd "$GRID_DIR"
export PYTHONUNBUFFERED=1
./.venv/bin/python -m src.inference \
  experiment=tiger_inference_flat \
  data_dir=data/amazon_data/beauty \
  semantic_id_path="$SID" \
  ckpt_path="$CKPT" \
  num_hierarchies=4
