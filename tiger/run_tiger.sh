#!/usr/bin/env bash
# GRID step 4a: train TIGER on the generated semantic IDs.
# The override block below is REQUIRED -- see README section 4.
set -euo pipefail
DIM="${1:?usage: run_tiger.sh <dim> <semantic_id_path> [extra hydra overrides...]}"
SID="${2:?usage: run_tiger.sh <dim> <semantic_id_path> [extra hydra overrides...]}"
shift 2
GRID_DIR="${GRID_DIR:-$HOME/science/GRID}"
cd "$GRID_DIR"
export PYTHONUNBUFFERED=1
./.venv/bin/python -m src.train \
  experiment=tiger_train_flat \
  data_dir=data/amazon_data/beauty \
  semantic_id_path="$SID" \
  num_hierarchies=4 \
  trainer.devices=1 \
  data_loading.train_dataloader_config.dataloader.num_workers=1 \
  data_loading.val_dataloader_config.dataloader.num_workers=1 \
  data_loading.test_dataloader_config.dataloader.num_workers=1 \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.every_n_train_steps=200 \
  ++model.should_check_prefix=false \
  extras.print_config=false \
  ++should_skip_retry=True \
  "$@"
