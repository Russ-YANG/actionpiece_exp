#!/bin/bash
#SBATCH --job-name=opq-t2i2
#SBATCH --account=p201605
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/opq_baseline/train-text2-image2-%j.out
#SBATCH --error=logs/opq_baseline/train-text2-image2-%j.out

set -euo pipefail
cd /project/home/p201605/u104824/actionpiece_exp
: "${OPQ_BASELINE_COMMIT:?Expected commit must be supplied at submission}"
test "$(git rev-parse HEAD)" = "$OPQ_BASELINE_COMMIT"
test -z "$(git diff HEAD --name-only)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

hostname
date -u
echo "job=$SLURM_JOB_ID commit=$OPQ_BASELINE_COMMIT CPUs=$SLURM_CPUS_PER_TASK"

/project/home/p201605/u104824/envs/opq_rec/bin/python -u -B \
  scripts/opq_baseline.py train \
  --config experiments/opq_baseline_beauty_text2_image2_seed2024.json

echo OPQ_BASELINE_JOB_SUCCESS
