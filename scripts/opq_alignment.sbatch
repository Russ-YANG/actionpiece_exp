#!/bin/bash
#SBATCH --job-name=opq-align
#SBATCH --account=p201605
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/opq_alignment/train-%j.out
#SBATCH --error=logs/opq_alignment/train-%j.out

set -euo pipefail
cd /project/home/p201605/u104824/actionpiece_exp
: "${OPQ_BASELINE_COMMIT:?Expected commit must be supplied at submission}"
: "${OPQ_ALIGNMENT_CONFIG:?Set experiments/opq_alignment_beauty_seed2024_lambda0.json, lambda01.json or lambda02.json}"
test "$(git rev-parse HEAD)" = "$OPQ_BASELINE_COMMIT"
test -z "$(git diff HEAD --name-only)"
test -z "$(git ls-files --others --exclude-standard -- genrec scripts)"

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
  scripts/opq_alignment.py train \
  --config "$OPQ_ALIGNMENT_CONFIG"

echo OPQ_ALIGNMENT_JOB_SUCCESS
