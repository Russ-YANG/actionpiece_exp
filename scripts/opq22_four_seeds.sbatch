#!/bin/bash
#SBATCH --job-name=opq22-4seeds
#SBATCH --account=p201605
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/opq_baseline/opq22-four-%j.out
#SBATCH --error=logs/opq_baseline/opq22-four-%j.out

set -euo pipefail
cd /project/home/p201605/u104824/actionpiece_exp
: "${OPQ_BASELINE_COMMIT:?Expected commit must be supplied}"
test "$(git rev-parse HEAD)" = "$OPQ_BASELINE_COMMIT"
test -z "$(git diff HEAD --name-only)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "job=$SLURM_JOB_ID commit=$OPQ_BASELINE_COMMIT seeds=2024,2025,2026,2027"

srun --ntasks=4 --cpus-per-task=8 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpu-bind=cores --mpi=none --kill-on-bad-exit=0 \
  --output=logs/opq_baseline/opq22-four-%j-task%t.out \
  --error=logs/opq_baseline/opq22-four-%j-task%t.out \
  /project/home/p201605/u104824/envs/opq_rec/bin/python -u -B \
  scripts/run_opq22_seed.py

echo OPQ_FOUR_SEEDS_SUCCESS
