#!/bin/bash
#SBATCH --job-name=e4-opq22
#SBATCH --account=p201605
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/e4_beauty_opq22/four-%j.out
#SBATCH --error=logs/e4_beauty_opq22/four-%j.out
set -euo pipefail
cd /project/home/p201605/u104824/actionpiece_exp
: "${E4_COMMIT:?Supply reviewed Git commit}"
test "$(git rev-parse HEAD)" = "$E4_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export E4_CODEBOOKS=2
echo "job=$SLURM_JOB_ID commit=$E4_COMMIT seeds=2024,2025,2026,2027 E4=text2_image2_with_merge"
srun --ntasks=4 --cpus-per-task=8 --gpus-per-task=1 \
  --gpu-bind=single:1 --cpu-bind=cores --mpi=none --kill-on-bad-exit=0 \
  --output=logs/e4_beauty_opq22/four-%j-task%t.out \
  --error=logs/e4_beauty_opq22/four-%j-task%t.out \
  /project/home/p201605/u104824/envs/opq_rec/bin/python -u -B scripts/run_e4_seed.py
echo E4_FOUR_SEEDS_SUCCESS
