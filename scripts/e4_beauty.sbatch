#!/bin/bash
#SBATCH --job-name=e4-beauty
#SBATCH --account=p201605
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e4_beauty/train-%j.out
#SBATCH --error=logs/e4_beauty/train-%j.out
set -euo pipefail
cd /project/home/p201605/u104824/actionpiece_exp
: "${E4_COMMIT:?Supply reviewed Git commit}"
test "$(git rev-parse HEAD)" = "$E4_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
: "${CUDA_VISIBLE_DEVICES:?Slurm must supply GPU visibility}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "job=$SLURM_JOB_ID commit=$E4_COMMIT"
hostname
/project/home/p201605/u104824/envs/opq_rec/bin/python -u -B \
  scripts/e4_beauty.py train --output "results/e4_beauty/job_${SLURM_JOB_ID}"
