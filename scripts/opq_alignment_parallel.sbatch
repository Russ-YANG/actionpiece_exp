#!/bin/bash
# Independent runs, one GPU per config (not DDP). Default = stage 1 lambda screen.
# Submit with e.g.:
#   mkdir -p logs/opq_alignment
#   OPQ_BASELINE_COMMIT=$(git rev-parse HEAD) sbatch scripts/opq_alignment_parallel.sbatch
# Other config sets: export OPQ_ALIGNMENT_CONFIGS=a.json,b.json,... and override
# --ntasks/--gres on the sbatch command line to the same count.
#SBATCH --job-name=opq-align-par
#SBATCH --account=p201605
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/opq_alignment/par-%j.out
#SBATCH --error=logs/opq_alignment/par-%j.out

set -euo pipefail
cd /project/home/p201605/u104824/actionpiece_exp
: "${OPQ_BASELINE_COMMIT:?Expected commit must be supplied at submission}"
test "$(git rev-parse HEAD)" = "$OPQ_BASELINE_COMMIT"
test -z "$(git diff HEAD --name-only)"
test -z "$(git ls-files --others --exclude-standard -- genrec scripts)"

export OPQ_ALIGNMENT_CONFIGS="${OPQ_ALIGNMENT_CONFIGS:-experiments/opq_alignment_beauty_seed2024_lambda0.json,experiments/opq_alignment_beauty_seed2024_lambda01.json,experiments/opq_alignment_beauty_seed2024_lambda02.json}"
n_configs=$(tr ',' '\n' <<<"$OPQ_ALIGNMENT_CONFIGS" | grep -c .)
test "$n_configs" = "$SLURM_NTASKS"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

hostname
date -u
echo "job=$SLURM_JOB_ID commit=$OPQ_BASELINE_COMMIT tasks=$SLURM_NTASKS configs=$OPQ_ALIGNMENT_CONFIGS"

# --kill-on-bad-exit=0: one failed config does not stop the others.
srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-task=1 \
  --gpu-bind=single:1 --cpu-bind=cores --mpi=none --kill-on-bad-exit=0 \
  --output=logs/opq_alignment/par-%j-task%t.out \
  --error=logs/opq_alignment/par-%j-task%t.out \
  /project/home/p201605/u104824/envs/opq_rec/bin/python -u -B \
  scripts/run_opq_alignment_task.py

echo OPQ_ALIGNMENT_PARALLEL_SUCCESS
