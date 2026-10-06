#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# account / partition / qos MUST be specified explicitly on sbatch command line.
# Smoke overrides: --cpus-per-task=1 --mem=2G --time=00:05:00.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm on a compute node}"
: "${SLURM_CPUS_PER_TASK:?Missing allocated CPU count}"
: "${OPQ_PYTHON:?Set absolute environment Python path before submission}"
: "${OPQ_COMMIT:?Set exact reviewed Git commit}"
test "$(git rev-parse HEAD)" = "$OPQ_COMMIT"
git diff --quiet
git diff --cached --quiet
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
hostname
date -u
printf 'job=%s commit=%s CPUs=%s\n' "$SLURM_JOB_ID" "$OPQ_COMMIT" "$SLURM_CPUS_PER_TASK"
case "${1:-}" in
  smoke)
    "$OPQ_PYTHON" scripts/opq_only_smoke.py --output-root "results/opq_only/smoke_${SLURM_JOB_ID}"
    ;;
  fit)
    : "${2:?Specify JSON config}"
    "$OPQ_PYTHON" scripts/opq_only.py preflight --config "$2"
    "$OPQ_PYTHON" scripts/opq_only.py fit --config "$2"
    "$OPQ_PYTHON" scripts/opq_only.py validate --config "$2"
    ;;
  *) printf '%s\n' 'Usage: job.sbatch smoke | fit CONFIG' >&2; exit 2 ;;
esac
printf '%s\n' 'OPQ_JOB_SUCCESS'
