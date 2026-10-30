# Beauty historical E4 preparation

Prepared only; no training submitted. Copy the two scripts into the same paths
in the local `opq-baseline` repository, review, commit and synchronize through
GitHub. No existing experiment configuration is changed.

Objective: run the existing ActionPiece E4 training configuration with separate
Qwen local 8B text/image 768D embeddings, OPQ4x256 per modality, and ActionPiece
merges. Keep the historical hash feature, sequence merges, target tokenization,
loss and decoder. No alignment loss or OPQ-only baseline adapter is used.
Seed 2024; batch 128/64; maximum 200 epochs; patience 50. This is one run,
not a four-seed effectiveness claim or guaranteed numerical paper reproduction.

The entry loads processed sequences/mapping rather than downloading raw data.
It creates a fresh per-job cache containing read-only embedding symlinks, then
lets the existing tokenizer fit OPQ and learn merges there. It refuses output
reuse and embedding generation. The old AutoDL model paths in the original
YAML are historical and are not needed when these cached embeddings load.
Existing embedding sidecars are recorded as provenance; `check` verifies file
size and mapping, not full embedding semantics or quantizer execution.

Resources: 1 requested GPU, 8 CPUs, 32 GiB RAM, 24 hours maximum. Actual runtime
unknown. GPU nodes previously used exclusive allocation: a one-GPU run may
occupy/bill an entire four-GPU node. Confirm policy before submitting; this
template does not promise shared-node utilization. No unrelated held job is touched.

## Server commands, after reviewed scripts have been committed and pushed

Each code block is one command (a multiline block is pasted as a whole).

```bash
cd /project/home/p201605/u104824/actionpiece_exp
```

```bash
git status --short
```

If there are changes, stop and preserve them before syncing. Fetch the reviewed
commit; replace REVIEWED_COMMIT with its full SHA. Keep branch history; do not reset.

```bash
git fetch origin opq-baseline
```

```bash
git merge --ff-only REVIEWED_COMMIT
```

```bash
git rev-parse HEAD
```

```bash
/project/home/p201605/u104824/envs/opq_rec/bin/python -B scripts/e4_beauty.py check --output results/e4_beauty/preflight_unused
```

This does not fit OPQ, learn merges or initialize CUDA. Dependencies and actual
tokenizer execution remain unverified until a compute-node run. Do not claim
this check is a full smoke test.

Only after approving the run/resource budget:

```bash
mkdir -p logs/e4_beauty
```

```bash
E4_JOB=$(sbatch --parsable --export=ALL,E4_COMMIT="$(git rev-parse HEAD)" scripts/e4_beauty.sbatch)
```

```bash
tail -F "logs/e4_beauty/train-${E4_JOB}.out"
```

```bash
sacct -j "$E4_JOB" --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

Check `E4_JOB_SUCCESS`, `results/e4_beauty/job_JOBID/result.json`, checkpoint,
manifest and the isolated tokenizer cache. On errors, return the traceback and
job status; preserve failed outputs. Missing dependencies must be resolved in
the environment deliberately. Missing cache must not trigger new embeddings.
Historical metric comparisons require the canonical Encoder/truncation Sheets;
no paper score is asserted here.

## Four-seed parallel variant (selected)

Use `scripts/e4_beauty_four_seeds.sbatch` instead of the single-seed template.
One node, four GPUs, 32 CPUs total (8/task), 128 GiB total, maximum 48 hours.
Seeds 2024--2027 each run a separate ActionPiece pipeline and isolated cache.
This changes the pipeline seed (hash/merge/training RNG), not just model
initialization. Faiss clustering keeps the historical implementation defaults;
no claim is made that all Faiss clustering RNGs are controlled by `--seed`.
The four runs independently fit OPQ, so this is not a frozen-SID model-only
seed comparison. Avoid merging these results with the earlier frozen OPQ-only
baseline as though the entire experimental protocol were identical.

The task wrapper binds one visible GPU before torch import and removes DDP
rank variables, using the same approach as prior working OPQ jobs.

```bash
E4_JOB=$(sbatch --parsable --export=ALL,E4_COMMIT="$(git rev-parse HEAD)" scripts/e4_beauty_four_seeds.sbatch)
```

```bash
tail -F "logs/e4_beauty/four-${E4_JOB}-task0.out"
```

Tasks 0--3 map to seeds 2024--2027. Each must print E4_SEED_SUCCESS; the parent
must print E4_FOUR_SEEDS_SUCCESS and exit 0:0. Results are under
`results/e4_beauty/job_JOBID/seedSEED/`. A failing task does not kill its siblings.
Local syntax and eight mocked GPU/seed-isolation cases passed. No real CUDA,
Faiss, merge, or full training execution has been validated for this new entry.

Four-seed rerun: time limit raised to 48 hours; 200 epochs maximum and patience 50 retained. Starts from scratch; previous outputs retained. Epoch-61 slowdown from job 5392122 remains unexplained. Checkpoints still contain model weights only, without full training-state resume.

## E4 OPQ2+OPQ2 with merges

Use `scripts/e4_beauty_opq22_four_seeds.sbatch`; four seeds 2024--2027,
4 GPUs, 32 requested CPUs, 128 GiB, 48 hours, 200 epochs maximum and patience 50.
Uses the existing `e4_qwen3_vl_8b_separate_opq2x256_beauty_train.yaml`.
All other model/merge settings retain that historical configuration.
The shared entry defaults to 4+4; this launcher explicitly selects 2+2.
Outputs: `results/e4_beauty_opq22/job_JOBID/seedSEED/`;
logs: `logs/e4_beauty_opq22/four-JOBID-taskRANK.out`.
Each seed independently fits OPQ and learns merges in its isolated cache.
Existing global OPQ2 caches are not copied into these runs.
This is an ActionPiece E4 experiment, not the no-merge OPQ baseline.
No alignment loss. Full training-state resume is still unsupported.
