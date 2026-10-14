"""Run one independent OPQ2+OPQ2 seed per Slurm task."""
import os
import runpy
import socket
import sys

rank = int(os.environ["SLURM_PROCID"])
if not 0 <= rank < 4:
    raise RuntimeError(f"Unexpected Slurm task rank: {rank}")
seed = 2024 + rank

# Independent models: prevent Accelerate from interpreting tasks as DDP workers.
for key in list(os.environ):
    if key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE") or key.startswith(("PMI_", "PMIX_", "OMPI_", "MV2_")):
        os.environ.pop(key, None)
os.environ["ACCELERATE_USE_CPU"] = "false"

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"Expected exactly one visible CUDA device for seed {seed}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
gpu = torch.cuda.get_device_name(0)
if "A100" not in gpu:
    raise RuntimeError(f"Expected A100, got {gpu}")

print(
    f"SEED_START seed={seed} task={rank} node={socket.gethostname()} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} gpu={gpu}",
    flush=True,
)
sys.argv = [
    "scripts/opq_baseline.py", "train", "--config",
    f"experiments/opq_baseline_beauty_text2_image2_seed{seed}.json",
]
runpy.run_path("scripts/opq_baseline.py", run_name="__main__")
print(f"OPQ_SEED_SUCCESS seed={seed}", flush=True)
