"""Run one independent alignment config per Slurm task (one GPU each, no DDP).

OPQ_ALIGNMENT_CONFIGS is a comma-separated list; task rank r runs entry r.
The number of entries must equal SLURM_NTASKS. OPQ_ALIGNMENT_DRY_RUN=1 prints
the assignment and exits before CUDA/training (for checking on a login node).
"""
import json
import os
import runpy
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
configs = [c.strip() for c in os.environ["OPQ_ALIGNMENT_CONFIGS"].split(",") if c.strip()]
n_tasks = int(os.environ["SLURM_NTASKS"])
rank = int(os.environ["SLURM_PROCID"])
if len(configs) != n_tasks:
    raise RuntimeError(f"{len(configs)} configs for {n_tasks} Slurm tasks")
if not 0 <= rank < n_tasks:
    raise RuntimeError(f"Unexpected Slurm task rank: {rank}")
# Every task validates the whole set so that a bad list fails on all ranks.
outputs = []
for path in configs:
    raw = json.loads((ROOT / path).read_text())
    if "alignment_weight" not in raw:
        raise RuntimeError(f"Not an alignment config (no alignment_weight): {path}")
    outputs.append(raw["output_dir"])
if len(set(outputs)) != len(outputs):
    raise RuntimeError(f"Configs share an output_dir: {outputs}")
config = configs[rank]

# Independent models: prevent Accelerate from interpreting tasks as DDP workers.
for key in list(os.environ):
    if key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE") or key.startswith(("PMI_", "PMIX_", "OMPI_", "MV2_")):
        os.environ.pop(key, None)
os.environ["ACCELERATE_USE_CPU"] = "false"

# Select one GPU before PyTorch initializes CUDA.
original_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
devices = [value.strip() for value in original_visible.split(",") if value.strip()]
local_rank = int(os.environ["SLURM_LOCALID"])
if len(devices) == n_tasks and len(set(devices)) == n_tasks:
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
elif len(devices) != 1:
    raise RuntimeError(f"Unexpected GPU visibility: {original_visible!r}")
print(
    f"GPU_BIND task={rank} original={original_visible!r} "
    f"selected={os.environ['CUDA_VISIBLE_DEVICES']} config={config}",
    flush=True,
)
if os.environ.get("OPQ_ALIGNMENT_DRY_RUN") == "1":
    sys.exit(0)

import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"Expected exactly one visible CUDA device for {config}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
gpu = torch.cuda.get_device_name(0)
if "A100" not in gpu:
    raise RuntimeError(f"Expected A100, got {gpu}")

print(
    f"ALIGNMENT_START config={config} task={rank} node={socket.gethostname()} gpu={gpu}",
    flush=True,
)
# opq_alignment.py imports opq_baseline from scripts/; run_path does not add it.
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv = ["scripts/opq_alignment.py", "train", "--config", config]
runpy.run_path(str(ROOT / "scripts/opq_alignment.py"), run_name="__main__")
print(f"OPQ_ALIGNMENT_TASK_SUCCESS config={config}", flush=True)
