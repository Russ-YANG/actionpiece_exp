"""One independent E4 process per Slurm task; bind before importing torch."""
import os
import runpy
import sys

rank = int(os.environ["SLURM_PROCID"])
local_rank = int(os.environ["SLURM_LOCALID"])
if not 0 <= rank < 4 or not 0 <= local_rank < 4:
    raise RuntimeError("Expected four tasks on one node")
for key in list(os.environ):
    if key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE") or key.startswith(("PMI_", "PMIX_", "OMPI_", "MV2_")):
        os.environ.pop(key, None)
os.environ["ACCELERATE_USE_CPU"] = "false"
devices = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
if len(devices) == 4 and len(set(devices)) == 4:
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
elif len(devices) != 1:
    raise RuntimeError("Unexpected GPU visibility: " + repr(devices))
codebooks = int(os.environ.get("E4_CODEBOOKS", "4"))
if codebooks not in (2, 4):
    raise ValueError("E4_CODEBOOKS must be 2 or 4")
output_root = "results/e4_beauty_opq22" if codebooks == 2 else "results/e4_beauty"
seed = 2024 + rank
print(f"E4_START seed={seed} task={rank} GPU={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
sys.argv = ["scripts/e4_beauty.py", "train", "--seed", str(seed),
            "--codebooks-per-modality", str(codebooks),
            "--output", f"{output_root}/job_{os.environ['SLURM_JOB_ID']}/seed{seed}"]
runpy.run_path("scripts/e4_beauty.py", run_name="__main__")
print(f"E4_SEED_SUCCESS seed={seed}", flush=True)
