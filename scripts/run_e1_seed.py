"""One independent E1 process per Slurm task; bind before importing torch."""
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
# E1_RUNS lists one codebooks:seed pair per task, e.g. "2:2024,2:2025,4:2024,4:2025".
runs = [tuple(map(int, run.split(":"))) for run in os.environ["E1_RUNS"].split(",")]
if len(runs) != 4:
    raise ValueError("E1_RUNS must list exactly four codebooks:seed pairs")
codebooks, seed = runs[rank]
print(f"E1_START codebooks={codebooks} seed={seed} task={rank} GPU={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
sys.argv = ["scripts/e1_beauty_codes.py", "train", "--seed", str(seed),
            "--codebooks", str(codebooks),
            "--output", f"results/e1_beauty_codes/job_{os.environ['SLURM_JOB_ID']}/text{codebooks}_seed{seed}"]
runpy.run_path("scripts/e1_beauty_codes.py", run_name="__main__")
print(f"E1_SEED_SUCCESS codebooks={codebooks} seed={seed}", flush=True)
