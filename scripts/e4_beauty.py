"""Historical E4 with existing embeddings; isolate all generated artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "train"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    processed = ROOT / "cache/AmazonReviews2014/Beauty/processed"
    mapping = json.loads((processed / "id_mapping.json").read_text())
    sequences = json.loads((processed / "all_item_seqs.json").read_text())
    items = mapping["id2item"][1:]
    if any(mapping["item2id"].get(item) != i for i, item in enumerate(items, 1)):
        raise ValueError("Mapping row order mismatch")
    if any(len(seq) < 4 or any(item not in mapping["item2id"] for item in seq)
           for seq in sequences.values()) or not sequences:
        raise ValueError("Invalid sequences")
    embeddings = []
    for suffix in ("sent_emb", "image_emb"):
        matches = list(processed.glob("qwen_local.*d768.*." + suffix))
        if len(matches) != 1:
            raise ValueError("Expected exactly one local Qwen768 " + suffix)
        path = matches[0]
        if path.stat().st_size != len(items) * 768 * 4:
            raise ValueError("Embedding size mismatch: " + str(path))
        if not Path(str(path) + ".manifest.json").is_file():
            raise ValueError("Missing embedding source manifest")
        embeddings.append(path)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    print(json.dumps({"items": len(items), "users": len(sequences),
                      "embeddings": list(map(str, embeddings)),
                      "merge": True, "semantic_slots": 8, "seed": args.seed}, indent=2))
    if args.mode == "check":
        print("E4_DATA_CHECK_SUCCESS")
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training requires Slurm")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible CUDA device required")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    from genrec.pipeline import Pipeline
    import genrec.datasets as registry
    from genrec.dataset import AbstractDataset
    from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer

    # Keep historical model/tokenizer/loss; replace only raw-data acquisition.
    class CachedBeauty(AbstractDataset):
        def __init__(self, config):
            super().__init__(config)
            self.cache_dir = str(output / "cache/AmazonReviews2014/Beauty")
            self.category = "Beauty"
            self.id_mapping = mapping
            self.all_item_seqs = sequences

    class CachedEmbeddingsTokenizer(ActionPieceTokenizer):
        def _encode_sent_emb(self, *unused, **kwargs):
            raise RuntimeError("Missing cached embedding; regeneration forbidden")

        def _encode_image_emb(self, *unused, **kwargs):
            raise RuntimeError("Missing cached image embedding; regeneration forbidden")

    output.mkdir(parents=True)
    destination = output / "cache/AmazonReviews2014/Beauty/processed"
    destination.mkdir(parents=True)
    for path in embeddings:
        (destination / path.name).symlink_to(path)
        sidecar = Path(str(path) + ".manifest.json")
        (destination / sidecar.name).symlink_to(sidecar)
    registry.AmazonReviews2014 = CachedBeauty
    config = {"rand_seed": args.seed, "patience": 50,
              "run_id": f"beauty_e4_4plus4_seed{args.seed}",
              "cache_dir": str(output / "cache"),
              "log_dir": str(output / "logs"),
              "ckpt_dir": str(output / "checkpoints"),
              "tensorboard_log_dir": str(output / "tensorboard"),
              "result_path": str(output / "result.json")}
    pipeline = Pipeline("ActionPiece", "AmazonReviews2014",
                        tokenizer=CachedEmbeddingsTokenizer,
                        config_file=str(ROOT / "experiments/e4_qwen3_vl_8b_separate_opq_beauty_train.yaml"),
                        config_dict=config)
    (output / "manifest.json").write_text(json.dumps({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "config": {k: v for k, v in pipeline.config.items() if k not in ("accelerator", "device")},
        "embedding_sources": {p.name: json.loads(Path(str(p) + ".manifest.json").read_text()) for p in embeddings},
        "python": sys.version, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0)}, indent=2, default=str))
    pipeline.run()
    if not (output / "result.json").is_file():
        raise RuntimeError("Missing final results")
    print("E4_JOB_SUCCESS")


if __name__ == "__main__":
    main()
