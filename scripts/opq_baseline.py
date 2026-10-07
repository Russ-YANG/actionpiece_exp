"""OPQ baseline: reuse backbone, batching and trainer; bypass merge."""
import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AtomicCodec:
    """Primitive-feature adapter. No merge vocabulary or merge operations."""

    def __init__(self, features, seed):
        self.n_categories = 5
        self.vocab = [(-1, -1)]
        for slot, size in enumerate((256, 256, 256, 256, 128)):
            self.vocab.extend((slot, value) for value in range(size))
        self.rank = {feature: token for token, feature in enumerate(self.vocab)}
        self.vocab_size = self.n_init_feats = len(self.vocab)
        self.catalog = {tuple(row) for row in features.values()}
        self.rng = random.Random(seed)

    def encode(self, states, shuffle="none"):
        if shuffle not in ("none", "feature"):
            raise ValueError("Only none/feature shuffle is supported")
        output = []
        for state in states:
            row = list(map(int, state))
            if shuffle == "feature":
                self.rng.shuffle(row)
            output.extend(row)
        return output

    def decode_single_state(self, tokens):
        if len(tokens) != self.n_categories:
            return None
        features = []
        for slot, token in enumerate(tokens):
            if not 0 < token < len(self.vocab):
                return None
            feature = self.vocab[token]
            if feature[0] != slot:
                return None
            features.append(feature)
        if tuple(value for _, value in features) not in self.catalog:
            return None
        return features


def load_inputs(config):
    processed = Path(config["processed_dir"])
    mapping = json.loads((processed / "id_mapping.json").read_text())
    sequences = json.loads((processed / "all_item_seqs.json").read_text())
    raw = json.loads(Path(config["sid_path"]).read_text())
    items = [item for item in mapping["item2id"] if item != "[PAD]"]

    if set(raw) != set(items):
        raise ValueError("SID and catalog item sets differ")
    for item, codes in raw.items():
        if len(codes) != 4 or any(
            type(c) is not int or not 0 <= c < 256 for c in codes
        ):
            raise ValueError(f"Invalid OPQ4 codes: {item}")
    if not sequences or any(len(seq) < 4 for seq in sequences.values()):
        raise ValueError("Sequences must support training and two held-out items")
    if any(item not in raw for seq in sequences.values() for item in seq):
        raise ValueError("Sequence references a missing catalog item")
    if any(
        mapping["id2item"][iid] != item
        for item, iid in mapping["item2id"].items()
    ):
        raise ValueError("item2id/id2item disagree")

    groups = collections.defaultdict(list)
    for item in items:
        groups[tuple(raw[item])].append(item)
    rng = random.Random(config["disambiguation_seed"])
    features = {}
    for sid in sorted(groups):
        members = sorted(groups[sid])
        if len(members) > 128:
            raise ValueError("Collision group exceeds 128 disambiguation values")
        values = rng.sample(range(128), len(members))
        for item, value in zip(members, values):
            features[item] = (*sid, value)
    if len(set(features.values())) != len(items):
        raise ValueError("Disambiguated labels are not unique")
    return mapping, sequences, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "smoke", "train"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raw_config = json.loads(Path(args.config).read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor",
         raw_config["base_commit"], commit],
        cwd=ROOT, capture_output=True,
    )
    if ancestry.returncode:
        raise RuntimeError("Current commit does not descend from the reviewed base")

    if args.mode == "train":
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "scripts/opq_baseline.py"],
            cwd=ROOT, capture_output=True,
        )
        changed = subprocess.check_output(
            ["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True
        ).strip()
        if tracked.returncode or changed:
            raise RuntimeError("Commit and review baseline code before full training")

    mapping, sequences, features = load_inputs(raw_config)
    codec = AtomicCodec(features, raw_config["rand_seed"])
    for row in features.values():
        tokens = [codec.rank[(slot, value)] for slot, value in enumerate(row)]
        assert codec.decode_single_state(tokens) == list(enumerate(row))
        assert [token - 1 for token in tokens[:4]] == [
            slot * 256 + row[slot] for slot in range(4)
        ]

    if args.mode == "check":
        print(json.dumps({
            "items": len(features),
            "users": len(sequences),
            "target_features": 5,
            "merge": False,
            "disambiguation_seed": raw_config["disambiguation_seed"],
            "loss": "full_vocabulary_cross_entropy",
            "base_commit": raw_config["base_commit"], "git_commit": commit,
            "status": "DATA_CHECK_SUCCESS",
        }, indent=2))
        return

    import os
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("smoke/train must run inside a Slurm allocation")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))

    from genrec.pipeline import Pipeline
    import genrec.datasets as dataset_registry
    import genrec.models as model_registry
    from genrec.dataset import AbstractDataset
    from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer
    from genrec.models.ActionPiece.model import ActionPiece
    from genrec.models.ActionPiece.core import ActionPieceCore

    def forbidden(*args, **kwargs):
        raise RuntimeError("MERGE_CORE_FORBIDDEN")

    ActionPieceCore.__init__ = forbidden
    ActionPieceCore.train = forbidden
    ActionPieceCore.from_pretrained = forbidden

    class CachedDataset(AbstractDataset):
        def __init__(self, config):
            super().__init__(config)
            self.cache_dir = str(Path(raw_config["processed_dir"]).parent)
            self.category = "Beauty"
            self.id_mapping = mapping
            self.all_item_seqs = sequences

    class OPQTokenizer(ActionPieceTokenizer):
        def _init_tokenizer(self, dataset):
            self.item2feat = features
            self._check_conflicts(features)
            return AtomicCodec(features, raw_config["rand_seed"])

    class OPQModel(ActionPiece):
        def forward(self, batch):
            return self.t5(**batch)

    # Process-local registration; repository modules are not edited.
    dataset_registry.AmazonReviews2014 = CachedDataset
    model_registry.ActionPiece = OPQModel
    config = dict(raw_config)
    config.update({
        "model": "OPQBaseline",
        "category": "Beauty",
        "target_tokenization": "atomic",
        "history_granularity": "legacy",
        "history_eval_granularity": None,
        "history_tokenization_scope": "sequence",
        "semantic_cache_seed": None,
        "length_control_history_tokens_per_item": 0,
        "length_control_target_tokens": 0,
        "beam_search_mode": "eos_aware",
        "tokenizer_only": args.mode == "smoke",
    })
    output = Path(raw_config["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)
    for key, name in (
        ("log_dir", "logs"),
        ("tensorboard_log_dir", "tensorboard"),
        ("ckpt_dir", "checkpoints"),
    ):
        config[key] = str(output / name)
    config["result_path"] = str(output / "result.json")

    pipeline = Pipeline(
        "ActionPiece", "AmazonReviews2014",
        tokenizer=OPQTokenizer, config_dict=config,
    )
    (output / "item.target_features.json").write_text(
        json.dumps(features) + "\n"
    )
    (output / "manifest.json").write_text(json.dumps({
        "base_commit": raw_config["base_commit"], "git_commit": commit,
        "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "opq_manifest": json.loads(
            Path(raw_config["sid_path"]).with_name("manifest.json").read_text()
        ),
        "parameters": raw_config,
        "resolved_config": {
            key: value for key, value in pipeline.config.items()
            if key not in ("accelerator", "device")
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "training_loss": "full_vocabulary_cross_entropy",
        "decoding": "slot_constrained_beam_with_EOS_and_catalog_validation",
        "ensemble": "existing_fork_rank_weighted_fusion",
        "token_mapping": "OPQ token = 1 + codebook_id * 256 + code_value",
        "special_tokens": {"PAD": 0, "BOS": 1153, "EOS": 1154},
    }, indent=2, default=str) + "\n")

    if args.mode == "train":
        pipeline.run()
        return

    model = OPQModel(
        pipeline.config, pipeline.raw_dataset, pipeline.tokenizer
    ).cuda()
    examples = []
    for sequence in list(sequences.values())[:2]:
        train = sequence[:-2]
        examples.append({
            "state_seq": pipeline.tokenizer._tokenize_once(train[-3:])
        })
    batch = pipeline.tokenizer.collate_fn_train(examples)
    batch = {key: value.cuda() for key, value in batch.items()}
    result = model(batch)
    assert torch.isfinite(result.loss).item()
    result.loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )

    model.eval()
    with torch.no_grad():
        generated = model.beam_search(
            batch["input_ids"], batch["attention_mask"],
            pipeline.tokenizer.generation_max_length,
            num_beams=2, num_return_sequences=1,
        )
    assert generated.shape[1] == pipeline.tokenizer.generation_max_length
    assert (generated[:, -1] == pipeline.tokenizer.eos_token).all().item()
    for tokens in generated[:, 1:-1].cpu().tolist():
        for slot, token in enumerate(tokens):
            assert token in pipeline.tokenizer.atomic_target_tokens[slot]

    examples_eval = [{
        "state_seq": pipeline.tokenizer._tokenize_once(sequence[-3:])
    } for sequence in list(sequences.values())[:2]]
    evaluation = pipeline.tokenizer.collate_fn_test(examples_eval)
    evaluation = {key: value.cuda() for key, value in evaluation.items()}
    with torch.no_grad():
        predictions = model.generate(evaluation, n_return_sequences=10)
    assert predictions.shape == (2, 10, 5)
    for tokens in predictions.cpu().reshape(-1, 5).tolist():
        if tokens[0] != -1:
            assert pipeline.tokenizer.actionpiece.decode_single_state(tokens) is not None

    print("smoke loss:", result.loss.item())
    print("generated token rows:", generated.cpu().tolist())
    pipeline.accelerator.end_training()
    print("OPQ_BASELINE_SMOKE_SUCCESS; merge core was forbidden")


if __name__ == "__main__":
    main()
