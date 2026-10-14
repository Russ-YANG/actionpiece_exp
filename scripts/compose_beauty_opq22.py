"""Compose historical Beauty OPQ2 + OPQ2 maps; no fitting or merges."""
import hashlib
import json
import subprocess
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = Path("cache/AmazonReviews2014/Beauty/processed")
    out = Path("results/opq_only/beauty_text2_image2_seed2024")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")

    mapping = json.loads((p / "id_mapping.json").read_text())
    items = mapping["id2item"][1:]
    assert len(items) == len(set(items)) == 12101
    assert all(mapping["item2id"][item] == row + 1
               for row, item in enumerate(items))
    seqs = json.loads((p / "all_item_seqs.json").read_text())
    assert all(item in mapping["item2id"] and item != "[PAD]"
               for seq in seqs.values() for item in seq)
    fit = {item for seq in seqs.values() for item in seq[:-2]}
    assert len(fit) == 12068

    mapping_sha = sha(p / "id_mapping.json")
    seq_sha = sha(p / "all_item_seqs.json")
    order_sha = hashlib.sha256("\n".join(items).encode()).hexdigest()
    maps, sources = [], []
    expected = (
        ("text", "298bea8f84eff6509d090c9c345c58d126890c67e51695ec309ababb06e03a69"),
        ("image", "9c1299e52f54fb6601cb12a0849def672d37d1bc6bdf4de793b5e33e45750f24"),
    )
    candidates = list(p.glob("qwen_local*.opq2x256.seed2024.sem_ids"))

    for modality, expected_sha in expected:
        found = [
            f for f in candidates
            if (".image." in f.name) == (modality == "image")
        ]
        assert len(found) == 1, f"Expected one {modality} OPQ2 map"
        path = found[0]
        meta = json.loads(Path(str(path) + ".manifest.json").read_text())
        assert sha(path) == meta["sha256"] == expected_sha
        assert meta["id_mapping_sha256"] == mapping_sha
        assert meta["item_seqs_sha256"] == seq_sha
        assert meta["training_item_count"] == len(fit)
        assert meta["item_count"] == len(items)
        assert (
            meta["dimension"], meta["n_codebooks"],
            meta["codebook_size"], meta["seed"]
        ) == (768, 2, 256, 2024)

        emb = Path(meta["embedding_path"])
        assert sha(emb) == meta["embedding_sha256"]
        embmeta = json.loads(Path(str(emb) + ".manifest.json").read_text())
        assert embmeta["model_id"] == "Qwen/Qwen3-VL-Embedding-8B"
        assert embmeta["item_order_sha256"] == order_sha
        assert embmeta["dimension"] == 768
        assert embmeta["item_count"] == len(items)
        assert embmeta["backend"] == (
            "qwen_local_image" if modality == "image" else "qwen_local"
        )

        codes = json.loads(path.read_text())
        assert set(codes) == set(items)
        assert all(
            len(row) == 2
            and all(type(v) is int and 0 <= v < 256 for v in row)
            for row in codes.values()
        )
        maps.append(codes)
        sources.append({
            "modality": modality, "sid_path": str(path),
            "opq_manifest": meta, "embedding_manifest": embmeta,
        })

    sids = {item: maps[0][item] + maps[1][item] for item in items}
    tokens = {
        item: [book * 256 + value for book, value in enumerate(row)]
        for item, row in sids.items()
    }
    for item in items:
        restored = dict(divmod(token, 256) for token in reversed(tokens[item]))
        assert [restored[book] for book in range(4)] == sids[item]

    reverse = {}
    for item, row in sids.items():
        reverse.setdefault(",".join(map(str, row)), []).append(item)
    groups = {k: v for k, v in reverse.items() if len(v) > 1}
    stats = {
        "item_count": len(items),
        "unique_sid_count": len(reverse),
        "collision_group_count": len(groups),
        "duplicate_item_count": sum(len(v) - 1 for v in groups.values()),
        "affected_item_count": sum(len(v) for v in groups.values()),
        "largest_group": max(map(len, reverse.values())),
    }
    assert stats["largest_group"] <= 128

    manifest = {
        "mode": "composition_of_verified_historical_opq2_maps",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "codebook_layout": ["text_0", "text_1", "image_0", "image_1"],
        "codebook_size": 256,
        "sources": sources,
        "source_training_rule": "leave_one_out_sequence_minus_last_two",
        "source_fit_count_recorded": len(fit),
        "encoded_item_count": len(items),
        "preprocessing_applied": "none",
        "token_mapping": "codebook_id*256+code_value; model adds 1 for PAD",
        "quantizers_available": False,
        "quantizer_reload_verified": False,
        "historical_provenance_limit":
            "Original fit/reload cannot be replayed without saved quantizers",
        "collision_stats": stats,
    }

    out.mkdir(parents=True, exist_ok=False)
    files = {
        "item.sem_ids": sids,
        "item.tokens.json": tokens,
        "sid_to_items.json": reverse,
        "collisions.json": dict(stats, groups=groups),
        "manifest.json": manifest,
    }
    for name, value in files.items():
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"output": str(out), **stats}, indent=2))

if __name__ == "__main__":
    main()
