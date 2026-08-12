# NineRec data adapter

The adapter converts an official NineRec subset into the same processed cache
contract that ActionPiece uses for Amazon. It does not download or redistribute
NineRec data.

## Prepare DY

Point `--source-dir` at a directory containing:

```text
DY_behaviour.tsv       # preferred when present
DY_pair.csv            # used when behaviour is absent
DY_item.csv
DY_cover/
  <item_id>.jpg
```

Run:

```bash
python scripts/prepare_ninerec_data.py \
  --subset DY \
  --source-dir /path/to/official/DY \
  --text-language bilingual \
  --require-all-images
```

The default pair-column convention is `item,user,timestamp`, matching the
official NineRec `get_behaviour.py`. For an archive with the README-style
`user,item,timestamp` layout, add `--pair-order user-item-time` explicitly.
When both behaviour and pair files exist, the already ordered behaviour file
wins.

Outputs are written below `cache/NineRec/DY/processed/`:

```text
all_item_seqs.json
id_mapping.json
item_sources.json
metadata.qwen_text.json
metadata.qwen_multimodal.json
metadata.qwen_separate.json
metadata.qwen_fused.json
image_download_manifest.jsonl
ninerec_summary.json
```

`ninerec_summary.json` records counts, missing covers, and the exact image
manifest SHA-256. Copy that hash into the E3 and E5 experiment configs before
embedding generation. This prevents images from being silently changed or
reordered between runs.

## Run the dataset through ActionPiece

The dataset is selected with `--dataset NineRec`:

```bash
python main.py \
  --dataset NineRec \
  --model ActionPiece \
  --config_file experiments/e1_qwen3_vl_8b_text_dy.yaml
```

Prepared experiment configs are provided for:

- `experiments/e1_qwen3_vl_8b_text_dy.yaml`
- `experiments/e3_qwen3_vl_8b_native_multimodal_dy.yaml`
- `experiments/e5_qwen3_vl_8b_separate_fused_dy.yaml`

Generate embeddings in the CUDA/Qwen environment with the corresponding
scripts:

```bash
python scripts/generate_qwen_local_text_embeddings.py \
  --config-file experiments/e1_qwen3_vl_8b_text_dy.yaml

python scripts/generate_qwen_local_multimodal_embeddings.py \
  --config-file experiments/e3_qwen3_vl_8b_native_multimodal_dy.yaml

python scripts/generate_qwen_local_text_embeddings.py \
  --config-file experiments/e5_qwen3_vl_8b_separate_fused_dy.yaml

python scripts/generate_qwen_local_image_embeddings.py \
  --config-file experiments/e5_qwen3_vl_8b_separate_fused_dy.yaml
```

The DY configs retain the original E1/E3/E5 OPQ4 setup. Override
`pq_n_codebooks=32` only for an explicitly named OPQ32 experiment and keep its
artifacts separate from OPQ4.
