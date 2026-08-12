# ActionPiece Multimodal Experiments

This repository is an experimental extension of
[ActionPiece](https://arxiv.org/abs/2502.13581) for multimodal generative
recommendation. The current work evaluates text, image, and fused product
representations on Amazon Reviews 2014, starting with the Beauty category.

## Current experiment

E1 replaces the original Sentence-T5 text encoder with the open-weight
`Qwen/Qwen3-VL-Embedding-8B` model. It produces 768-dimensional normalized text
embeddings, converts them into four OPQ/PQ codes, and constructs a 40,000-token
ActionPiece vocabulary.

The workflow has two stages:

1. Build and freeze the embeddings, semantic IDs, item features, tokenizer,
   and merge log.
2. Train and evaluate the recommendation model from those cached artifacts.

## Setup

```bash
conda create -n actionpiece python=3.11 -y
conda activate actionpiece
pip install -r requirements.txt
```

The local E1 encoding stage also requires the official
[Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) package and
the `Qwen/Qwen3-VL-Embedding-8B` checkpoint.

## Run E1

Build the tokenizer artifacts:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config_file=experiments/e1_qwen3_vl_8b_text_beauty.yaml
```

Train for 200 epochs from the frozen cache:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config_file=experiments/e1_qwen3_vl_8b_text_beauty.yaml \
  --tokenizer_only=False \
  --epochs=200 \
  --eval_batch_size=64 \
  --run_id=beauty_e1_qwen3_vl_8b_train
```

## Run E4

E4 reuses the aligned 768D Qwen text and image embedding caches, trains a
separate OPQ4 index for each modality, concatenates the resulting codes into
eight slots, and then learns the ActionPiece vocabulary. Build only the frozen
tokenizer artifacts with:

```bash
python main.py \
  --config_file=experiments/e4_qwen3_vl_8b_separate_opq_beauty.yaml
```

After the tokenizer-only run completes, summarize text-text, image-image, and
cross-modal merges (including frequency weights and merge-stage trends) with:

```bash
python scripts/analyze_modality_merges.py \
  cache/AmazonReviews2014/Beauty/processed/actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.t768.th53fe43ac7a78.i768.ihe8d8f4d4b3a1.topq4x256.iopq4x256.seed2024.h128.v40000.merge_log.jsonl \
  --text-slots 4 \
  --image-slots 4 \
  --hash-slot 8 \
  --bins 10 \
  --json-output results/e4_merge_modality_summary.json
```

Train the recommendation model only from the completed frozen E4 cache:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config_file=experiments/e4_qwen3_vl_8b_separate_opq_beauty_train.yaml
```

Generated datasets, model weights, embeddings, tokenizers, checkpoints, and
logs are intentionally excluded from Git. A transferred E1 cache must retain
its original directory structure under
`cache/AmazonReviews2014/Beauty/processed/`; do not rebuild or mix tokenizer
artifacts from a different encoder configuration.

## Planned comparisons

- Text-only Qwen embedding.
- Image-only Qwen embedding.
- Native text-image joint embedding.
- Independently quantized text and image codes concatenated into eight slots.
- Independently encoded text and image vectors concatenated before four-code
  OPQ/PQ.
- A paired CLIP text-image alignment control.

Current findings, open hypotheses, and the corresponding validation plans are
tracked in [RESEARCH_NOTES.md](RESEARCH_NOTES.md).

## Origin and license

This project modifies the implementation released with the ICML 2025 paper
“Contextually Tokenizing Action Sequences for Generative Recommendation.” The
original paper and its authors should be cited when this code is used in
research.

The inherited source code remains available under the Apache License 2.0. See
[LICENSE](LICENSE).
