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

## Origin and license

This project modifies the implementation released with the ICML 2025 paper
“Contextually Tokenizing Action Sequences for Generative Recommendation.” The
original paper and its authors should be cited when this code is used in
research.

The inherited source code remains available under the Apache License 2.0. See
[LICENSE](LICENSE).
