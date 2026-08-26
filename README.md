# ActionPiece Multimodal Experiments

This repository is an experimental extension of
[ActionPiece](https://arxiv.org/abs/2502.13581) for multimodal generative
recommendation. It contains the code and configurations used to study
multimodal fusion and semantic bandwidth on Beauty, DY, and QB. Sports is kept
as an additional diagnostic configuration outside the primary study scope.

## Experiment scope

The implemented representation routes are:

- text-only and image-only Qwen embeddings;
- native joint text-image Qwen embeddings;
- separate continuous text/image embeddings concatenated before shared OPQ;
- separate text/image OPQ streams followed by ActionPiece tokenization;
- embedding-prefix and fixed-bottleneck semantic-bandwidth controls;
- merge-provenance, candidate-opportunity, and exact slot-permutation
  diagnostics.

Historical experiment filenames use `E4` for separate quantization followed by
ActionPiece and `E5` for continuous concatenation. Use the descriptive route
names when comparing these configurations with external manuscripts.

The validated 768D E1/E2/E3 matrices are published separately in
[qwen-multimodal-recommendation-embeddings](https://github.com/Russ-YANG/qwen-multimodal-recommendation-embeddings).

## Workflow

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

## Implemented configurations

- `experiments/` contains the Beauty, DY, QB, and Sports generation/training
  configurations.
- `scripts/` contains data preparation, Qwen/Gemini embedding generation,
  MRL-prefix derivation, OPQ construction, packing controls, plotting, and
  merge-analysis utilities.
- `tests/` contains focused tests for multimodal cache routing, NineRec data
  adaptation, ActionPiece opportunity tracking, and merge analysis.
- `results/modality_opportunities/` contains the small, machine-readable
  summaries and report for the corrected slot-permutation analysis.
- `plots/truncation/` contains the generated semantic-bandwidth figures.

`RESEARCH_NOTES.md` is a historical working log. Generated datasets, model
weights, dense embeddings, tokenizers, merge logs, checkpoints, credentials,
and run logs are intentionally excluded from this repository.

## Origin and license

This project modifies the implementation released with the ICML 2025 paper
“Contextually Tokenizing Action Sequences for Generative Recommendation.” The
original paper and its authors should be cited when this code is used in
research.

The inherited source code remains available under the Apache License 2.0. See
[LICENSE](LICENSE).
