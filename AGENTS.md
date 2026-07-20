# ActionPiece Project Guide

This file is the persistent working guide for coding agents and contributors in
this repository. Read it before making changes, and update it whenever the
project's research direction, experiment interface, or validation workflow
changes.

## Project purpose

This repository started as the implementation of the ICML 2025 Spotlight paper
“Contextually Tokenizing Action Sequences for Generative Recommendation.”
The current experimental branch extends ActionPiece with multimodal Amazon item
features.

The active research question is:

> Can ActionPiece learn useful contextual and cross-modal tokens from product
> text and image features, and do those tokens improve generative
> recommendation?

The main experimental comparison is among:

- `metadata=sentence`: text-only baseline.
- `metadata=sentence_image`: text and image embeddings are quantized
  independently, then their discrete features are given to ActionPiece
  together. This mode can produce explicit text-image merge tokens.
- `metadata=sentence_image_fused`: normalized text and image embeddings are
  concatenated before product quantization. `fused_image_weight` controls the
  image contribution.

The default checked-in configuration remains the text-only baseline unless an
experiment intentionally changes it.

The full planned Beauty experiment matrix has seven conditions: one upstream
baseline, five Qwen3-VL-Embedding ablations, and one CLIP alignment control.
All semantic-code counts below exclude the collision-resolution hash feature.

The upstream baseline is:

0. Sentence-T5 text baseline: encode product text with
   `sentence-transformers/sentence-t5-base`, then OPQ/PQ into four codes.

The five Qwen3-VL-Embedding conditions are:

1. Text only: text vector, then OPQ/PQ into four codes.
2. Image only: image vector, then OPQ/PQ into four codes.
3. Native multimodal fusion: jointly encode text and image, then OPQ/PQ the
   fused vector into four codes.
4. Discrete late fusion: encode text and image independently, quantize each
   into four codes, then concatenate the codes into eight slots.
5. Continuous late fusion: encode text and image independently, normalize and
   concatenate the two vectors, then OPQ/PQ into four codes.

Conditions 4 and 5 must reuse the exact independently cached vectors from
conditions 1 and 2; they are downstream transformations, not new encoder calls.
Keep output dimension and prompt/instruction policy fixed across conditions.

The CLIP alignment control is:

6. CLIP aligned continuous fusion: encode product text with the CLIP text
   encoder and its image with the paired CLIP image encoder, normalize and
   concatenate the two 512-dimensional vectors, then train one OPQ/PQ on the
   fused vectors and produce four codes. This keeps the recommendation target
   length equal to the text-only baseline while replacing the current
   independently distributed Sentence-T5 and CLIP inputs with paired encoders
   trained in CLIP's shared contrastive space.

Condition 6 must use the same CLIP checkpoint for both encoders, the same
catalog text and images used by the other conditions, the same train-item set
for quantizer fitting, and the same four-code PQ settings. The eight-code CLIP
late-fusion controls are deferred. Condition 6 is not implemented yet; add a
dedicated metadata mode and cache names before running it so it cannot reuse
the existing Sentence-T5/CLIP fusion artifacts.

The active E1 uses the explicit open-weight `Qwen/Qwen3-VL-Embedding-8B`
checkpoint at revision `2c4565515e0f265c6511776e7193b22c0968ddc7`, not the
parameter-undisclosed DashScope API model. It emits 768-dimensional,
L2-normalized MRL prefixes with the shared English instruction, a maximum input
length of 2048 tokens, and uses SDPA in bfloat16. The official encoder source is
fixed at revision `393e2978d27852b0d0230d6994f37f9c15bed73c`. It is configured in
`experiments/e1_qwen3_vl_8b_text_beauty.yaml` and has a dedicated local-backend
cache identity. Run its tokenizer-only stage on the RTX PRO 6000 with:

```bash
python main.py --config_file=experiments/e1_qwen3_vl_8b_text_beauty.yaml
```

The previous API-based E1 remains in `experiments/e1_qwen_text_beauty.yaml` for
provenance only and must not be mixed with the local-8B run. Both Qwen encoders
write item-aligned batches to a `.partial` file and commit progress after each
successful batch. The final `.sent_emb` filename appears only after every item
is encoded. Do not delete partial or progress artifacts when resuming an
interrupted run.

## Current implementation

- Dataset: Amazon Reviews 2014, configured in
  `genrec/datasets/AmazonReviews2014/config.yaml`.
- Text encoder: Sentence Transformers, currently
  `sentence-transformers/sentence-t5-base`.
- Image encoder: CLIP, currently `openai/clip-vit-base-patch32`.
- Continuous embeddings are converted to discrete semantic IDs with Faiss
  OPQ/PQ.
- ActionPiece learns a vocabulary by merging frequently co-occurring feature
  patterns within actions and across adjacent actions.
- Image download or decode failures produce zero image embeddings and are
  recorded in `image_download_failures.jsonl` under the processed cache.
- Tokenizer construction can emit a JSONL merge log for modality analysis.

## Important files

- `main.py`: experiment entry point.
- `genrec/default.yaml`: shared training, evaluation, cache, and run settings.
- `genrec/datasets/AmazonReviews2014/config.yaml`: dataset category and metadata
  mode.
- `genrec/datasets/AmazonReviews2014/dataset.py`: metadata extraction, including
  product image URLs.
- `genrec/models/ActionPiece/config.yaml`: encoders, PQ settings, vocabulary
  size, merge logging, model hyperparameters, and fusion weight.
- `genrec/models/ActionPiece/core.py`: ActionPiece vocabulary construction and
  merge logging.
- `genrec/models/ActionPiece/tokenizer.py`: text/image encoding, quantization,
  fused features, tokenizer caching, and collation.
- `genrec/models/ActionPiece/qwen_local.py`: resumable local Qwen3-VL text
  embedding, MRL truncation, normalization, and provenance manifest.
- `genrec/models/ActionPiece/model.py`: recommendation model and generation.
- `scripts/analyze_modality_merges.py`: summarizes text-only, image-only,
  cross-modal, and hash-containing merge events.
- `scripts/prepare_amazon_multimodal_data.py`: creates a deterministic local
  text/image catalog and downloads product images with resumable manifests.
- `scripts/demo_qwen_vl_embedding_api.py`: runs a two-request, one-product
  Qwen API smoke test for independent and fused embeddings without persisting
  full vectors or secrets.
- `README.md`: upstream-facing overview and paper reproduction instructions.

## Local Beauty data

Prepare the Amazon Reviews 2014 Beauty multimodal data independently of the ML
training dependencies:

```bash
python scripts/prepare_amazon_multimodal_data.py \
    --category Beauty \
    --download-images
```

The ignored local cache is `cache/AmazonReviews2014/Beauty/`. Important files:

- `raw/reviews_Beauty_5.json.gz`: 5-core review interactions.
- `raw/meta_Beauty.json.gz`: original product metadata.
- `processed/multimodal_catalog.jsonl`: ASIN-aligned text, structured-text
  ablation input, and image URL.
- `processed/image_download_manifest.jsonl`: local image path, status, shape,
  byte count, and checksum per ASIN.
- `processed/multimodal_summary.json`: coverage statistics.
- `images/`: locally cached original image bytes.

The catalog's `text` field mirrors the existing repository preprocessing and
is the input for strict encoder comparisons. `structured_text` adds field labels
and must be treated as a separate prompt/preprocessing ablation.

## Two-stage compute workflow

The intended experiment workflow is split into two artifact boundaries:

1. Tokenizer-only stage: prepare/cache inputs and embeddings, train OPQ/PQ,
   build semantic IDs, construct the ActionPiece vocabulary, and analyze merge
   logs. API-derived embeddings can be built on the MacBook; the explicit local
   Qwen3-VL-Embedding-8B encoder runs on the RTX PRO 6000. These artifacts must
   be deterministic and portable.
2. Recommendation stage: train the recommendation model and run
   validation/test evaluation on a GPU machine from the frozen tokenizer and
   feature artifacts.

Do not start a full embedding, OPQ/PQ, ActionPiece merge, training, or evaluation
run without first presenting the expected API calls/cost and outputs to the
user and receiving explicit approval. For a tokenizer-only run, stop after
producing and validating semantic IDs, the ActionPiece tokenizer vocabulary,
and merge logs; do not continue into recommendation-model training.

Before transferring artifacts, save the full resolved configuration, model
identifier or API model name, prompt/instruction text, embedding dimension,
normalization policy, item-to-row mapping, PQ parameters, random seed, and file
checksums. Never rebuild the tokenizer independently on the 5090 machine for a
reported run.

For the Qwen API workflow, keep `DASHSCOPE_API_KEY` only in the ignored,
user-readable `.env.local` file. Never log, print, commit, or copy the key into
experiment artifacts. The supported API instruction parameter is `instruct`;
use the same exact instruction for independent and native-fusion calls. Cache
independent text/image vectors once and reuse them for the text-only,
image-only, discrete-late-fusion, and continuous-late-fusion conditions.

## Running experiments

The general command is:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --metadata=sentence \
    --run_id=descriptive_run_name
```

Use command-line overrides instead of editing checked-in defaults for one-off
experiments. Examples:

```bash
# Build a tokenizer with separate text and image feature slots.
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --metadata=sentence_image \
    --tokenizer_only=True \
    --run_id=sports_sentence_image_tokenizer

# Build a tokenizer from fused text-image embeddings.
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --metadata=sentence_image_fused \
    --fused_image_weight=1.0 \
    --tokenizer_only=True \
    --run_id=sports_sentence_image_fused_tokenizer
```

Training downloads datasets, encoder weights, and product images when the
corresponding artifacts are not cached. Expect the first multimodal run to be
network-, disk-, and GPU-intensive.

## Merge-log analysis

Merge logging is controlled by:

- `actionpiece_merge_log`
- `actionpiece_merge_log_interval`
- `actionpiece_merge_log_path`

If no explicit path is supplied, the log is written as
`processed/actionpiece.merge_log.jsonl` inside the dataset cache.

For the default separate multimodal layout (four text PQ slots, four image PQ
slots, and one hash slot), run:

```bash
python scripts/analyze_modality_merges.py \
    path/to/actionpiece.merge_log.jsonl \
    --text-slots=4 \
    --image-slots=4 \
    --hash-slot=8
```

Do not interpret the fused mode with those text/image slot boundaries: fused PQ
codes do not retain separately attributable text and image slots.

## Cache awareness

This project caches processed metadata, embeddings, semantic IDs, item
features, and tokenizer vocabularies. When changing an encoder, feature layout,
PQ setting, metadata mode, fusion weight, or vocabulary construction rule:

1. Inspect how the affected cache filename is derived.
2. Ensure incompatible artifacts cannot silently reuse the same filename.
3. Remove or regenerate only the affected cache artifacts for the experiment.
4. Record the exact configuration used for reported results.

Never broadly delete user caches, checkpoints, logs, or experiment outputs
without explicit approval.

## Development rules

- Preserve the text-only ActionPiece baseline while adding experimental modes.
- Keep feature-slot ordering explicit and consistent between tokenizer
  construction, merge analysis, and documentation.
- Treat missing images as a measured data-quality issue; report failure counts
  rather than silently assuming complete image coverage.
- Avoid data leakage: train Faiss/PQ indexes only on items observed in the
  training split, as the current implementation intends.
- Do not mix results from stale tokenizer caches with a changed feature
  configuration.
- Prefer configuration-driven behavior over hard-coded experiment values.
- Keep changes scoped; do not rewrite upstream code unless the experiment
  requires it.
- Do not commit generated datasets, downloaded images, caches, embeddings,
  checkpoints, logs, or model weights.

## Validation checklist

Use the smallest checks proportional to the change:

1. Run syntax/import checks for edited Python files.
2. Exercise tokenizer construction with a small or already cached dataset when
   possible.
3. For multimodal changes, verify embedding shapes, item alignment, missing
   image handling, and semantic-ID slot counts.
4. For vocabulary changes, verify tokenizer serialization and inspect the merge
   log.
5. For training changes, run a short smoke experiment before a full run.
6. Check `git diff` and ensure generated artifacts are not included.

Full-scale training is expensive and is not required for every code change, but
the handoff must clearly state which checks were and were not run.

## Maintaining this guide

Update this file in the same change whenever any of the following changes:

- the active research question or baseline;
- metadata modes or feature-slot semantics;
- encoders, quantization, fusion, or tokenizer behavior;
- important entry points or analysis scripts;
- cache naming or invalidation requirements;
- standard run commands or validation expectations.

Keep this guide factual and concise. Put public installation and paper
reproduction instructions in `README.md`; put persistent project-working
context here.
