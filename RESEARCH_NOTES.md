# Research Notes

Last updated: 2026-07-27

> Historical working notes, frozen at the date above. The OPQ32 task and the
> “Next experiment priority” list below are no longer active tasks. Treat this
> file as provenance for the code, not as an authoritative result table.

## Current Beauty results

All Qwen experiments below use Qwen3-VL-Embedding-8B and four OPQ tokens.

| Experiment | Representation | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| E1-768 | Text only | 0.0507 | 0.0346 | 0.0779 | 0.0434 |
| E1-4096 | Text only | 0.0486 | 0.0323 | 0.0756 | 0.0410 |
| E3-768 | Native joint text-image embedding | 0.0531 | 0.0349 | 0.0816 | 0.0441 |
| E3-4096 | Native joint text-image embedding | 0.0519 | 0.0356 | 0.0782 | 0.0440 |
| E5-768 | Separate 768D text/image embeddings, concat, shared OPQ4 | **0.0552** | **0.0368** | **0.0824** | **0.0455** |

E5-768 is the current best configuration on all four metrics. It exceeds the
reported RPG result on Recall@5 (0.0550) and Recall@10 (0.0809), but not yet on
NDCG@5 (0.0381) or NDCG@10 (0.0464).

## Hypothesis 1: MRL dimension versus a fixed OPQ bottleneck

### Observation

Both E1 and E3 perform slightly worse at 4096 dimensions than at 768
dimensions when both are compressed into the same four 8-bit OPQ codes.

### Working explanation

Qwen3-VL-Embedding uses Matryoshka Representation Learning (MRL), so its first
768 dimensions may preserve most recommendation-relevant semantics. OPQ4 has a
fixed capacity of 32 bits regardless of input dimension. After rotation, each
codebook must quantize approximately 192 dimensions for a 768D input but 1024
dimensions for a 4096D input. The higher-dimensional input may therefore incur
more quantization distortion, contain more task-irrelevant detail, and be
harder to estimate from only about 12K Beauty items.

Do not interpret dimension ratios as information-retention percentages. Four
OPQ tokens are four categorical 8-bit codes, not four retained continuous
dimensions.

### Clean validation

1. Reconstruct vectors from OPQ4 and compare 768D versus 4096D using normalized
   MSE and reconstruction cosine similarity.
2. Compare top-k neighbor preservation before and after OPQ4.
3. Report per-codebook usage, entropy, and imbalance.
4. If an end-to-end check is needed, keep the output at four tokens and add one
   intermediate input dimension such as 2048D. This avoids the sequence-length
   confound introduced by OPQ8 or OPQ16.

OPQ8/16 is not a clean primary test: it increases representational capacity
but also lengthens the autoregressive target sequence and can make
recommendation prediction harder.

## Hypothesis 2: Modality dominance in E3 native joint fusion

### Question

Does Qwen's native joint text-image embedding allow one modality—most likely
text—to dominate the fused E3 representation, limiting the complementary
contribution of the other modality?

### Required representations

For the same items, construct normalized 768D embeddings with the same model,
item order, and instruction:

- `J = f(text, image)`: E3 joint embedding.
- `T = f(text)`: text-only embedding.
- `I = f(image)`: image-only embedding.

The instruction must be identical in all three conditions. Existing E3 and E1
embeddings use the same general product instruction, but the current E5
image-only cache uses an image-specific instruction. Therefore, strict E3
dominance analysis requires a new image-only cache using the E3 instruction;
the E5 image cache can only support an approximate preliminary analysis.

### Primary measurements

1. **Representation alignment**
   - Compute paired distributions of `cos(J, T)` and `cos(J, I)`.
   - Define the alignment gap as
     `D_align = mean(cos(J,T) - cos(J,I))`.
   - Use paired bootstrap confidence intervals rather than only the mean.

2. **Neighborhood inheritance**
   - Build top-k neighbors for `J`, `T`, and `I`.
   - Compare `overlap(N_J, N_T)` against `overlap(N_J, N_I)` at several k.
   - A consistently larger overlap indicates that the joint representation
     inherits more of that modality's semantic geometry.

3. **Counterfactual sensitivity**
   - On a fixed sample, replace the text while holding the image fixed and
     measure the displacement of `J`.
   - Separately replace the image while holding the text fixed.
   - Define each modality's influence using the mean cosine displacement.
   - Use both random cross-category swaps and within-category hard swaps so the
     result is not driven only by obviously incompatible pairs.

### Interpretation safeguards

- Greater alignment with one modality demonstrates dominance, but not
  necessarily harmful dominance. The dominant modality may simply be more
  informative for Beauty recommendation.
- E2 image-only recommendation performance is needed to contextualize the
  representation analysis.
- Prompt differences, missing images, and image quality must be controlled.
- The strongest evidence of harmful dominance would combine alignment,
  neighborhood, and counterfactual results showing weak image influence with
  E5's consistent improvement from explicitly preserving both modalities
  before shared OPQ.

## Next experiment priority

1. Generate and validate Beauty OPQ32 from the existing Qwen 4096D text
   embeddings for the RPG-aligned experiment described below.
2. Preserve and back up the completed E5-768 artifacts and training result.
3. Run the low-cost E3 modality-dominance analysis.
4. Complete E2 image-only and E4 separate-OPQ controls.
5. Add OPQ reconstruction and neighbor-preservation diagnostics for the
   768D-versus-4096D hypothesis.

## RPG-aligned Beauty OPQ32 experiment

### Confirmed configuration

The official `facebookresearch/RPG_KDD2025` Beauty reproduction command uses:

```yaml
n_codebook: 32
codebook_size: 256
```

In the RPG tokenizer, `n_digit` returns `n_codebook`, and the corresponding
FAISS factory is `OPQ32,IVF1,PQ32x8`. Therefore, the reported Beauty result
uses a 32-digit semantic ID, not OPQ4.

The official RPG representation uses `text-embedding-3-large` at 3072D,
followed by PCA to 512D and OPQ32. Our immediate experiment is an adaptation,
not an exact representation-level reproduction: reuse the already computed
Qwen3-VL-Embedding-8B 4096D text embeddings and quantize them with OPQ32.

### Available local inputs

- Beauty catalog size: 12,101 items.
- Items available to train OPQ under the leave-two-out protocol: 12,068.
- Input dimension: 4096.
- OPQ32 subvector dimension: 128.
- Existing Qwen embedding:
  `cache/AmazonReviews2014/Beauty/processed/qwen_local.Qwen3-VL-Embedding-8B.d4096.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.sent_emb`

The existing standalone builder reorders FAISS codes by stored item IDs before
writing the item-to-SID JSON, which is safer than assuming IVF list order.

### Generation command

Run this in an environment with a compatible FAISS Python package:

```bash
python scripts/build_opq_sem_ids.py \
  --embedding-path cache/AmazonReviews2014/Beauty/processed/qwen_local.Qwen3-VL-Embedding-8B.d4096.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.sent_emb \
  --id-mapping-path cache/AmazonReviews2014/Beauty/processed/id_mapping.json \
  --item-seqs-path cache/AmazonReviews2014/Beauty/processed/all_item_seqs.json \
  --output-path cache/AmazonReviews2014/Beauty/processed/qwen_local.Qwen3-VL-Embedding-8B.d4096.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.opq32x256.seed2024.sem_ids \
  --dimension 4096 \
  --n-codebooks 32 \
  --codebook-size 256 \
  --threads 8 \
  --seed 2024
```

Do not overwrite the existing OPQ4 artifact. The `opq32x256` filename is
intentional because the tokenizer cache identity includes the OPQ length and
codebook size.

### Required validation

Before RPG training, verify:

1. 12,101 item-to-SID entries.
2. Exactly 32 integer codes per item.
3. Raw codes are all in `[0, 255]`.
4. Item-key equality with `id_mapping.json`.
5. Exact-SID collision statistics.
6. Artifact SHA-256.

RPG should consume this raw `*.sem_ids` mapping. It must not consume the
ActionPiece `item.*.feat` file, because that artifact includes a hash feature,
or the learned `actionpiece*.json` merge vocabulary.
