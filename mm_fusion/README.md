# Multimodal late-fusion semantic IDs (Amazon Beauty)

Text semantic IDs and image semantic IDs are quantised **independently** and then
concatenated into one ActionPiece item representation:

    2+2 -> 4 semantic slots      3+3 -> 6      4+4 -> 8
    plus one random hash-bucket slot appended by _get_hashed_feat

Everything needed to reproduce the run, plus the diagnostics that were run along
the way and what they did or did not show, is in this directory.

## The code fix

The released `qwen_separate` metadata mode was only an *artifact-path hook*: it
pointed the feature cache at a pre-built file but contained no code that
concatenated text and image semantic IDs. That branch existed only for
`metadata: sentence_image`, whose image encoder is hard-wired to CLIP. Every
existing Qwen multimodal artifact had been produced on another machine.

`patch_tokenizer.py` makes two changes to
`genrec/models/ActionPiece/tokenizer.py`:

1. `_get_item2feat` takes the concatenation branch for `metadata: qwen_separate`
   as well as `sentence_image`;
2. image semantic IDs get a cache name containing the quantiser settings
   (`_image_semantic_artifact_stem`), so the three arms cannot read or overwrite
   each other's cache.

The original file is kept as
`genrec/models/ActionPiece/tokenizer.py.bak-before-mm-fusion`.

## Inputs and provenance

The raw 768-d Beauty vectors live in the RPG checkout and are symlinked in:

    cache/AmazonReviews2014/Beauty/processed/
      qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948...asdpa.sent_emb
          -> RPG Text-emb-768.sent_emb
      Qwen3-VL-Embedding-8B.image_emb
          -> RPG Image-emb-768.image_emb

Evidence that these are the right vectors:

* the 4096-d text export is **byte-identical** across the two projects
  (sha256 `7200c2c364a30a0b…`), and ActionPiece's E1 experiment consumed that
  file, so both projects share one Qwen3-VL-Embedding-8B encoding run;
* the stem the code computes from the config reproduces the cached stem exactly
  (`i711daf44f948` = sha256 of the instruction, first 12 hex);
* `id2item` is identical in both projects (12,102 entries including PAD).

Caveat: the 768-d *image* vectors have no byte-identical counterpart. They come
from the same export directory, the item order aligns, and 24 rows are exactly
zero (missing images). Their association with the cached image semantic IDs is
strong but indirect — see `image_provenance.py`.

## The three arms

`experiments/mm_sep{22,33,44}_beauty.yaml` — identical except the number of
codebooks per modality. Codebook size is fixed at 256 so slot count is the only
variable.

## File index

| file | purpose |
| --- | --- |
| `patch_tokenizer.py` | applies the code fix (idempotent, prints a diff) |
| `gen_mm_configs.py` | writes the three arm configs |
| `prep_bigcb.py` | writes two large-codebook variants (prepared, not launched) |
| `run_tokenizer.sh <arm>` | builds one arm's semantic IDs + vocabulary |
| `validate_vocab.py [arm…]` | checks a vocabulary loads and has the right width |
| `mm_driver.sh` | schedules the three arms (wait → validate → train) |
| `mm_finisher.sh` | waits for all arms, then builds the results table |
| `mm_postfinish.sh` | splices the table into the report |
| `rerun_test.sh <arm> [bs]` | re-runs only the test pass from a checkpoint |
| `make_final_table.py` | main table + diagnostics + caveats |
| `mm_results.py`, `latest_compare.py` | quick progress comparisons |
| `RECOVERY.md` | how to restart / recover each stage |

Diagnostics:

| file | question it answers |
| --- | --- |
| `provenance_check.py` | are the candidate vectors the ones the SIDs came from? |
| `image_provenance.py` | the zero-vector collapse test for the image vectors |
| `sid_structure.py` | what the cached semantic IDs actually look like |
| `identity_card.py` | how many distinct item identities a representation gives |
| `semantic_vs_hash.py` | how much identity is semantic vs the random hash digit |
| `vocab_modality.py` | is the merged vocabulary genuinely cross-modal? |
| `vocab_modality_weighted.py` | the same, weighted by merge priority |
| `verify_arms.py` | are the three arms' artefacts distinct and sane? |

## Diagnostics: what held up

**The late fusion is real.** Expanding the merge tree of each vocabulary,
56–61% of merged tokens reference at least one text slot *and* at least one image
slot (63.5–65.7% weighted by merge priority). The merged vocabulary genuinely
combines the modalities rather than keeping two separate streams.

**A falsifiable mechanism.** `_get_hashed_feat` appends a *random*
collision-indexed bucket. The fewer semantic slots an arm has, the more of its
vocabulary absorbs that random digit:

| arm | type: contains hash | weighted: contains hash |
| --- | --- | --- |
| mm 2+2 | 57.9% | 44.6% |
| mm 3+3 | 38.4% | 29.3% |
| mm 4+4 | 27.2% | 21.7% |

This predicts **4+4 > 3+3 > 2+2**. A monotone ordering supports the mechanism;
any other ordering falsifies it. The prediction is recorded in
`RESULTS_DRAFT.md` before the 2+2 and 4+4 results exist.

## Diagnostics: what did NOT hold up

Two explanations were proposed during the run and are not supported:

1. **"Coarse semantic IDs force memorisation."** Measured on the item features
   with the hash digit removed, the new arms are *finer* than the best baseline
   (3+3 has 9,766 distinct semantic tuples versus 6,913 for text-only, whose
   items rely on the hash digit 42.9% of the time versus 19.3%). Granularity
   therefore does not explain the gap.
2. **"Lower training loss means overfitting."** Per-token loss is not comparable
   across arms because the number of ActionPiece categories, and hence the
   teacher-forced prediction steps per item, differs (5 vs 7 vs 9).

Earlier intermediate-epoch conclusions were also premature: `mm 3+3` was called
"plateaued" around epoch 63 and then improved by a further 0.005 over the next 30
epochs, while the legacy arms only peak at epochs 136–172.

## How the results are produced

```
mm_driver.sh      wait for each arm's tokenizer → validate → train
mm_finisher.sh    wait for all training → make_final_table.py → RESULTS_TABLE.md
mm_postfinish.sh  wait for the table → splice into RESULTS_DRAFT.md → RESULTS.md
```

All three are detached `nohup` processes, so the pipeline survives a closed
session. See `RECOVERY.md`.
