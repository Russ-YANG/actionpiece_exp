# Multimodal late-fusion semantic IDs on Amazon Beauty — results

Draft. Methodology and diagnostics below are final; the results table is filled
in by `mm_fusion/make_final_table.py` once the three arms finish.

## What was run

Text semantic IDs and image semantic IDs are quantised **independently** and then
concatenated into one ActionPiece item representation:

| arm | text codes | image codes | ActionPiece categories |
| --- | --- | --- | --- |
| mm 2+2 | 2 × 256 | 2 × 256 | 5 (4 semantic + 1 hash) |
| mm 3+3 | 3 × 256 | 3 × 256 | 7 (6 semantic + 1 hash) |
| mm 4+4 | 4 × 256 | 4 × 256 | 9 (8 semantic + 1 hash) |

Codebook size is fixed at 256 in all three arms, so **slot count is the only
variable**. Everything else is identical across them: Qwen3-VL-Embedding-8B at
768 dimensions for both modalities, the same 12,101-item Beauty catalogue, seed
2024, `actionpiece_vocab_size=40000`, `n_hash_buckets=128`,
`max_item_seq_len=20`, `num_beams=50`, `n_inference_ensemble=5`, 200 epochs with
no early stopping.

## What had to be built

1. **A code fix.** The released `qwen_separate` metadata mode was only an
   *artifact-path hook* — it pointed the feature cache at a pre-built file but
   contained no code that concatenated text and image semantic IDs. That branch
   existed only for `metadata: sentence_image`, whose image encoder is hard-wired
   to CLIP. Two changes to `genrec/models/ActionPiece/tokenizer.py`: let
   `qwen_separate` take the concatenation branch, and give image semantic IDs a
   cache name containing the quantiser settings so the three arms cannot
   overwrite each other's cache. A backup of the original file is kept.

2. **Input vectors.** The 768-d Beauty text and image embeddings were in neither
   ActionPiece checkout; they were linked in from the RPG project. Provenance:
   the 4096-d text export is **byte-identical** across the two projects
   (sha256 `7200c2c3…`), and `id2item` matches exactly (12,102 entries).

## Controlled comparisons this setup does and does not support

- **Strictly supported:** the three new arms against each other.
- **Strictly supported:** `mm 4+4` versus the text-only baseline — the two share
  the *same* text semantic-ID file (7,047 distinct IDs in both), so the only
  difference is the four added image codes.
- **Supported:** `mm 2+2` versus the legacy separate baseline (both Qwen-768 text
  + Qwen-768 image), differing in codebook size and the freqmax/pack2
  post-processing.
- **Not supported:** any image-side comparison against the legacy image-only or
  joint baselines, which used CLIP ViT-B/32 at 512 dimensions.

## Diagnostics that were checked and did NOT hold up

Two explanations were proposed during the run and are **not supported by the
data**. They are recorded here so the final report does not repeat them.

**"Coarse semantic IDs force the model to memorise."** Measured semantic-tuple
cardinality (item features with the hash digit removed):

| arm | distinct semantic tuples | items needing the hash digit | largest group |
| --- | --- | --- | --- |
| mm 2+2 (new) | 8,049 | 0.335 | 62 |
| mm 3+3 (new) | 9,766 | 0.193 | 36 |
| mm 4+4 (new) | 10,769 | 0.110 | 20 |
| text only 4×256 | **6,913** | **0.429** | 65 |
| image only 4×256 | 9,638 | 0.204 | 41 |
| joint 4096 4×256 | **5,858** | **0.516** | 78 |
| legacy separate 2×65536 | 10,821 | 0.106 | 17 |

The new arms are *finer* than the best-performing baseline, so granularity does
not explain the gap. (`_get_hashed_feat` appends a collision-indexed random digit,
so every arm ends up with 12,101 unique full identities regardless.)

**"Lower training loss means overfitting."** Per-token training loss is not
comparable across arms: the number of ActionPiece categories, and therefore the
teacher-forced prediction steps per item, differs (5 vs 7 vs 9). The observed
lower loss for `mm 3+3` is therefore confounded and cannot be read as
overfitting.

## What the three arms finally show

All three arms finished 200 epochs and produced test metrics.

| arm | slots (+hash) | val nDCG@10 | val R@10 | test nDCG@10 | test R@10 | best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| **mm 2+2**, seed 2024 | 5 | 0.0581 | 0.1046 | **0.0446** | **0.0808** | 199 |
| **mm 2+2**, seed 42 | 5 | **0.0589** | 0.1039 | 0.0440 | 0.0798 | 198 |
| **mm 3+3** | 7 | 0.0557 | 0.1009 | 0.0420 | 0.0791 | 161 |
| **mm 4+4** | 9 | 0.0556 | 0.1019 | 0.0420 | 0.0776 | 179 |
| text only 4×256 (baseline) | 5 | 0.0584 | 0.1043 | 0.0434 | 0.0779 | 172 |
| joint 4096 4×256 (baseline) | 5 | 0.0573 | 0.1028 | 0.0441 | 0.0783 | 167 |
| image only 4×256 (baseline) | 5 | 0.0503 | 0.0914 | 0.0370 | 0.0687 | 157 |
| legacy separate 2×65536 | 5 | 0.0514 | 0.0897 | n/a | n/a | 136 |

**The ordering is `2+2 > 3+3 ≈ 4+4`.** Adding semantic slots did not help: 3+3 and
4+4 are separated by 0.0001 on validation and are *identical* on test to four
decimals. Going from four slots to six bought nothing, and from six to eight
bought nothing further.

**Against the baselines**, using the 2+2 mean of its two seeds:

| comparison | val nDCG@10 | test nDCG@10 | test R@10 |
| --- | --- | --- | --- |
| mm 2+2 mean vs text only | +0.0001 | +0.0009 | +0.0024 |
| mm 2+2 mean vs joint 4096 | +0.0012 | +0.0002 | +0.0020 |
| mm 2+2 mean vs image only | +0.0082 | +0.0073 | +0.0116 |
| mm 2+2 mean vs legacy separate 2×65536 | +0.0071 | n/a | n/a |

So late fusion at 2+2 **matches** the two strongest single-modality baselines on
nDCG@10 — the margins there are +0.0001…+0.0012, i.e. at or below the measured seed
spread of 0.0006–0.0007 — while being clearly ahead of image-only and of the legacy
separate baseline, the latter despite using codebooks 256× smaller. On **test
recall@10 the margin over both strong baselines (+0.0020…+0.0024) is 2–3× the
measured spread**, so that part is a real difference rather than a tie.

The practical reading: text+image late fusion does not beat the best
single-modality text baseline on ranking quality at this scale, but it does not
lose to it either, and it returns noticeably more of the relevant items within the
top 10.

### Corrections recorded along the way

Three intermediate readings in this project turned out to be wrong. They are kept
here because each one is a trap a reader could fall into again.

1. **"3+3 has plateaued and is a failure" (epoch ~66).** Its best validation
   nDCG@10 was 0.040 while every baseline was already at 0.044–0.047. The arm then
   improved to 0.0557. Judging at one third of the schedule was the error.
2. **"3+3 is ahead of the legacy baseline" (epoch ~131, from a single epoch).**
   The arm is volatile (epoch-to-epoch std ≈ 0.0019 versus ≈ 0.0004 for converged
   runs), so one epoch can show a lead that the smoothed curve does not support.
   The final verdict is that 3+3 does beat that baseline, but the single-epoch
   evidence for it was not sound.
3. **An extrapolation of 4+4 to 0.051–0.055 from its epoch-109 value.** It finished
   at 0.0556. 4+4 gained +0.0130 over its last 91 epochs — more than 2+2's +0.0119
   — so it had the steepest late trajectory of the three and linear extrapolation
   understated it.

## Caveats that must travel with the numbers

1. **Codebook size differs between the new arms and the legacy separate
   baseline** (256 vs 65536). A gap against that baseline may reflect quantiser
   capacity rather than the fusion scheme.
2. The legacy separate run **never completed its test pass**
   (`torch.AcceleratorError` at 69%), so it has validation numbers only.
3. Batch sizes differ: new arms train at 128 and evaluate at 32 (4+4 evaluates at
   16, because beam-search memory scales with token sequence length and the
   9-slot configuration was projected to exceed the 24 GB card). Legacy arms used
   256 / 64. Batch size changes batching, not the metric.
4. All arms peak late (legacy best epochs 136–172), so intermediate-epoch
   rankings do not predict final rankings.

## Vocabulary-level diagnostics

A saved ActionPiece vocabulary stores the first `n_init_feats` tokens as
`[slot, value]` and every later token as the merge rule `[-1, child_a, child_b]`,
so the slots a merged token covers were recovered by expanding that binary merge
tree.

| arm | text-only tokens | image-only tokens | **cross-modal tokens** | tokens containing the hash slot | mean slots per token |
| --- | --- | --- | --- | --- | --- |
| mm 2+2 | 20.5% | 23.0% | **56.1%** | **56.5%** | 2.75 |
| mm 3+3 | 18.6% | 21.6% | **59.5%** | 37.2% | 2.98 |
| mm 4+4 | 19.0% | 20.0% | **60.7%** | 26.0% | 3.12 |

Weighting each merged entry by its merge priority (pair frequency, a proxy for
usage) rather than counting types gives the same ordering, so the pattern is not
an artefact of rare tokens:

| arm | weighted: contains hash | weighted: cross-modal |
| --- | --- | --- |
| mm 2+2 | **44.6%** | 63.5% |
| mm 3+3 | **29.3%** | 65.0% |
| mm 4+4 | **21.7%** | 65.7% |

Two things follow.

**The late fusion is real.** 56–61% of vocabulary tokens reference at least one
text slot *and* at least one image slot, so the merged vocabulary genuinely
combines the two modalities rather than keeping two separate token streams.

**A falsifiable mechanism — and it has been refuted.**

`_get_hashed_feat` appends a *random* collision-indexed bucket. The tables show
that the fewer semantic slots an arm has, the more of its vocabulary absorbs that
random digit — 57.9% for 2+2 versus 27.2% for 4+4 by type, and 44.6% versus 21.7%
weighted by usage. A random digit merged into semantic tokens looks like injected
noise, which yields the prediction **4+4 > 3+3 > 2+2**, i.e. more semantic slots
should generalise better. That was written down before the 2+2 and 4+4 results
existed, so it could not be fitted after the fact.

All three arms have now finished and the order is **exactly the reverse at the
top, with the middle two tied**:

| arm | val nDCG@10 | test nDCG@10 | tokens whose vocabulary absorbs the hash (weighted) |
| --- | --- | --- | --- |
| mm 2+2 (fewest slots) | **0.0581 / 0.0589** | **0.0446 / 0.0440** | 44.6% (most polluted) |
| mm 3+3 | 0.0557 | 0.0420 | 29.3% |
| mm 4+4 (most slots) | 0.0556 | 0.0420 | 21.7% (least polluted) |

The arm whose vocabulary is *most* contaminated by the random digit performs
best, and the two arms that differ by a factor of 1.35× in contamination are
indistinguishable (val differs by 0.0001; test is identical to four decimals).
**The hash-pollution mechanism is refuted.** The vocabulary-composition statistic,
however striking, does not predict downstream ranking quality.

Two further notes recorded for honesty:

* An intermediate extrapolation in this project forecast 4+4 at 0.051–0.055 from
  its epoch-109 value; it finished at 0.0556, above the top of that range. 4+4
  gained +0.0130 over its last 91 epochs, *more* than 2+2's +0.0119, so it had the
  steepest late trajectory of the three. Linear extrapolation from mid-run values
  is not reliable here.
* The seed-42 replica reproduces the 2+2 hash statistics exactly (same 8,049
  distinct semantic tuples, same 0.335 share needing the hash, same vocabulary
  composition). That is expected: the OPQ codes are seed-independent, so a new
  seed permutes which disambiguating digit each colliding item receives without
  changing the aggregate structure. The replica therefore measures *training* (plus
  relabelling) variance, not a genuinely different item representation.

## Comparisons the runs already in flight will settle

No additional arm is needed to answer these three questions; they fall out of the
runs already queued plus the legacy baselines.

| # | comparison | variable isolated | held constant |
| --- | --- | --- | --- |
| 1 | mm 2+2 vs mm 3+3 vs mm 4+4 | number of semantic slots per modality | codebook size (256), encoders, seed, protocol |
| 2 | mm 2+2 vs legacy separate 2×65536 | **codebook size** (256 vs 65536) | modality layout (2 text + 2 image), Qwen-768 text and image |
| 3 | text-only 4×256 vs mm 4+4 | **adding four image codes** | the text codes themselves (the same `sem_ids` file, 7,047 distinct IDs) |

Comparison 2 is the interesting one for the capacity confound: the legacy
baseline is the only arm that already exists with a different codebook size on
the same modality layout. It comes with a caveat — the legacy artefacts also
went through a `freqmax_pack2` post-processing step that is not implemented in
this checkout — so a difference there is "codebook size plus post-processing",
not codebook size alone.

## What would still need a new run

Only if comparison 2 shows a large capacity effect: a `3+3` arm with large
codebooks, to separate slot count from capacity at a fixed slot layout. Two such
configs are written but not launched (`experiments/mm_sep33_cb4096_beauty.yaml`,
`experiments/mm_sep33_cb65536_beauty.yaml`), each about 10–14 hours.

## How to read the final table

The numbers now exist, so the applicable case is resolved below the guide.

| case | what the table looks like | conclusion |
| --- | --- | --- |
| **A** | val and test both increase monotonically 2+2 → 3+3 → 4+4, with gaps above ~0.002 | Would support the hash-pollution mechanism. Not observed. |
| **B** | the three arms are within ~0.002 of each other | Slot count in the 4–8 range makes no measurable difference at a 256-entry codebook. Say that, rather than ranking them. |
| **C** | any arm at or above text-only (0.0584 val) | Late fusion with Qwen-768 is competitive with the strongest single-modality baseline. |
| **D** | all three below the legacy separate baseline (0.0514 val) | Codebook capacity matters more than slot count here. The next experiment is the large-codebook arm. |
| **E** | test metrics missing for an arm | Its test pass crashed. Re-run it with `rerun_test.sh` before interpreting anything. |

**Resolution.** Case D does not apply (every new arm is above 0.0514). Case A does
not apply. What happened is a combination of **B and C with one exception**:

* 3+3 and 4+4 fall under case **B** — they are separated by 0.0001 on validation
  and are identical on test, so slot count between six and eight made no
  measurable difference and should not be ranked.
* 2+2 sits about 0.0028 (val) and 0.0023 (test) above both of them, which is
  above the ~0.002 threshold, so *that* difference is real — but it runs in the
  direction opposite to the mechanism, with the fewest slots winning.
* 2+2 reaches case **C**: its two seeds bracket the text-only baseline
  (0.0581 and 0.0589 against 0.0584), so late fusion is competitive with, but not
  demonstrably better than, the strongest single-modality baseline.
* Case **E** does not apply — all four new runs completed their test passes,
  including the 4+4 arm whose beam search was projected to exceed the card.

Two things the table cannot show, and that must not be claimed:

* seed replication exists for the **2+2** arm only; 3+3 and 4+4 each have a single
  seed, so differences of ~0.0007 involving them remain unresolvable;
* the legacy baselines differ in image encoder (CLIP-512 versus Qwen-768), so
  image-side comparisons across that line are not like for like.

## Statistical caveat: what is separable from noise?

An earlier draft of this section could only offer a lower bound, because each arm
had been run at a single seed. A seed-42 replica of the 2+2 arm has since been
run, so the spread is now measured rather than guessed.

**Measured seed-to-seed spread of the 2+2 arm** (two full runs; seed 42 reprises
the quantiser naming, the hash buckets, the ActionPiece vocabulary and the
training run, while the OPQ codes themselves are seed-independent):

| arm | val nDCG@10 | val R@10 | test nDCG@10 | test R@10 |
| --- | --- | --- | --- | --- |
| 2+2, seed 2024 | 0.0581 | 0.1046 | 0.0446 | 0.0808 |
| 2+2, seed 42 | 0.0589 | 0.1039 | 0.0440 | 0.0798 |
| **spread** | **0.0007** | 0.0007 | **0.0006** | 0.0010 |
| 3+3, seed 2024 | 0.0557 | 0.1009 | 0.0420 | 0.0791 |
| **2+2 mean − 3+3** | **+0.0028** | +0.0024 | **+0.0023** | +0.0012 |

So the 2+2 advantage over 3+3 is **3.8× the seed spread on validation and 3.6× on
test**. It is not seed luck.

The earlier epoch-to-epoch estimate of the noise floor (0.0004–0.0006 for
converged runs) predicted this spread well, which is mild evidence that the
estimator was reasonable.

Note what this does *not* do: it gives a spread for the **2+2** arm only. The 3+3
and 4+4 arms still have a single seed each, so a difference of ~0.0007 involving
them would remain unresolvable.

**Where the new arms sit against the baselines.** With the 2+2 mean used in place
of the single run:

| arm | val nDCG@10 | test nDCG@10 |
| --- | --- | --- |
| mm 2+2 (mean of 2 seeds) | 0.0585 | 0.0443 |
| text only 4×256 | 0.0584 | 0.0434 |
| joint 4096 4×256 | 0.0573 | 0.0441 |
| image only 4×256 | 0.0503 | 0.0370 |
| legacy separate 2×65536 | 0.0514 | n/a |

Against the two strongest single-modality baselines the differences are
+0.0001…+0.0012, i.e. **at or below the measured seed spread**: the honest reading
is that late fusion *matches* them rather than clearly beating them. Against
image-only and against the legacy separate baseline the margins are +0.0071 and
larger, which is well outside the spread.

## Results

### Main comparison

| arm | text codes | image codes | slots (+hash) | distinct text SIDs | distinct image SIDs | val nDCG@10 | val R@10 | test nDCG@10 | test R@10 | best epoch |
|---|---|---|---|---|---|---|---|---|---|---|
| mm 2+2 (new) | 2 x 256 | 2 x 256 | 5 | 2453 | 4747 | 0.0581 | 0.1046 | 0.0446 | 0.0808 | 199 |
| mm 2+2 seed42 (replica) | 2 x 256 | 2 x 256 | 5 | 2453 | 4747 | 0.0589 | 0.1039 | 0.0440 | 0.0798 | 198 |
| mm 3+3 (new) | 3 x 256 | 3 x 256 | 7 | 4637 | 8226 | 0.0557 | 0.1009 | 0.0420 | 0.0791 | 161 |
| mm 4+4 (new) | 4 x 256 | 4 x 256 | 9 | 7047 | 9590 | 0.0556 | 0.1019 | 0.0420 | 0.0776 | 179 |
| text only 4x256 | 4 x 256 | - x - | 5 | 7047 | - | 0.0584 | 0.1043 | 0.0434 | 0.0779 | 172 |
| image only 4x256 | - x - | 4 x 256 | 5 | - | - | 0.0503 | 0.0914 | 0.0370 | 0.0687 | 157 |
| joint 4096 4x256 | 4 x 256 | - x - | 5 | 5858 | - | 0.0573 | 0.1028 | 0.0441 | 0.0783 | 167 |
| separate 2x65536 (old) | 2 x 65536 | 2 x 65536 | 5 | 9092 | 8444 | 0.0514 | 0.0897 | - | - | 136 |

### How much of each item identity is semantic vs a random hash digit

| arm | feature width | distinct full identities | distinct semantic tuples | share needing the hash | largest semantic group |
|---|---|---|---|---|---|
| mm 2+2 (new) | 5 | 12101 | 8049 | 0.335 | 62 |
| mm 2+2 seed42 (replica) | 5 | 12101 | 8049 | 0.335 | 62 |
| mm 3+3 (new) | 7 | 12101 | 9766 | 0.193 | 36 |
| mm 4+4 (new) | 9 | 12101 | 10769 | 0.110 | 20 |
| text only 4x256 | 5 | 12101 | 6913 | 0.429 | 65 |
| image only 4x256 | 5 | 12101 | 9638 | 0.204 | 41 |
| joint 4096 4x256 | 5 | 12101 | 5858 | 0.516 | 78 |
| separate 2x65536 (old) | 5 | 12101 | 10821 | 0.106 | 17 |

### Vocabulary modality composition (expand of the merge tree)

| arm | merged tokens | type: contains hash | weighted: contains hash | type: cross-modal | weighted: cross-modal |
|---|---|---|---|---|---|
| mm 2+2 (new) | 38847 | 57.9% | 44.6% | 57.8% | 63.5% |
| mm 2+2 seed42 (replica) | 38847 | 57.9% | 44.6% | 57.8% | 63.5% |
| mm 3+3 (new) | 38335 | 38.4% | 29.3% | 62.1% | 65.0% |
| mm 4+4 (new) | 37823 | 27.2% | 21.7% | 64.2% | 65.7% |

### Caveats that must travel with these numbers

- All arms share `actionpiece_vocab_size=40000`, `n_hash_buckets=128`, `num_beams=50`, `n_inference_ensemble=5`, `max_item_seq_len=20`, `epochs=200` with no early stopping, so the protocol matches.
- The legacy "image only" and "joint 4096" baselines encoded images with CLIP ViT-B/32 (512-d); the new arms and the legacy "separate" baseline used Qwen3-VL-Embedding-8B (768-d). Image-side comparisons across that line are not like for like.
- Codebook size differs between the new arms (256) and the legacy separate baseline (65536).
- The legacy separate run never completed its test pass (`torch.AcceleratorError` at 69%), so it has validation numbers only.
- Batch sizes differ: new arms train 128 / evaluate 32 (4+4 evaluates at 16 to avoid a beam-search OOM); legacy arms used 256 / 64. Batch size changes batching, not the metric.
- Every legacy arm peaks late (best epochs 136-172), so intermediate-epoch rankings do not predict final rankings.

## Follow-up options, in order of value

1. **Finish the 2+2 and 4+4 arms.** Whether the three points are monotone is the
   cheapest and most informative result available, and needs no new experiment.
2. Only then consider the large-codebook variants
   (`experiments/mm_sep33_cb4096_beauty.yaml`,
   `experiments/mm_sep33_cb65536_beauty.yaml`, prepared but not launched). Their
   original rationale — that 256-entry books starve the representation — is now
   weakened by the cardinality measurements above.
