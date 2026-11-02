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
