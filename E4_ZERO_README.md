# E4-Zero Beauty control

E4-Zero keeps the original E4 raw input shape while removing item-specific
image information:

```text
Text OPQ4 + NULL image slots x4 + collision hash = 9 raw feature slots
```

The learned image codes occupy `0..255`; each NULL image slot therefore uses
the reserved value `256`. Slot identity remains part of the ActionPiece atomic
feature, so the four NULL positions are distinct tokens. This route reuses the
text OPQ4 artifact but does not read or regenerate image embeddings or image
semantic IDs. Its item-feature, tokenizer, merge-log, and training artifacts
have names distinct from the original E4 experiment.

From the repository root, build the E4-Zero item features and 40K ActionPiece
vocabulary:

```bash
python main.py \
  --config_file experiments/e4_qwen3_vl_8b_text_null4_beauty.yaml
```

Then train with the matched original-E4 budget:

```bash
python main.py \
  --config_file experiments/e4_qwen3_vl_8b_text_null4_beauty_train.yaml
```

Interpretation requires two comparisons:

- Original E4 versus E4-Zero estimates the net contribution of item-specific
  image codes under the same nine raw slots.
- Text-only E1 versus E4-Zero diagnoses the cost of four additional constant
  slots and the resulting ActionPiece/tokenization path.

Because ActionPiece is retrained, post-merge sequence lengths can differ even
when raw slot counts match. Report raw length, post-merge mean/P95 length,
vocabulary/merge statistics, and training/inference cost. Do not attribute a
difference solely to sequence length without those checks.
