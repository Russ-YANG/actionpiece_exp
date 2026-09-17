# E1-API: same-checkpoint EOS-aware evaluation

This is an inference-only control. No weights, target grammar, tokenizer,
history features, or item deduplication rules are changed. Historical decoding
remains the default (`beam_search_mode: legacy`). Opt in with
`beam_search_mode: eos_aware`; both modes load exactly the same state dict.

## Search contract

- Score the first EOS using the existing constrained log-softmax. Never add
  probability after that EOS or expand a completed hypothesis.
- Keep up to B unfinished prefixes and a separate top-B completion archive.
  Every EOS extension of a reachable prefix competes for that archive. Only
  non-EOS extensions compete for live slots. This is an explicit change to
  candidate retention; it is not merely truncating legacy outputs afterward.
- No per-hypothesis length normalization is introduced. Ranking uses the sum
  of constrained log probabilities through EOS, including EOS. The public
  `return_score=True` result retains the legacy division by the common
  `max_length - 1`; multiply back by that constant for cumulative scores.
- At the generation horizon, unfinished paths compete with completed paths.
  This preserves the existing no-EOS item decoding fallback. No new catalog
  trie, semantic validity filter, item aggregation, or suffix rule is added.
- Return fixed-width, PAD-filled sequences after EOS. If fewer than K paths
  exist, unreachable slots have score -inf and contain an empty EOS sequence;
  ordinary item decoding rejects these rather than duplicating a real item.
- Finished hypotheses are not evaluated by T5 again. Live slots remain batched;
  unreachable slots can still occupy tensor rows until the batch has no live
  prefixes. This patch is not a GPU speedup claim.

## Paired evaluation

Use the **original E1-API baseline configuration and checkpoint**, including
its original tokenizer/cache, model dimensions, seed, beam count, ensemble
count, batch size and feature-shuffle settings. Do not substitute the H3/T3
training YAML or a local-8B E1 configuration for the API baseline. Supply all
original layered YAML files in their original order if more than one was used.

Replace the two absolute placeholder paths below with the verified artifacts:

```bash
E1_CONFIG=/absolute/path/to/original_e1_api_config.yaml
E1_CHECKPOINT=/absolute/path/to/original_e1_api_checkpoint.pth

python main.py --config_file "$E1_CONFIG" \
  --eval_checkpoint="$E1_CHECKPOINT" \
  --beam_search_mode=legacy \
  --require_cached_item_features=true --profile_history=true \
  --run_id=e1_api_eos_control_legacy \
  --result_path=results/eos_control/e1_api_legacy.json

python main.py --config_file "$E1_CONFIG" \
  --eval_checkpoint="$E1_CHECKPOINT" \
  --beam_search_mode=eos_aware \
  --require_cached_item_features=true --profile_history=true \
  --run_id=e1_api_eos_control_corrected \
  --result_path=results/eos_control/e1_api_eos_aware.json
```

`eval_checkpoint` skips training and strictly loads the existing weights. Both
result files contain the checkpoint, evaluation-only flag, full configuration,
item metrics and the existing evaluation profile. Profiling requires a single
device. Ensure the original frozen tokenizer cache also exists; requiring item
features alone does not prevent rebuilding a missing ActionPiece vocabulary.

Start each run in a fresh process with the same seed, batch size and ensemble
settings so stochastic history tokenization follows the same sequence. Use
unique result paths for repeat runs. Compare NDCG@10 and Recall@10 first; keep
the legacy output as provenance and do not overwrite canonical Sheet results.
Per-example ranking traces and post-EOS penalty diagnostics are not emitted by
the existing result writer; they are not claimed by this patch.

If a gap remains, evaluate the H3/T3 checkpoint under both modes with its own
matching configuration. An inference-only correction does not redo historical
checkpoint selection, which used the original validation decoder.

## Local checks

```bash
python -m unittest discover -s tests -p 'test_actionpiece_eos_beam.py' -v
python -m unittest discover -s tests -p 'test_actionpiece_pilot_pipeline.py' -v
```

The focused checks cover exhaustive variable-length search, EOS score freezing,
completion retention, batch isolation, no-EOS fallback, narrow atomic grammar,
HTPad forced suffixes, strict real-T5 weight loading, and evaluation-only pipeline
execution. They do not establish Beauty metrics or GPU performance.
