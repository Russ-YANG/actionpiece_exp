# Recovery guide — multimodal late-fusion run

Everything below is for `/home/lzr/science/actionpiece_exp`.
The experiment is driven by four detached processes; if the machine restarts they
are all gone and must be relaunched in the order below.

## Normal state

```
driver      bash mm_fusion/mm_driver.sh       (schedules the three arms)
training    python main.py --run_id=beauty_mm_sep<arm>_train
finisher    bash mm_fusion/mm_finisher.sh     (wait for all, build the table)
postfinish  bash mm_fusion/mm_postfinish.sh   (splice the table into the report)
```

Check them with:

```bash
cd /home/lzr/science/actionpiece_exp
ps -eo pid,etime,cmd | grep -E 'bash (mm_fusion/)?mm_' | grep -v grep
tail -3 logs/mm_driver.log logs/mm_finisher.log logs/mm_postfinish.log
```

## Which arm is where

```bash
python mm_fusion/validate_vocab.py                 # all three tokenizers
python mm_fusion/latest_compare.py                 # current epoch vs baselines
for a in 22 33 44; do
  printf '%s: ' $a
  tr '\r' '\n' < logs/mm_sep${a}_train.log 2>/dev/null \
    | grep -aoE '\[Epoch [0-9]+\] Train Loss: [0-9.]+' | tail -1
  echo
done
```

## Restart the driver

The driver waits for whatever is still running and then trains the remaining
arms. It is safe to relaunch it: it re-checks liveness rather than assuming.

```bash
cd /home/lzr/science/actionpiece_exp
nohup bash mm_fusion/mm_driver.sh > /dev/null 2>&1 &
```

`mm_driver.sh` trains arm 22 with `eval_batch_size=32` and arm 44 with 16 (the
9-slot beam search does not fit in 24 GB at 32).

## If a run dies during its test pass

The legacy separate baseline lost its test metrics exactly this way
(`torch.AcceleratorError: CUDA error: unknown error` at 69%). Re-run only the
test pass from the best checkpoint:

```bash
cd /home/lzr/science/actionpiece_exp
bash mm_fusion/rerun_test.sh 33 16      # arm, eval_batch_size
```

It writes `logs/mm_sep<arm>_testonly.log` and logs `Test_Metric/*` into a new
TensorBoard run named `beauty_mm_sep<arm>_test`.

Note the pipeline computes the real test batch as
`eval_batch_size // n_inference_ensemble`, so `16` means a batch of 3 with the
5-ensemble setting.

## Restart ComfyUI

It was stopped to free ~21 GB of RAM for the parallel tokenizer builds and the
training. Restart it the same way it was running:

```bash
cd /home/lzr/ComfyUI && ./venv/bin/python -u main.py --listen 127.0.0.1 \
  --port 8195 --vram-headroom 1
```

The exact original command line is recorded in
`logs/comfyui_restart_note.txt`.

## Where the results land

| file | contents |
| --- | --- |
| `mm_fusion/RESULTS_TABLE.md` | main comparison + diagnostics + caveats |
| `mm_fusion/RESULTS.md` | the full report with the table spliced in |
| `mm_fusion/RESULTS_DRAFT.md` | methodology, falsified hypotheses, prediction |
| `tensorboard/AmazonReviews2014/ActionPiece/beauty_mm_sep*` | learning curves |
| `ckpt/beauty_mm_sep*_train-*.pth` | best checkpoints |

## Rebuilding from scratch

```bash
python mm_fusion/patch_tokenizer.py        # code fix (backup is kept)
python mm_fusion/gen_mm_configs.py         # the three arm configs
bash   mm_fusion/run_tokenizer.sh 33       # per-arm tokenizer build
```

Input embeddings are symlinked into
`cache/AmazonReviews2014/Beauty/processed/` from
`RPG_KDD2025/input/Qwen3VL-Beauty-Raw-Embeddings/`; see `mm_fusion/README.md`
for the provenance argument.
