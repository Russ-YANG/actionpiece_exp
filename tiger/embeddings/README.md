# Prebuilt Qwen->GRID embeddings

`d1536/merged_predictions_tensor.pt` is the Qwen3-VL-Embedding-8B Beauty
embedding (4096-d prefix-truncated to 1536, renormalised to unit norm) already
**permuted into GRID's item order**, so it can be fed straight to `src.train`:

```bash
python -m src.train experiment=rqvae_train_flat \
  data_dir=data/amazon_data/beauty \
  embedding_path=$PWD/tiger/embeddings/d1536/merged_predictions_tensor.pt \
  embedding_dim=1536 num_hierarchies=3 codebook_width=256
```

Shape (12101, 1536) float32. Regenerate with `bridge/build_perm.py` +
`bridge/bridge_qwen2grid.py` if you have the RPG-side `Text-trunc-1536.sent_emb`.
