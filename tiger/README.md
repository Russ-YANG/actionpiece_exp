# TIGER × Qwen-Beauty（Table 3 的 TIGER 行）

给 Table 3（encoder × 降维方式）补 **TIGER** 这一行用的可复现包。
RPG 那几行用的是 OPQ，TIGER 用的是 **RQ-VAE，这是另一个 backbone + 另一种 SID 方法**，所以单独一条管线。

## 0. 结论速览：这条管线跑的是什么

| 项 | 值 |
|---|---|
| 框架 | [`snap-research/GRID`](https://github.com/snap-research/GRID) @ `2fe3475` |
| backbone | TIGER（GRID 的 `tiger_train_flat`） |
| SID 方法 | RQ-VAE，**3 层 × 256 码本**，再补 1 位去重 → SID 长度 4 |
| 数据 | Amazon Reviews 2014 / Beauty，**12,101 items / 198,502 交互** |
| encoder | Qwen3-VL-Embedding-8B，4096 原生，**截前缀** 到 128 / 768 / 1536 |
| 维度 | 128 / 768 / 1536（师兄定的三个，不用 4096） |
| 训练 | `max_steps=320000`，早停监控 `val/recall@5`，`patience=10`，`val_check_interval=1600` |

## 1. 环境

GRID 自己的 venv 就够（无需额外装包）：

```
torch 2.11.0+cu128   tensorflow 2.18.0   transformers 4.47.0
numpy 2.0.2   scipy 1.15.3   omegaconf 2.3.0   datasets 3.2.0
```

第 2 步（LLM 生成 embedding）**不需要跑**：直接复用 RPG 那边已经算好的 Qwen `*.sent_emb`。

## 2. ⚠️ 最关键的一步：商品顺序对齐

GRID 和 RPG/Qwen 的 Beauty **商品集合完全一致（12,101 = 12,101，零差异），但顺序不同**：

| | 第 0 行 | 第 3 行 |
|---|---|---|
| GRID（按 ASIN 字典序） | `7806397051` | — |
| RPG / Qwen embedding | `B004756YJA` | `7806397051` |

**不重排就会把 embedding 喂错商品。** `bridge/build_perm.py` 用 ASIN 建双射并验证集合相等；`bridge/bridge_qwen2grid.py` 按该双射重排，产出 GRID 要的格式：

```bash
python bridge/build_perm.py        # 验证集合并写出 rpg2grid_perm.npy
python bridge/bridge_qwen2grid.py  # .sent_emb -> GRID pickle
```

产物布局（GRID 的 `src.train` 直接吃这个路径）：

```
qwen_emb/d128/pickle/merged_predictions_tensor.pt    (12101, 128)  float32
qwen_emb/d768/pickle/merged_predictions_tensor.pt    (12101, 768)
qwen_emb/d1536/pickle/merged_predictions_tensor.pt   (12101, 1536)
```

源 embedding 在 `RPG_KDD2025/cache/AmazonReviews2014/Beauty/processed/Text-trunc-{dim}.sent_emb`
（裸 float32，`12101 × dim × 4` 字节；**截前缀 = 切列 + 重归一化到 ‖·‖=1**，已与 RPG 的口径逐元素核对一致）。

## 3. 四步命令

```bash
# 3a: RQ-VAE 训练（约 35 分钟）
python -m src.train experiment=rqvae_train_flat \
  data_dir=data/amazon_data/beauty \
  embedding_path=$PWD/qwen_emb/d128/pickle/merged_predictions_tensor.pt \
  embedding_dim=128 num_hierarchies=3 codebook_width=256

# 3b: 生成语义 ID（约 15 秒）
python -m src.inference experiment=rqvae_inference_flat \
  data_dir=data/amazon_data/beauty \
  embedding_path=$PWD/qwen_emb/d128/pickle/merged_predictions_tensor.pt \
  embedding_dim=128 num_hierarchies=3 codebook_width=256 \
  ckpt_path=<3a 的 checkpoint>

# 4a: TIGER 训练 —— 注意下面这一整串覆盖参数，缺一不可
python -m src.train \
  experiment=tiger_train_flat \
  data_dir=data/amazon_data/beauty \
  semantic_id_path=<3b 的>/pickle/merged_predictions_tensor.pt \
  num_hierarchies=4 \
  trainer.devices=1 \
  data_loading.train_dataloader_config.dataloader.num_workers=1 \
  data_loading.val_dataloader_config.dataloader.num_workers=1 \
  data_loading.test_dataloader_config.dataloader.num_workers=1 \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.every_n_train_steps=200 \
  ++model.should_check_prefix=false \
  extras.print_config=false \
  ++should_skip_retry=True

# 4b: 推理出指标
python -m src.inference experiment=tiger_inference_flat \
  data_dir=data/amazon_data/beauty \
  semantic_id_path=<3b 的>/pickle/merged_predictions_tensor.pt \
  ckpt_path=<4a 的 tiger_best / checkpoint> \
  num_hierarchies=4
```

## 4. 踩过的坑（省你几小时）

### 坑 1：`num_workers=8` 会让第一次验证直接崩

默认配置 `num_workers: 8`，但 **evaluation split 只有一个 tfrecord 文件**。8 个 worker 分 1 个文件，空分片的 worker 立刻抛：

```
tensorflow ... OutOfRangeError  (in DataLoader worker process N)
  → RuntimeError: Caught RuntimeError in DataLoader worker process 1
```

5 分钟就死（正好死在第一次 val）。**必须把 train/val/test 三个 dataloader 全设成 `num_workers=1`。**
成功跑完的 2026-07-07 / 07-13 两次运行用的就是 1。

### 坑 2：崩溃后会被 restart_job 掩盖

崩了之后 GRID 的 `RestartAndLoadCheckpointCallback` 会尝试重启，然后又因为

```
FileNotFoundError: .../metadata/restart_metadata.json
```

彻底放弃，日志最后看到的是那个 FileNotFoundError 而不是真正的 `OutOfRangeError`。
**加 `++should_skip_retry=True`**，让真正的错误直接暴露。

### 坑 3：`+model.should_check_prefix=false` 要写成 `++`

`+` 是"追加新键"，但该键在配置里已存在，Hydra 会报
`Could not append to config. An item is already at 'model.should_check_prefix'`。
用 `++`（覆盖）。

### 坑 4：`max_steps=320000` 是硬上限，模型很早就会饱和

按 3.7 it/s，320,000 步要 **24 小时**。但实测 `val/recall@5` 在 **step ≈ 3,200** 就到顶（0.0558），之后在 0.050–0.055 震荡。早停要 `patience=10 × val_check_interval=1600 = 16,000` 步才触发，所以会白跑很久。
**建议把 `trainer.max_steps` 压到 20000–30000**，或者盯着 `checkpoints/` 里 best 的时间，饱和后手动停。

### 坑 5：上游没有 `rqvae_inference_flat.yaml`

`configs/experiment/` 里上游只给了 `rqvae_train_flat.yaml`，**缺 inference 那份**。本包 `configs/rqvae_inference_flat.yaml` 是补上的，直接拷进 `configs/experiment/`。

## 5. 已得结果（Beauty，seed 2024，RQ-VAE 3×256，OPQ-free）

### Qwen3-VL-Emb-8B（MRL）× RPG × OPQ32 —— 已有，作对照

| 宽度 | test N@10 | test R@10 |
|---|---|---|
| 128 | 0.0449 | 0.0767 |
| 768 | 0.0446 | 0.0757 |
| 1536 | 0.0402 | 0.0697 |
| 4096 | 0.0319 | 0.0541 |

### E5-Mistral-7B（**非 MRL**）× RPG × OPQ32 —— 本轮新增

| 宽度 | test N@10 | test R@10 |
|---|---|---|
| 128 | 0.0438 | 0.0757 |
| 768 | 0.0444 | 0.0770 |
| 1536 | 0.0419 | 0.0738 |

**非 MRL 在 128 维没有崩（只比自己最好宽度低 1.4%）。** 这条结果的含义（师兄定的表述）：
> 说明的是「**增加维度带来的信息收益远小于量化带来的信息损失**」，而不是「MRL 降维能保留信息」。

### TIGER × Qwen-Beauty —— 进行中

| 维度 | RQ-VAE | SID | TIGER |
|---|---|---|---|
| 128 | ✅ 35 min | ✅ 15 s | 🔄 训练中（best 出现在 step 3,200） |
| 768 | 待跑 | 待跑 | 待跑 |
| 1536 | 待跑 | 待跑 | 待跑 |

## 6. 文件说明

```
bridge/build_perm.py            ASIN 双射：RPG 顺序 -> GRID 顺序（含集合相等校验）
bridge/bridge_qwen2grid.py      .sent_emb -> GRID pickle（三步：读、重排、存）
bridge/sid_check.py             SID 张量体检（形状应为 (4, 12101)，取值 0..255）
configs/rqvae_inference_flat.yaml   上游缺失的那份 inference 配置
run_rqvae.sh / run_sid.sh / run_tiger.sh    上面四步的包装脚本
grid_local_changes.patch        GRID 本地改动（忽略空白后仅 642 行，主要是 config 注释汉化）
```
