# AHSB 第一阶段：P0 / P1 / P2

本阶段验证多粒度历史表示是否值得继续研究，不包含 learned allocator、逐请求硬预算、
recency 分配或预算匹配随机分配（P3/P4）。代码验证通过不等于实验取得提升。

## 固定条件与实现边界

- Beauty，复用历史 E4 的独立 text OPQ4 + image OPQ4 + hash，共 9 个 primitive slots。
- P0/P1/P2 共用 item-local、排除 hash 合并的 40k 词表；不为 P0 缩小词表。
- target 都是 Atomic-9 + EOS；训练 CE 与 beam generation 使用同一个 position-wise 合法词表。
  这是 slot grammar，不是 catalog trie；它不保证所有生成的 slot 组合都对应目录中的真实物品。
- 固定相同的历史物品窗口（20 items），BOS/EOS 与 hash atom 均保留；本阶段不匹配 token 总数。
- 关闭额外的特征／token shuffle 和 inference ensemble，避免与粒度采样混淆。
- 合并路径是现有 ActionPiece `shuffle=none` 的确定性 item-local 轨迹，保留每次合法合并后的
  中间状态；不是枚举所有合法 segmentation，也不是正式 SST 的 contiguous-merge 实现。
- `raw` 取轨迹起点，`full` 取终点，`middle` 取中间合并深度，`random` 在真实存在的深度中均匀采样。
  不假设每个 item 都有 1/2/4-token 三档；无合并规则的 item 只有原始表示。
- 所有表示可展开为同一组 SID/hash atoms；token 变少不表示删除了原 SID 信息。

| 配置 | 训练 history | 默认验证／测试 history |
|---|---|---|
| `ahsb_p0_raw.yaml` | raw atoms | raw atoms |
| `ahsb_p1_full.yaml` | 全部合法 item-local 合并 | 相同静态合并 |
| `ahsb_p2_random.yaml` | 每次出现重新采样合法合并深度 | full，固定用于选择 checkpoint |

训练采样有独立 RNG，不消耗模型初始化 RNG。随机评估由已观测历史和 `history_eval_seed`
确定，与 target、batch 大小、样本顺序和训练 RNG 无关。更换评估 seed 才得到另一组随机评估。

## 1. 先准备词表、候选与统计

在安装好项目依赖的环境中，从 ActionPiece 仓库根目录运行：

```bash
python main.py \
  --config_file experiments/ahsb_beauty_common.yaml \
  --tokenizer_only=true
```

默认要求现有 item feature 缓存。缺文件时直接报出所需路径，不会调用 Qwen 或重新训练 OPQ。
共享 item-local 词表若不存在，会从 training split 构建一次；候选随后保存为该词表旁的
`*.history_candidates.json`。后续运行按词表、priority 与 item states 校验来源并复用缓存。
已有其他目录中的同名产物可通过 `--cache_dir=...` 指定缓存根目录。

候选统计输出：`results/ahsb/beauty_candidate_summary.json`。包含可选长度数量、最短长度分布、
平均 raw／最短长度。这是目录物品的非加权统计，不是训练交互加权的历史长度。
先检查 `n_with_multiple_lengths`，确认 allocator 将来确实有可操作空间。

2026-09-10 本地检查：三套配置指向同一份已存在的 E4 特征；对应 item-local/no-hash 词表尚不存在。
该状态只适用于本次检查的本地 checkout，不代表远程机器状态。

## 2. 依次运行 P0 / P1 / P2

```bash
python main.py \
  --config_file experiments/ahsb_beauty_common.yaml \
  --config_file experiments/ahsb_p0_raw.yaml

python main.py \
  --config_file experiments/ahsb_beauty_common.yaml \
  --config_file experiments/ahsb_p1_full.yaml

python main.py \
  --config_file experiments/ahsb_beauty_common.yaml \
  --config_file experiments/ahsb_p2_random.yaml
```

默认 batch=128，eval batch=32，最多 200 epochs，沿用默认学习率／warmup／early stopping。
若先做真实数据管线 smoke，可在各命令追加 `--epochs=2 --warmup_steps=0`，同时使用不同的
`--run_id` 和 `--result_path`；这种短跑只验证流程，不用于判断方法优劣。
实际 pilot 的训练预算和调参预算应对 P0/P1/P2 一致。

默认结果：

```text
results/ahsb/p0_raw_seed2024.json
results/ahsb/p1_full_seed2024.json
results/ahsb/p2_random_seed2024.json
```

JSON 保存 item-level 推荐指标、checkpoint 路径、配置、每个训练 epoch 的时间／峰值显存，
以及最后测试的历史长度、padding 比例和总评估耗时。`profile_history` 目前仅支持单设备；
多 GPU 训练需关闭此项，不能将各进程的局部计时混作全局效率结果。

长度统计包含 BOS/EOS 和 hash；计时包含 collate、设备搬运、生成与指标计算，不是纯模型
latency。CPU 的 CUDA 显存字段为 null，不是零；这些统计不应直接解释为线上 P95 延迟。
候选冷启动成本不在评估循环计时内，首次 tokenizer 准备耗时应另外记录。

## 3. 同一个 P2 checkpoint 切换评估表示

先从 P2 结果 JSON 中取得 `checkpoint`。下面的 `CHECKPOINT` 需替换为实际文件路径：

```bash
python main.py \
  --config_file experiments/ahsb_beauty_common.yaml \
  --config_file experiments/ahsb_p2_random.yaml \
  --eval_checkpoint=CHECKPOINT \
  --history_eval_granularity=random \
  --history_eval_seed=1 \
  --run_id=ahsb_p2_eval_random1 \
  --result_path=results/ahsb/p2_eval_random1.json
```

此入口跳过训练并严格加载模型参数；仍需提供与 checkpoint 一致的基础模型、SID、词表配置。
可改为 `raw` / `middle` / `full` / `random`。随机策略更换 `history_eval_seed` 重复评估，
报告波动，不挑选最优随机 seed。checkpoint 默认由 full-granularity validation 选择，
切换评估表示后的结果应披露这一点；不能在 test 上挑 policy。

## 4. 多个模型 seed

common 配置把 `semantic_cache_seed` 固定为 2024，且禁止生成新特征。
`rand_seed` 控制模型／训练随机性，改变它不会更换 SID、hash features 或共享词表：

```bash
python main.py \
  --config_file experiments/ahsb_beauty_common.yaml \
  --config_file experiments/ahsb_p2_random.yaml \
  --rand_seed=2025 \
  --run_id=ahsb_beauty_p2_random_seed2025 \
  --result_path=results/ahsb/p2_random_seed2025.json
```

P0/P1 同样设置，避免把不同表示来源混入 seed 比较。更换 seed 或实验条件时显式更换结果路径，
防止覆盖之前的 JSON。单次 P2 提升只支持多粒度训练有用，不证明上下文 allocator 有用。

## 已完成的代码验证

使用已有 `actionpiece-e1` 环境，通过 33 项 unittest，包括：

- 每个合并中间状态可恢复 canonical features、hash 保持 atomic；缓存复用和来源变化失效。
- P0/P1/P2 target 不变、测试样本与标签一一对应。
- 随机评估不依赖 target 或批次顺序，也不推进训练 RNG。
- 小型真实 T5 的 masked CE、梯度、optimizer step 与 beam generation。
- 完整 tiny-data Pipeline 训练、验证、保存 checkpoint、重新加载切换策略评估及 JSON 输出。

```bash
python -m unittest discover -s tests -v
```

本次未运行正式 Beauty P0/P1/P2 训练、未生成完整 Beauty 候选缓存，也未验证 GPU 加速收益。
