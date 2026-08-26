# E4 模态融合公平追踪验证报告

日期：2026-08-13

## 结论

经候选结构校正后，现有三个 E4 实验都没有显示“真实图文边界产生了显著更多的直接跨模态 merge”。

- Beauty：真实直接图文率与置换零分布几乎完全一致。
- Sports：真实直接图文率仅高于零均值 0.58 个百分点，单侧精确 `p=0.0857`，不显著。
- NineRec-DY：真实直接图文率显著低于随机 4/4 槽位边界，是零分布中的最小值之一。

因此，原始的 `55%–65% mixed token` 只能说明最终词表中存在大量同时含有图文叶子特征的 token，不能作为 ActionPiece 偏好跨模态融合的证据。

## 更新后的主指标

只把两个操作数仍为纯模态时的 `text + image` 记为一次“首次跨模态融合”。`mixed + text`、`mixed + image` 和 `mixed + mixed` 单独统计，不重复计作首次融合。

主统计量为：

```text
direct_cross_rate = text_image /
  (text_text + image_image + text_image)
```

为校正 4 个文本槽和 4 个图像槽造成的组合数量优势，固定完整 merge 图，穷举 8 个语义槽中任选 4 个作为一侧的全部 70 种有标签划分（等价于 35 种无方向划分）。真实划分为 `{0,1,2,3}` 对 `{4,5,6,7}`。

每一种划分都重新递归分类全部 merge，形成条件零分布，并报告：

- 真实值；
- 零分布均值和标准差；
- `observed / null_mean` 富集倍数；
- 正向富集单侧精确 p 值；
- 双侧精确 p 值。

## 主要结果：首次直接跨模态融合

| 数据集 | 真实图文率 | 置换零均值 | 富集倍数 | 正向单侧 p | 双侧 p |
|---|---:|---:|---:|---:|---:|
| Beauty | 59.15% | 59.16% | 1.000 | 0.4571 | 0.9429 |
| Sports | 59.72% | 59.14% | 1.010 | 0.0857 | 0.1714 |
| NineRec-DY | 54.93% | 59.50% | 0.923 | 1.0000 | 0.0286 |

针对“ActionPiece 是否正向偏好真实图文融合”的主假设，三个数据集均不能拒绝零假设。Sports 有很弱的正向趋势，但未达到 0.05；Beauty 无效应；DY 的方向相反。

## 递归融合结果

### 排除 hash 后的广义融合规则

这里将 `text_image`、`mixed_text`、`mixed_image` 和 `mixed_mixed` 合并，但仍然受 mixed token 的吸收效应影响，因此只作为辅助指标。

| 数据集 | 真实比例 | 置换零均值 | 富集倍数 |
|---|---:|---:|---:|
| Beauty | 71.09% | 72.79% | 0.977 |
| Sports | 68.32% | 72.97% | 0.936 |
| NineRec-DY | 60.49% | 71.94% | 0.841 |

### 最终 mixed token

| 数据集 | 真实比例 | 置换零均值 | 富集倍数 |
|---|---:|---:|---:|
| Beauty | 64.49% | 66.44% | 0.971 |
| Sports | 62.40% | 68.27% | 0.914 |
| NineRec-DY | 54.79% | 66.23% | 0.827 |

三个数据集的真实 mixed 比例都不高于随机槽位边界。Sports 和 DY 的真实边界在这些递归指标上位于置换零分布下端。由于同时查看了多个相关指标，辅助指标的 p 值不作为独立确认性发现。

## 新增的未来训练追踪

ActionPiece 训练日志现可选择在每次 merge 前记录：

- 每类有效候选 pair 数；
- 每类候选共现 priority 总量；
- 每类候选最大 priority；
- 最终胜出的候选类别。

分析器会据此计算实际选择数相对候选数量和共现质量的富集度。旧日志没有这些逐步快照，因此本报告对旧实验使用不需要重训的精确槽位置换校正。

小规模 Beauty 重放的前 23 步与原日志的被选 pair 和 priority 逐项一致，说明追踪本身不改变训练决策。完整逐步重放因本地耗时过长未作为本报告的必要条件。

## 局限性

槽位置换是对现有 merge 图的条件检验，能校正槽位数量和 mixed 递归标注造成的结构性膨胀，但不能回答“正确配对的商品图像是否比随机图像促进融合”。后一个因果问题仍需：

1. 以四位图像 SID 为整体，在 item 之间分层置换；
2. 保持图像码边际分布、缺图状态、流行度和类别尽量不变；
3. 每次重新训练 ActionPiece；
4. 将真实首次融合率与重训零分布比较。

另外，8 个 OPQ 槽未必完全可交换。不同 codebook 的熵和使用不平衡可能不同，因此应补充逐槽熵、有效码数和候选频率诊断。

## 复现

```bash
python scripts/analyze_modality_merges.py MERGE_LOG.jsonl \
  --text-slots 4 \
  --image-slots 4 \
  --hash-slot 8 \
  --bins 10 \
  --examples 0 \
  --exact-slot-permutation \
  --json-output CORRECTED_SUMMARY.json
```

## 产物

| 数据集 | 校正摘要 | SHA-256 |
|---|---|---|
| Beauty | `beauty.e4.corrected_summary.json` | `276b25bdf4247117f4c8a122f6276708211ce04a311be7d35ca06b7d12b2e9a8` |
| Sports | `sports.e4.corrected_summary.json` | `bfddbeec7446d8035872751a2851afee4f52bfddd05ec57373fda99cbbde187b` |
| NineRec-DY | `dy.e4.corrected_summary.json` | `77d44e1ddcf9223703bf46084da4c1b53b3a1d7994d699dd1af9a808ed0ba4f5` |

三个输入日志分别包含 37,823、37,695 和 37,823 条 merge，`malformed_records=0`。
