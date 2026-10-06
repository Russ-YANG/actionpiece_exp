# OPQ-only Semantic IDs on MeluXina

## 事实、选择和未验证部分

代码事实：`genrec/dataset.py` 的 leave-one-out 训练序列为 `sequence[:-2]`；
末尾倒数第二项是 validation target，最后一项是 test target。拟合集合是所有用户训练序列
出现物品的并集，不是“删除任何曾作为测试 target 的物品”：同一物品可以在其他用户的训练历史中出现。
这不是时间全局切分；若要求全局时间隔离，需要另行确定切分，不能偷偷更改当前协议。

`tokenizer.py::_emb_to_sem_id` 和已有独立 `scripts/build_opq_sem_ids.py` 使用
`OPQM,IVF1,PQMx8`，inner-product metric；OPQ 内部和最终 PQ 聚类仍涉及量化误差。
IVF 默认 `by_residual=True`，保存的 PQ 码来自旋转向量相对一个 IVF 中心的残差。
M 个码本不是逐层 residual quantization，不能称为 RQ 的粗到细层级。
本入口保留原 factory / seed 逻辑；不换成 `OPQM,PQMx8`。Faiss 保存后 OPQ rotation 可读回为一般 `LinearTransform`，因此 encode-only 的训练参数
从原拟合 manifest 继承，不从反序列化类型推断。Faiss 的 standalone encoding
经过 pretransform；IVF1 的 list-id 前缀为零字节。本地测试对照真实 inverted lists 并显式乱序 ID。
参考：[Faiss IndexPreTransform](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexPreTransform.cpp)、
[Faiss IndexIVFPQ](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVFPQ.cpp)。

最小改动：从已有 builder 抽出 `create_quantizer`（含 Faiss 三处聚类 seed），原 CLI 保留；
新增独立编排入口，仅读取 numpy/Faiss/标准库和 builder，完全不导入 `genrec`，不初始化 merge、
hash 特征、模型或训练 pipeline。旧 builder 现在明确拒绝非 256 码本，以免 packed bytes 误读。

设计选择：示例为现存 Beauty local Qwen **768D、OPQ4×256、seed=2024、4 CPU threads**。
这不是选出的最优维度。新长度仅改配置，不扫描。输入已经 `truncate_then_l2`；本入口不额外做
截断、归一化或 PCA。OPQ 的 Faiss 默认迭代/采样参数在 manifest 中记录。固定 seed 不保证
不同平台/Faiss/线程数逐字节相同；跨系统验收优先看本机保存重载一致及来源、结构完整性。

尚未验证：MeluXina 当前资源、模块/依赖安装、真实完整 OPQ4 拟合与性能。
合成测试不是 Qwen 或推荐质量证据。该工作不复现 ActionPiece merge/论文贡献/推荐效果。

## 最小依赖与输入合同

Python 3.10/3.11，`numpy==1.25.2`, **CPU** `faiss-cpu==1.8.0`；本地实际运行的是 Python 3.10.20。
不需要 Torch、transformers、Qwen 权重或 ActionPiece 的全套 requirements。
服务器若不能联网，从能联网的同平台 Linux/Python 环境准备 wheelhouse，不使用 macOS wheels。
其他 Faiss 版本先跑 smoke，记录版本差异，不能直接当作历史位级复现。

`--config` 为 JSON；相对路径都从仓库根目录解析。输入：

- `embedding_path`：C-order little-endian float32 原始二进制，无 header；shape `[N,D]`。
- `id_mapping_path`：`id2item` 为 `["[PAD]", item1, ...]`；`item2id` 是精确逆映射。
  第 r 行对应 `id2item[r+1]`，没有 PAD embedding 行。item ID 必须是唯一字符串。
- `embedding_manifest_path`：local Qwen 模型及代码 revision、instruction identity、维度、归一化、
  N 与 item-order identity。用原 sidecar 文件；不是仅凭文件名认定 Qwen。
- `item_seqs_path`：`user_id → [item_id, ...]`，与 mapping 一致，保持既定顺序。
- `split_rule` 必须明确为 `leave_one_out_sequence_minus_last_two`；不自动猜测切分。

检查尺寸、映射、有限数值和 provenance，并核对已存在的 item-order SHA；该 order fingerprint
无法单独证明二进制行顺序，依赖 encoder 导出时记录的正确性。现有 sidecar 提供 embedding
SHA256，运行前校验内容绑定。输入文件在运行期间须保持不变。
输入 SHA 在 preflight/每次完整运行或复用时计算一次，用于跨系统传输完整性和避免旧产物错误复用；
不会给每个中间步骤计算 hash。

至少 256 个 distinct fitting items 才能拟合 256 centroids；低于 Faiss 建议的 39×256 发出警告。
smoke 采用 **10240 fitting / 10242 catalog / 16D** 的随机向量，故不是无法聚类的微型数据；
仅合成测试 fixture 使用明确标注的 synthetic provenance，不是伪造真实 encoder 证据。

## 输出与复用

新目录由入口独占创建；不覆盖现有目录。成功最后写 `manifest.json`。不完整目录会拒绝复用；
先检查失败日志，再选择新的目录，不自动删除旧产物。

| 文件 | 内容 |
|---|---|
| `item.sem_ids` | JSON `item_id → [c0,...,cM-1]`，固定码本顺序，整数 0–255 |
| `item.tokens.json` | `book*256+value`，不同码本值相同仍为不同 token |
| `sid_to_items.json` | `"c0,c1,..." → [all matching item IDs]`，包括 singleton |
| `collisions.json` | 碰撞组、重复物品数 `Σ(group_size−1)`、受影响物品数和最大组 |
| `fit_items.json` | 实际拟合物品，按输入行序；encode-only 为 [] |
| `quantizer.faiss` | 已训练 OPQ/IVF1/PQ，ntotal=0，含 rotation/center/codebooks，无 catalog index |
| `manifest.json` | 输入路径/来源/摘要、切分、参数、环境、commit/dirty status、碰撞与重载检查 |

没有配置特殊 token：语义 token 范围 `[0,M*256−1]`，从 `M*256` 起留给下游另行定义。
`restore_tokens` 能从任意排列的带码本身份 token 恢复原始 SID，拒绝重复/缺少码本。
这仅是可逆表示，不是排列不变的推荐模型。

同配置完整目录：先核对内容、配置，重载并对**全部输入**重新编码核对后复用，绝不重新拟合。
历史 `.sem_ids` 缺少量化器时不能反推出 OPQ/PQ；`audit-sid` 只做全目录结构和碰撞审计，
不把它升级成“量化器已验证”。已有 Beauty OPQ32 原样保留，不重生成。
现存 Beauty OPQ4 也原样保留；本次新的 server fit 目的是补齐可复用量化器与 manifest，
不宣称与缺少 seed provenance 的旧 tokenizer SID 完全相同。

新物品：复制并填写 `encode_new.example.json`，指向原 fit bundle；相同 encoder/model revision、
code revision、instruction、预处理、维度与推理设置必须匹配。新 mapping/sidecar 与新行序一致。

```bash
"$OPQ_PYTHON" scripts/opq_only.py encode --config <NEW_ITEMS_CONFIG_JSON>
"$OPQ_PYTHON" scripts/opq_only.py validate --config <NEW_ITEMS_CONFIG_JSON>
```

`<...>` 均为必须替换的占位符，不能直接复制执行。

## 同步和执行

按 [RUN_MELUXINA.md](RUN_MELUXINA.md) 执行六个步骤：确认资源 → 同步固定 commit →
准备依赖和四件套数据 → Slurm smoke → 用户提交完整拟合 → 检查日志与产物。
账号/partition/QOS/module/path 都先确认；不运行登录节点重计算、不自动 push/submit。

## 本地检查和故障定位

```bash
"$OPQ_PYTHON" -m unittest discover -s tests -p test_opq_only.py -v
```

三项测试：码本身份/碰撞、输入边界、足量聚类数据端到端。包含重载、完整复用、编码不拟合、
历史 inverted-list 字节对应、holdout 不参与拟合、错误 encoder identity 拒绝、未导入 genrec。
大目录验收由 `validate` 完成，不用小测试替代真实产物检查。

- mapping/order/dimension 错误：回查四件套来自同一次导出，不按文件名猜行序。
- embedding hash 错误：排查传输或源缓存变化；不要改 sidecar hash 让它通过。
- 聚类样本不足：不是调小 codebook size 规避；回查训练切分或另建明确诊断配置。
- Faiss import/SWIG/sa_encode 错误：回传 Python、numpy、Faiss、compile options，先跑 pins 与 smoke。
- output exists/incomplete：保留目录与日志，使用新 run 目录；完整同配置会验证后复用。
- TIMEOUT/OOM：回传 sacct 和聚类日志，校准 CPU/内存/时限，不宣称成功或悄悄降低量化设置。
- dirty 工作区：不清理历史实验；服务器使用独立干净 checkout 固定 commit。
