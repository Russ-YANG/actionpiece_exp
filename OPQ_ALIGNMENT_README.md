# OPQ4 + T5：FLASH式训练辅助对齐

这是对现有OPQ baseline的单组件迁移，不是FLASH完整复现。

## 行为

- 原OPQ4、第五位消歧码、T5历史token展开、自回归五特征+EOS目标、全词表CE、beam约束和ensemble都保留。
- 只对训练历史中实际出现的商品做对齐；各商品四个语义token在T5共享embedding表中查表，按码位顺序拼接并投影至d_model。消歧码不参与对齐，原始文本冻结并投影至同一维度。
- `L = L_CE + alignment_weight * (-mean cosine(z, phi(x)))`。
- 历史商品重复出现时按出现次数计。辅助分支的固定码位顺序不受history feature shuffle影响；商品行号由完整五特征确定，四码碰撞不会导致文本取错行。
- 训练日志和TensorBoard增加CE、负cos损失、加权损失、平均cos。对齐统计按有效历史出现数加权；CE诊断同样按历史出现数加权，可能不等于旧日志按batch平均的CE。总loss可以为负，不能跨lambda比较总loss来判断推荐质量。
- 负cos没有负样本：两侧投影都把所有商品映射到同一方向（塌缩）时平均cos同样接近1。因此另记三项in-batch诊断（no_grad、按固定步长取最多1024个出现、不消耗RNG）：`alignment_negative_cosine`为不同商品行之间的平均cos，`alignment_cosine_gap`=正对cos−负对cos，`alignment_top1`为每个z在batch文本中最相似者恰为本商品的比例。同一商品的重复出现不算负对；四码碰撞的不同商品仍算负对，所以碰撞会压低top1上限。判读：cos升高而gap≈0属于塌缩/共同方向，不是商品级对齐。
- 辅助层在隔离CPU RNG上下文中初始化，不推进原T5/dropout/数据采样随机流。lambda=0直接保留原CE计算图。
- 评测与推理沿用原路径，不运行辅助分支；训练产生的共享embedding变化仍影响推理。新checkpoint包含两个辅助投影层，冻结文本／SID缓冲不写入checkpoint，因此恢复仍需相同的来源文件。

## 三个配对配置

`experiments/opq_alignment_beauty_seed2024_lambda{0,01,02}.json`分别对应0、0.1、0.2。

全部从`opq_baseline_beauty_text4_patience50_seed2024.json`继承相同设置，原配置不修改。先只在Beauty text4做匹配比较。新增配置含显式的原文本路径与sidecar路径；不要仅靠文件名替换向量。没有对齐字段的旧入口继续使用原模型。

## 来源检查

检查原始OPQ bundle manifest的embedding_source、4x256结构、dimension、normalization，与alignment sidecar一致。通过原bundle记录的SHA256绑定原始embedding、sidecar、id_mapping、all_item_seqs；检查行序、大小及有限非零向量。拒绝API来源与synthetic fixture；不自动生成embedding或重新拟合OPQ。检查读取一次完整embedding，不消耗GPU。

在仓库根目录，用原训练环境运行：

```bash
python scripts/opq_alignment.py check --config experiments/opq_alignment_beauty_seed2024_lambda01.json
```

本地checkout若没有真实缓存，这一步会明确失败，不能拿合成测试当作真实来源验收。源数据搬迁但内容相同时可更新配置路径；源文件内容变化必须重新审计其与OPQ对应关系。

## 运行边界

本改动只准备代码；不自动push、同步或提交作业。GPU smoke/train继续要求CUDA与Slurm，保留原提交前commit／clean检查，并额外拒绝`genrec/`、`scripts/`下未跟踪的代码文件（`git diff HEAD`看不到它们）；sbatch同步检查。manifest新增`code_sha256`，记录baseline入口、对齐入口、对齐模块与trainer的哈希。`results/opq_alignment/`已加入.gitignore，前一组输出不会挡住后一组。提交前检查squeue、保护既有作业，并确认canonical Sheet与真实artifact。新launcher沿用原1 GPU/8 CPU/16 GB/24h申请模板；这些是申请上限，不是实测耗时。

单次GPU smoke示例（仅在已分配的Slurm GPU作业内）：

```bash
python scripts/opq_alignment.py smoke --config experiments/opq_alignment_beauty_seed2024_lambda01.json
```

smoke与train都会独占创建output_dir；如先smoke，复制配置并将output_dir和run_id改成独立smoke目录，不复用三组正式目录。

`opq_alignment.sbatch`要求显式设置`OPQ_BASELINE_COMMIT`和`OPQ_ALIGNMENT_CONFIG`，不提供自动默认训练组。提交者先创建`logs/opq_alignment`，确认脚本内历史服务器路径／account／partition／QOS仍适用。三组都重新训练；测试集不用于选择lambda。先比较同seed验证指标，再按已记录的实验计划做独立种子确认。

### 多卡并行（独立任务，不是DDP）

`scripts/opq_alignment_parallel.sbatch`沿用`opq_text4_patience50_four_seeds.sbatch`的模式：一个作业内`srun`起N个独立任务，每任务1张A100、8 CPU，各跑一份配置；`scripts/run_opq_alignment_task.py`按`SLURM_PROCID`分配配置、清除DDP环境变量并收窄`CUDA_VISIBLE_DEVICES`。默认N=3，即λ0/0.1/0.2三组（3 GPU、48G、24h申请上限）。一组失败不终止其他组（`--kill-on-bad-exit=0`）。每个任务开始前都会检查整份列表：数量等于任务数、都含`alignment_weight`、output_dir互不相同。

```bash
mkdir -p logs/opq_alignment
OPQ_BASELINE_COMMIT=$(git rev-parse HEAD) sbatch scripts/opq_alignment_parallel.sbatch
```

其他配置集合（例如阶段2的六组）：`export OPQ_ALIGNMENT_CONFIGS=a.json,b.json,...`，并在sbatch命令行用`--ntasks=K --gres=gpu:K --mem=...`覆盖为同一数量；单节点最多4卡，6组需拆成两个作业。登录节点可用`OPQ_ALIGNMENT_DRY_RUN=1 SLURM_NTASKS=3 SLURM_PROCID=0 SLURM_LOCALID=0 CUDA_VISIBLE_DEVICES=0,1,2 python scripts/run_opq_alignment_task.py`检查分配，不加载CUDA。

## CPU工程验证

使用含原框架依赖的Python环境：

```bash
python -m unittest discover -s tests -p test_opq_alignment.py -v
python -m unittest discover -s tests -p test_actionpiece_atomic_target.py -v
python -m unittest discover -s tests -p test_actionpiece_eos_beam.py -v
```

新增测试使用微型真实T5和合成数据，覆盖lambda0初始化／dropout随机流／logits／CE／梯度／一步optimizer等价、同四码不同商品的行号与mask、对齐梯度、checkpoint恢复、beam输出等价和来源错误拒绝。只证明工程接口，不证明推荐增益或真实数据来源。
