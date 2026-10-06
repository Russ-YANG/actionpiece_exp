# MeluXina 用户执行步骤

所有 `<...>` 均需替换；先确认环境，再同步固定 commit，smoke 成功后才提交完整拟合。

## 数据同步（代码与数据分开）

源目录：`/Users/russ/Desktop/code/actionpiece/action_piece/cache/AmazonReviews2014/Beauty/processed/`。
目标：`$OPQ_REPO/cache/AmazonReviews2014/Beauty/processed/`，其中 `OPQ_REPO` 先在服务器确认。
只传以下四个文件，保留 basename；不传 `item.*.feat`、merge vocabulary、模型权重：

1. `qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.sent_emb`
2. 同名 `.sent_emb.manifest.json`
3. `id_mapping.json`
4. `all_item_seqs.json`

示例（**本机**执行，先填写服务器路径和登录名；数据传输不触发计算）：

```bash
OPQ_DEST='<CONFIRMED_USER>@login.lxp.lu:<CONFIRMED_REPO>/cache/AmazonReviews2014/Beauty/processed/'
OPQ_SOURCE='/Users/russ/Desktop/code/actionpiece/action_piece/cache/AmazonReviews2014/Beauty/processed'
OPQ_STEM='qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa'
scp -P 8822 "$OPQ_SOURCE/$OPQ_STEM.sent_emb" "$OPQ_SOURCE/$OPQ_STEM.sent_emb.manifest.json" "$OPQ_SOURCE/id_mapping.json" "$OPQ_SOURCE/all_item_seqs.json" "$OPQ_DEST"
```

端口也是历史线索，先确认当前连接；目标目录需先创建。已有目标文件先检查/备份，传入新目录后
校验，不盲目覆盖。原 sidecar 的历史 AutoDL 路径仅保留为来源元数据，程序从 config 读取当前路径。
路径变化允许，内容必须一致。`preflight` 核对 embedding SHA 和 item-order identity。

## 服务器顺序（用户执行）

### 1. 只读确认，不运行重计算

```bash
whoami
pwd
myquota
sinfo -o '%P %a %l %D %c %m'
sacctmgr -n show associations user="$USER" format=Account,Partition,QOS
module -t avail Python
module -t avail FAISS
```

若某命令不可用，回传报错。需要：当前 account、CPU partition、允许 QOS、Python/module、
单作业 CPU/内存/时限限制，以及确认的仓库绝对路径。`p201605` 与
`/project/home/p201605/u104824/actionpiece_exp` 只是历史线索。module 列表不一定揭示所有依赖，
也不证明 Faiss 可导入。可以用本目录 `preflight_meluxina.sh` 汇总。

### 2. 固定 reviewed commit，同步前先看 dirty 状态

本次不自动 push。本地只将新入口、builder 小改、测试和 `opq_only/` 小文件提交到一个 reviewed
commit，不包括已有脏文件或任何数据。GitHub 目标为用户 fork `russ`，不是 upstream `origin`。
当用户明确授权后，再 push 相应分支。服务器设置：

```bash
OPQ_REPO='<CONFIRMED_ABSOLUTE_REPO_PATH>'
OPQ_COMMIT='<REVIEWED_FULL_40_CHARACTER_COMMIT>'
cd "$OPQ_REPO"
git status --short
```

若有未提交修改，停在这里，回传状态并保留它们；不要 reset/clean/stash 自动覆盖。
确认干净且用户分支已发布后：

```bash
git remote -v
git fetch <CONFIRMED_FORK_REMOTE> <PUBLISHED_BRANCH>
git switch --detach "$OPQ_COMMIT"
git rev-parse HEAD
git status --short
```

checkout 后必须确认为指定 commit。没有 GitHub push 授权时，可本机生成仅代码的 Git bundle，
独立传到服务器，import 后仍 checkout 明确 commit；与大文件数据传输分开。
不通过 cherry-pick 产生未经确认的新 commit 后还声称版本相同。

离线代码 bundle（本机执行，`OPQ_COMMIT` 替换成交付的 commit）：

```bash
git bundle create /tmp/actionpiece-opq-only.bundle "$OPQ_COMMIT"
```

传输 bundle 后服务器执行：

```bash
git bundle verify <TRANSFERRED_BUNDLE_PATH>
git fetch <TRANSFERRED_BUNDLE_PATH> "$OPQ_COMMIT"
git switch --detach "$OPQ_COMMIT"
```

bundle 只含 Git 已跟踪历史；新改动仅包含代码/文档，不加入任何数据。它也包括已有仓库历史，
并非“只含此次几个文件”的压缩包；已有完整 checkout 时优先发布 reviewed 分支再 fetch。

### 3. 最小依赖与路径检查（登录节点可做安装/轻量预检）

先加载**已确认** Python module（若不需要 module 则跳过）；使用 Python 3.10/3.11。

```bash
module load <CONFIRMED_PYTHON_MODULE>
python3 -m venv .venv-opq
.venv-opq/bin/python -m pip install -r opq_only/requirements.txt
export OPQ_PYTHON="$PWD/.venv-opq/bin/python"
export OPQ_COMMIT="$(git rev-parse HEAD)"
"$OPQ_PYTHON" -c 'import faiss,numpy,sys; print(sys.version,faiss.__version__,numpy.__version__)'
"$OPQ_PYTHON" scripts/opq_only.py preflight --config opq_only/beauty_opq4.json
```

preflight 只读约 35MB embedding / JSON，不做拟合或 embedding 生成。
若本地集群政策要求所有数据检查也在计算节点运行，把该 preflight 放进以下作业即可。

### 4. 短时 CPU smoke（不读取真实 Qwen data）

先填写已确认值；资源请求是起始工程估计，须符合实际限制：

```bash
export OPQ_ACCOUNT='<CONFIRMED_ACCOUNT>'
export OPQ_PARTITION='<CONFIRMED_CPU_PARTITION>'
export OPQ_QOS='<CONFIRMED_ALLOWED_QOS>'
mkdir -p logs/opq_only
sbatch --account="$OPQ_ACCOUNT" --partition="$OPQ_PARTITION" --qos="$OPQ_QOS" --chdir="$PWD" --cpus-per-task=1 --mem=2G --time=00:05:00 --output=logs/opq_only/smoke-%j.out opq_only/job.sbatch smoke
```

如果没有需要指定的 QOS，确认后移除 `--qos` 参数；不要猜一个名字。
job 使用当前导出的绝对 `OPQ_PYTHON` / `OPQ_COMMIT`，验证 HEAD 和 tracked 文件未修改。
结果在 `results/opq_only/smoke_JOBID`，标注 synthetic。回传 job ID、日志、sacct 退出状态，
看到 `OPQ_JOB_SUCCESS` 且 ExitCode `0:0` 后再运行完整任务。

### 5. 用户提交完整拟合

```bash
sbatch --account="$OPQ_ACCOUNT" --partition="$OPQ_PARTITION" --qos="$OPQ_QOS" --chdir="$PWD" --output=logs/opq_only/fit-%j.out opq_only/job.sbatch fit opq_only/beauty_opq4.json
```

默认 4 CPUs / 16GB / 30min 是待 smoke/log 校准的资源申请，不是实测完整时长或最低内存。
配置线程数不能超过 `SLURM_CPUS_PER_TASK`。需要增加时限/内存时只改资源，保留量化参数。
最终 job 已含重载全目录 validation；没有启动任何推荐训练。

### 6. 检查结果

```bash
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
cat logs/opq_only/fit-<JOB_ID>.out
cat results/opq_only/beauty_local8b_d768_opq4_seed2024/manifest.json
```

回传：完整日志、sacct、manifest、`collisions.json`。若传回全部 bundle，我再核验 SID/mapping；
重编码验证应在计算节点执行，而不是登录节点反复做全目录计算。
仅在服务器成功日志和产物检查后称“已在 MeluXina 复现 OPQ SID 流程”。
