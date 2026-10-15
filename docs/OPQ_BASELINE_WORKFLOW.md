# OPQ baseline 开发与 MeluXina 同步

## 仓库与分支

- GitHub：https://github.com/Russ-YANG/actionpiece_exp
- 开发分支：`opq-baseline`
- 本地独立目录：`/Users/russ/Desktop/code/actionpiece/opq-baseline`
- 服务器当前运行目录：`/project/home/p201605/u104824/actionpiece_exp`
- 导入的服务器代码快照：`9046dd6f85115fa77a2219bfdedf15c9009774fb`

原本地目录 `/Users/russ/Desktop/code/actionpiece/action_piece` 保留历史研究工作和未提交修改。新目录是独立 clone，不共享工作区，也不自动复制数据。

## 工作约定

在本地修改代码、检查 diff、完成必要验证并提交，再同步到 GitHub 的 `opq-baseline` 分支。服务器代码修改、作业提交由用户执行；助手可只读核查日志和结果。推送须得到用户对同步的授权，不自动提交训练作业。

数据、embedding、模型权重、checkpoint、日志和运行结果通过独立方式管理，不加入 Git。现有历史实验产物保留。新目录没有数据缓存；本地需要真实数据验证时，明确引用原缓存或单独准备输入，不重新生成 embedding。

当前实验包括文本 OPQ4 baseline，以及独立文本 OPQ2 + 图像 OPQ2 拼接。原始 SID 保留码本身份，下游额外消歧码单独处理。下游复用现有模型与训练代码，不初始化、学习或应用 ActionPiece merge。

## 本地修改后

一条命令一个步骤：

```bash
cd /Users/russ/Desktop/code/actionpiece/opq-baseline
```

```bash
git status --short
```

```bash
git diff
```

选择明确的代码文件暂存，提交后：

```bash
git push origin opq-baseline
```

```bash
git rev-parse HEAD
```

记录完整 commit，作为下一次服务器运行版本。

## MeluXina 首次切换到开发分支

先确认当前工作区状态，以及是否有依赖该工作区的运行中或排队作业。不要修改正在被作业使用的代码；不要删除未跟踪文件来强行切分支。

```bash
cd /project/home/p201605/u104824/actionpiece_exp
```

```bash
squeue -u "$USER"
```

```bash
git status --short
```

确认可切换后：

```bash
git fetch origin
```

首次建立本地分支（已有同名本地分支则直接切换）：

```bash
git switch --track -c opq-baseline origin/opq-baseline
```

```bash
git rev-parse HEAD
```

核对完整 commit 与本地提供的版本一致，再由用户提交作业。

## 后续服务器同步

确认没有依赖当前工作区的作业后：

```bash
cd /project/home/p201605/u104824/actionpiece_exp
```

```bash
git status --short
```

```bash
git switch opq-baseline
```

```bash
git pull --ff-only origin opq-baseline
```

```bash
git rev-parse HEAD
```

工作区有代码修改或 fast-forward 失败时停止检查，不使用 reset --hard、强制推送或 git clean。

## 作业和验证

每次提交作业都固定完整 Git commit。例如四卡实验的提交方式：

```bash
OPQ_FOUR_JOB=$(sbatch --parsable --export=ALL,OPQ_BASELINE_COMMIT="$(git rev-parse HEAD)" scripts/opq22_four_seeds.sbatch)
```

先检查对应 seed 的输出目录是否已有产物。入口拒绝覆盖旧输出；需要新实验时使用独立配置和目录。不要为已经完成的相同配置重复提交。

四卡作业为四个独立单卡进程，训练 seed 为 2024–2027，OPQ 和消歧映射保持固定。每个进程在导入 PyTorch 前隔离 GPU。运行时仍需检查每个 seed 的 GPU_BIND、SEED_START、最终成功标志、Slurm 退出状态及 result.json；语法检查不等于服务器运行通过。

不重复跑 smoke。新错误或影响关键路径的改动才触发针对性检查。
