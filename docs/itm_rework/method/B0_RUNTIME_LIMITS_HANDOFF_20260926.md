# 01 → 02：B0 运行限额与 CPU 故障测试

读取 `8046f5e`。其对 `1d7a32d` 的输入验收无需重做，本批只落实运行限制。
方法提交可用 `git log -1 -- docs/itm_rework/method/B0_RUNTIME_LIMITS_HANDOFF_20260926.md` 定位。
契约仍为 `itm-rework-v0.3-draft`；02 的 package、预算、36 槽位 execution plan、数据和历史结果均未修改。

## 实现

新增 `scripts/run_mdm_b0_bounded.py` 消费 02 原 `execution_plan.json`，复用一条训练模板和 36 条单 case 采样模板。
默认只准备文件，不执行 GPU 预检或子进程。实际执行额外要求外部用户授权记录；本轮未创建该记录。
训练/采样 CLI 新增 `--run-context`，canonical 实际执行必须走受限运行器；原 CPU `--input-only-report` 保持可用。

| 限制 | 实际实现 |
| --- | --- |
| 训练 ≤200 updates | 每次 batch 前检查余量，optimizer.step 前预留一次 attempted；有限参数/optimizer 状态检查后记 completed；第 201 次调用拒绝 |
| 训练 ≤3600 秒 | 父进程在启动子进程前固定 monotonic deadline，包含 Python 导入、加载、计算、保存；不在模型开始时重置 |
| 全部采样 ≤10800 秒 | 36 子进程共用同一个绝对 deadline，每条加载/保存开销也计入；前一条耗时不返还或重置 |
| 总运行 ≤14400 秒 | 两阶段同时受从训练启动前计算的总 deadline 限制；阶段间 CPU 核验开销也占总时长 |
| 36,000 去噪调用 / 144,000 分支样本前向 | 子进程每次调用前预记，单槽位 ≤1000 / 4000；父队列累加并保存，失败后不再运行；这是扩散 backbone 分支计数，预编码开销由时间/显存约束覆盖 |
| 单 GPU | 显式物理 UUID、`CUDA_VISIBLE_DEVICES` 仅该设备、子进程 `cuda:0`、可见设备数必须为 1；全队列持有同 UUID 的互斥锁 |
| ≤24 GiB allocated | 模型分配前调用本机 PyTorch 2.5.1 的 `set_per_process_memory_fraction`，native allocator；较小设备使用更小上限；同步检查峰值，不在采样开始时重置加载峰值 |
| 超时/阻塞 | 父进程按剩余绝对期限 wait，超时立即 SIGKILL 整个子进程组；不依赖 Python 循环能及时返回 |

显存口径沿用 02 的 **PyTorch allocated memory**，不是 nvidia-smi 中含驱动/context 的全部设备占用。
GPU allocator 设置顺序和上限已用 mock 验证；本轮没有真实 GPU 限额实验。硬杀时记录最后观测的峰值，不声称能在死进程上恢复更晚的峰值。
GPU 预检代码将核对 UUID、空闲显存和已有 compute processes；互斥锁协调本运行器实例，不宣称能阻止其他用户另起程序。

训练在 backward 前拒绝非有限 loss，在 optimizer.step 前拒绝非有限梯度或溢出的梯度范数；更新后还核对参数与 optimizer 状态。
采样在每次 DDPM 调用检查 posterior/样本，在写文件前检查解码 joints。任何 OOM、非有限值、身份不符或其他异常均停止当前运行，保存错误类型/原因；无自动重试或 CPU 回退。

## 产物与失败分母

启动前 `runtime/status.json` 就包含全部 36 槽位。每条先标 running，只有子进程成功、receipt 完成且 NPZ/JSON 的 hash 验证通过才标 completed。
失败槽位保存原因、是否真的启动、返回码、receipt/已发布文件；后续槽位标 `not_run_after_stop`。超时前根本未启动的槽位另有 `launched=false`，不能计作一次生成尝试。
所有加载/计算均为单次尝试，不替换 case、不自动重新运行失败槽位。

checkpoint、NPZ 和 JSON 先写同目录临时文件、flush/fsync 后原子发布，且不覆盖已有正式文件。失败中断的 `.partial-*` 文件保留，不冒充完成产物。
既有每 epoch checkpoint 继续独立保存：中途停止时保留最后已完整发布的 epoch 文件和计数 receipt，**不承诺为失败 batch 补存权重**，也不尝试抢救 OOM/NaN 后的模型。
只有完成 200 次更新、最终 checkpoint 标记正确且训练进程正常完成，才能进入固定验证。即使最终文件刚写完，若随后触及截止时间，失败 receipt 仍会阻止启动采样。
部分 checkpoint 永不自动提升为最终模型。最终 checkpoint hash 在后续每个采样子进程开始时重新验证，防止队列中途换模型。

每个子进程保存 `.context.json`、`.receipt.json`、`.log`，记录绝对期限、GPU 身份、输入/代码 hash、预留/完成计数、峰值、已完成 artifact hash 和错误。
状态文件是运行进度；并非逐步训练恢复协议。目录不得复用，未实现续跑。

## 实际 CPU 验证

最终 **23 项限额/故障测试通过**；联合 **166 passed, 23 warnings，无 skip**。
摘要 `b0_limits_cpu_report_20260926.json`，完整日志/XML 在 `logs/b0_limits_regression_20260926_v2.*`。
自有代码、公共依赖和产物 hash 快照为 `code_snapshot_b0_limits_20260926.json`。
初批 22 项日志和 165 项联合日志保留，最后新增“一份授权绑定一个目录/GPU”的测试后得到最终结果。

- 标量 CPU SGD 真正调用 200 次 step；第 201 次被拒绝且参数不变。
- 注入 NaN loss、Inf gradient、有限梯度但范数溢出、更新后 NaN；更新前故障没有调用 step，更新后异常不提交 completed 或 checkpoint。
- 在真实 `sample_addressed_ddpm` 循环搭配微型假 diffusion 注入步数/分支上限、OOM、NaN、deadline；核对真实调用次数，没有额外一次 forward。
- CPU 子进程队列先完成一个文件，再注入 OOM/NaN/挂起：第一条文件保持不变，第二条停止，第三条不启动；所有 context 使用同一个截止时间。
- 实际超时杀进程组，子进程派生的进程也未能写出延迟 marker；不是仅 mock TimeoutExpired。
- 检查过期期限不启动进程、原子写失败保留 partial、部分 checkpoint 拒绝、身份变化在模型设置前拒绝、GPU 隔离/allocator 调用顺序。
- 准备模式不查询 GPU；没有授权参数的 execute 在创建运行目录前拒绝。

这些是明确授权的 CPU 软件故障 fixture，含小张量梯度更新和假 diffusion；**真实 B0 训练更新、真实模型生成、GPU 作业/预检均为 0**。

## 复现和待核验入口

```bash
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_run_limits.py

/home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_b0_bounded.py \
  --plan docs/itm_rework/protocol/canonical_method_acceptance_20260925_v1/execution_plan/execution_plan.json \
  --run-dir <全新准备目录>
```

完整 166 项联合命令为上一批 `CANONICAL_B0_HANDOFF_20260925.md` 所列最终联合命令，在测试文件列表前追加 `tests/test_mdm_run_limits.py`；
本次额外使用 `--junitxml=docs/itm_rework/method/logs/b0_limits_regression_20260926_v2.xml` 并将 stdout/stderr 保存到同名前缀 `.txt`。

本轮实际仅准备目录 `outputs/itm_rework/method/b0_limits_prepare_20260926_v2/`，前一准备版本 v1 保留。
最终状态表副本：`b0_limits_preparation_20260926_v2.json`，所有 36 槽位均 planned，authorization 为 null。
它是供审阅的准备结果，不能复用这个已有目录启动作业；执行需要全新目录。

将来预算获授权后，授权记录需包含 `authorized=true`、`approved_by=user`、实际用户指令引用、
原 `budget_sha256`、原 `execution_plan_sha256`、唯一绝对 `run_directory` 和选定 `gpu_uuid`。
记录由实际授权产生；运行器不会把旧预算中的 pending 状态改写成批准。一次授权不能通过更换目录或 GPU 自动重用。
执行参数为同一 `--plan`、授权对应的新 `--run-dir`，再加 `--execute --authorization-record <记录> --gpu-uuid <UUID>`。
运行器在任何子进程启动前做 GPU 预检，失败即记录并停止。本轮未执行这一命令。

请 02 核验本批具体限额、失败状态与保留策略；无需再对同一输入接口重复确认。
核验、预算授权及 GPU 预检完成后，才由 01 启动一次 B0。没有新 B0 效果或 B/C 研究证据。
