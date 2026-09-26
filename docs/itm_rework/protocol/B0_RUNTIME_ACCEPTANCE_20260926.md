# 02 独立验收：B0 运行限额

读取 **`1d44adc`**，公共契约保持 `itm-rework-v0.3-draft`。**本批运行限额通过 CPU 软件验收，无新增返工项。**此前 `8046f5e` 的输入/398 帧统计验收继续有效，不重做同一数据准入。预算授权和真实 GPU 预检尚未完成，未启动 B0。

## 已核验内容

| 项目 | 结论与依据 |
| --- | --- |
| 训练计数 | optimizer.step 前预留 attempted，完成后记录 completed；CPU 标量 SGD 完成 200 次，第 201 次拒绝且参数不变。非有限 loss/梯度/范数在 step 前阻止，更新后参数或 optimizer 状态异常也不记成功 |
| 训练时间 | 父进程在子进程启动前固定 3600 秒期限，覆盖导入、加载、计算、保存；超时杀整个进程组，不依赖训练循环返回 |
| 共享采样时间 | 36 子进程使用同一绝对期限，最多 10800 秒；每条模型加载/保存开销均计入，不逐条重置 |
| 总时长 | 从训练启动前计算 14400 秒总期限；父进程 checkpoint 核验耗时也影响后续可用时间 |
| 采样计数 | 单槽位最多 1000 次去噪、4000 分支样本计数；调用前预留。完整队列上限 36000 / 144000，失败后不继续或补跑 |
| GPU 与显存 | 代码要求指定唯一 UUID、只暴露该 GPU 并使用 cuda:0，持有 UUID 互斥锁；模型分配前设置 native allocator fraction，上限不超过 24 GiB allocated。设备较小时使用更小上限；加载峰值不被采样重置 |
| 原子保存 | 临时文件写完、flush/fsync 后发布，已有正式文件不覆盖；失败的 partial 保留。receipt 完成且 NPZ/JSON hash 核验通过，槽位才记 completed |
| 部分训练 | 保留已经发布的 epoch checkpoint；仅训练正常退出、200 updates 完成且最终标记正确的 checkpoint 能进入采样，不自动提升部分模型 |
| 失败分母 | 启动前已列全部 36 槽位；故障槽位记录原因/是否启动，后续为 not_run_after_stop，不删除、不替换、不自动重试或转 CPU |
| 授权边界 | 准备模式无 GPU 预检或子进程；execute 缺授权时在创建运行目录前拒绝。授权需绑定原预算、原计划、唯一运行目录和 GPU UUID；目录不可复用 |

显存结论是 **PyTorch allocated memory**，不是 nvidia-smi 中包含 context/驱动的全部占用。allocator 顺序/比例及越限检测用 mock 验证，尚无真实 GPU 限额实验。硬杀只能留下最后持久化的计数/峰值，不推断死进程未保存的更晚数值。互斥锁只协调该运行器，不会阻止其他用户启动程序；真实预检仍须检查已有 compute process。

## 02 执行的独立验证

证据目录：`b0_runtime_acceptance_20260926_v1/`。

1. **166 项联合回归通过，无失败、无 skip**，其中运行限额/故障测试 **23 项**；23 条既有 warnings。日志和 JUnit XML 在 `regression.log` / `regression.xml`。包括真实 CPU 子进程超时、派生进程组终止及已完成文件保留，并非全部用 mock TimeoutExpired。
2. 新增独立审计脚本 `scripts/accept_b0_runtime_limits.py`，补测真实顶层运行器控制流程的三个场景：全部完成、训练只有 199 次、第二条采样超时。授权/GPU/子作业在这些场景中均被显式 stub，**不创建真实授权文件，不运行真实训练或采样**。
3. 顶层软件场景验证：所有 sample context 共用同一截止时间，父进程验证开销没有重置总预算；完成路径累计 36000/144000；部分训练没有启动任何 sample；第二条超时后第一条产物保持、后 34 条均未启动。每个采样 context 还绑定同一个最终 checkpoint hash。
4. 实际 `sample_addressed_ddpm` 循环搭配 2×1×3 微型 CPU 张量/假 diffusion 完成 **1000 次调用、1000 次噪声抽取**；第 1001 次被拒绝。短于 1000 步的 completed receipt 被父队列拒绝。它不是 MDM 模型或动作生成实验。
5. 单独调用默认准备模式，36 槽位均 planned、authorization=null。只替换新旧运行根目录后，准备状态与 01 保存报告逐字段相等。准备目录为 `outputs/itm_rework/protocol/b0_runtime_prepare_20260926_v1/`，不可复用来实际启动。

独立报告是 `independent/acceptance.json`。`*_CPU_FIXTURE.json` 中的 completed、微型 checkpoint 和伪 NPZ 仅代表软件控制流程；文件明确带有 fixture-only 标记，不能作为训练或生成结果进入评价。真实 B0 更新、完整动作样本、GPU 作业与 GPU 预检均为 **0**。

## 下一步：授权后由 01 执行

01 不需继续重复输入验收或新增通用防护。本轮已具备可审阅的受限运行器与具体队列。

待用户批准的额度沿用原方案：**一次 B0、200 updates、36 条完整 DDPM 1000 步样本，单 GPU、最多 24 GiB allocated、训练 1 小时＋共享采样 3 小时，总计最多 4 GPU 小时**。未完成样本保留在原 36 槽位分母中。

授权来自用户此前明确要求“预算须获得相应授权”，不是本次新增加的确认流程。批准后由 01 选定现有空闲 GPU 和全新运行目录，另存授权记录，引用实际用户指令并绑定以下不变对象：

- 原预算：`canonical_b0_20260925_v3/b0_budget.json`。
- 原队列：`canonical_method_acceptance_20260925_v1/execution_plan/execution_plan.json`。
- 唯一绝对运行目录与 GPU UUID。

不要改写原预算/计划中的历史 pending 字段，避免破坏已绑定的 package/provenance。实际执行时，由 `run_mdm_b0_bounded.py` 做 GPU 身份、空闲显存和进程预检；失败即记录并停止。GPU 不可用时不自动改用 CPU 或换设备重试。

执行完成后 02 按原固定 36 槽位独立评价全部输出与失败记录；相容 witness 不阻塞本次 B0。真实设备部署标定、严格生成 IMU 闭环、相容编辑和模型有效性均未因软件验收而获得认证。

## 复现入口

```bash
# CPU 审计，目录必须全新；内部 GPU/授权/子作业均为测试 stub。
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python scripts/accept_b0_runtime_limits.py \
  --output <新报告目录> --fixture-dir <新CPU_fixture目录>

# 文件准备，不带 --execute；不会执行 GPU 预检或真实作业。
/home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_b0_bounded.py \
  --plan docs/itm_rework/protocol/canonical_method_acceptance_20260925_v1/execution_plan/execution_plan.json \
  --run-dir <新准备目录>
```

审计脚本同时核对本轮保存的准备状态；迁移机器时应先按命令新建准备状态并明确调整该比较路径，不能将缺失资产伪装成验收通过。
