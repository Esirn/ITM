# B0 前100步机制对照：入口验收与首次预检停止

日期：2026-10-08。实验主会话已完成方法、计数/故障保障和 CPU 评价入口。首次真实执行在资源预检停止，**worker 未启动，没有新的模型预测或机制结果**；不是等待逐次 GPU 授权。

## 冻结身份与入口

- [协议](../protocol/b0_prefix_mechanism_20261008_v1/package.json) SHA256 `8b32b863da7a530cc71350384663678adfd072f8db53bbb01add176b3040853c`，2639原依赖；F200/U200原权重、12配置与旧噪声计划保持不变。
- [实现 manifest](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v1/implementation.json) SHA256 `c7169161922a025552e72347c54d65868a418133a3b4a56fba911fee1a9a6570`，2654依赖；[CPU验收](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v1/CPU_acceptance.json) SHA256 `c42f8341eb049b540b6f7b80d3b87c2ed98be6ebed7bf8c0cd3759f7f143279c`。
- [本次委托执行记录](B0_PREFIX_MECHANISM_EXECUTION_20261008.json)使用用户长期授权及原共享资源政策，运行目录为 `outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v1`；记录已消耗，不复用该目录。

运行入口：[parent](../../../scripts/run_mdm_prefix_mechanism_bounded.py)、[worker](../../../scripts/run_mdm_prefix_mechanism_worker.py)、[协议/有界执行模块](../../../src/itm/models/mdm_prefix_mechanism.py)、[CPU评价](../../../scripts/evaluate_b0_prefix_mechanism.py)。没有修改任何原有已绑定 runtime/metric 文件。

## 实现行为与验证

36条 F/U/text-only 路径各执行原完整1000步 schedule 中 t999→900 的100步，使用原 posterior 计算、更新顺序与旧地址噪声；输出终端 x899，不解码完整动作。180个指定状态各用 F200/U200 做 factorized batch4 固定状态预测，caption 保持原生 batch1。两权重/各case缓存 observed/reference encoder，每次噪声地址及其实际数组 hash 在使用前核对。

3960 MDM calls/15840 examples、E48、caption12、旧噪声重生成3636次是整个真实队列上限；600秒绝对期限和24GiB allocated cap，原UUID隔离/共享预检/项目锁保持不变。每个调用先 durable reserve，再 actual，再 completed；新 append/fsync journal 支持被杀进程重放，允许末尾不完整记录，拒绝序列缺口。没有自动重试或恢复。

每步原始 f00/f10/f01/f11、x_t、实际组合返回与 mask 都在检查之前落盘，故障保留原始预测。3600步有3600 CFG算术检查，text-only有2400 absent分支检查，加上1380预冻结核心等式，共7380项。每个步骤/状态/trajectory/probe 的 completed 都在相应元数据写出之后标记。父入口完整产物清单为8292文件，不因缺失而缩减计划分母。

[CPU测试](../../../tests/test_mdm_prefix_mechanism.py) **7项通过**：首批5项387.45秒，新增父入口/真实输入 CPU 准备2项6.75秒。涵盖原 full-loop 合成 posterior 前100状态逐项 bitwise 回放、完整3960调用/7380checks/8292产物队列、首锚点失败后保存原始数组/全部未运行分母、调用异常预留计数与 journal 截断尾、期限/错误batch在前向前拒绝、真实F/U输入/权重CPU验证且不构建模型/查询GPU、预检失败时不启动worker。父 supervisor 的绝对 deadline、进程组停止与日志实现沿用原已验收 `supervise`，不伪称重新跑过全部历史测试。

完整**合成 fixture** 的 CPU 评价覆盖180状态、360组/1440分支、720同状态权重对照、180路径对照和36终端，全部 valid；失败 fixture 保留5 diagnostic-only状态＋175 missing，360组/1440分支missing，720/180对照not-comparable，零 primary-valid。它们只验算软件，不是模型结果；[完整 fixture 摘要](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v1/CPU_fixture_evaluation.json)和[故障 fixture 摘要](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v1/CPU_failure_evaluation.json)均已持久保存。实现全部2654依赖在真实启动前再次逐字节核对无漂移。

## 本次真实尝试与完整失败评价

总耗时 **2.83518秒**，只执行1次NVML资源查询；selected GPU空余7560MiB、GPU利用率 **99%**、显存控制器28%。空余显存超过4096MiB门槛，但GPU利用率超过已冻结的50%共享门槛，于第一次资源样本即停止，compute-process query也未执行；不能将空 `compute_processes` 列表解释为无其他进程。

这是计算负载门槛，不是要求完全空闲；既有进程只要资源观测满足共享条件仍可以运行。不干预其他进程，不改变科学容差，也不在本次协议的 `automatic_retry=false` 边界外自动重试。[原始预检](../../../outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v1/runtime/preflight/result.json)、[队列状态](../../../outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v1/runtime/status.json)、[执行摘要](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v1/run_summary.json)保留原位。

worker/context/receipt 都未创建，MDM/E/caption/noise 各 reserve/actual/completed 均为0；36路径、3600步、180状态、360组/1440分支、7380checks全部 `not_run_after_stop`。评价保留180 missing状态、360 missing组、1440 missing分支、720权重/180路径不可比对照、36 missing终端；没有读取合成 fixture 当真实结果。[失败评价摘要](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v1/summary.json)及[评价artifact index](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v1/artifact_index.json)。此阶段科学判决为 inconclusive；原F主配置行为 not_supported 与全部负例不变。

**唯一下一任务：在共享资源条件满足时，沿用此已验收实现和CPU验收，建立新的执行记录/目录运行该固定队列，再评价完整结果。** GPU使用权限仍有效，无需用户再次手动批准；不重做无变化的CPU测试/准备，不重跑已完成F/原生诊断，不追加训练或更改论文主张。
