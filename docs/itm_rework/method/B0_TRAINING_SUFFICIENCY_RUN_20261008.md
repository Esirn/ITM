# 固定状态诊断入口完成；首次运行因GPU计算进程预检停止

**有界入口、完整评价入口和增量CPU验收已完成；一次真实执行在预检阶段停止，尚无新模型诊断结果。** 选定GPU被预检检测到已有计算进程，worker未启动。训练充分性与动态收缩的跨权重诊断仍为证据不足，原F完整弓步失败结论不变。

用户的[长期GPU委托](../GPU_EXECUTION_POLICY_20261008.json)继续有效。此次停止是资源预检结果，不是等待再次手动授权；后续GPU时机由主会话判断，沿用已验收实现并创建新目录/执行记录。旧失败目录和记录不复用，未等待重试、换卡或干预其他进程。

## 身份与运行事实

| 项目 | 核验结果 |
| --- | --- |
| 冻结package | v2 `471376c4799b80975c09faee4a80e2adfa220a6119a73a325761ed2a89029593` |
| 最终implementation | `82df1581f5efe658857c509de3d314803ae81baf770a94fb45331c2dcab7ca76`；2556份依赖全部核验 |
| CPU增量验收 | **26 passed，114.27秒**；完整1152组人工fixture＋故障/缓存/日志重放/父入口/评价分母检查 |
| 具体执行记录 | [B0_TRAINING_SUFFICIENCY_EXECUTION_20261008.json](B0_TRAINING_SUFFICIENCY_EXECUTION_20261008.json)，基于既有用户长期委托创建 |
| GPU | `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`，此前cuda:0对应的物理GPU；本次按UUID隔离 |
| 唯一运行目录 | `outputs/itm_rework/method/training_sufficiency_authorized_20261008_v1` |
| 队列实际耗时 | **1.919640秒**；预检前设定单一绝对期限600秒，上限未扩展 |
| 停止原因 | `RuntimeError: Selected GPU already has a compute process` |
| worker /真实MDM /E /caption调用 | 未启动 /0 /0 /0；无真实worker计数器或allocated测量，峰值NA |
| 状态/组/分支/check | **192/1152/4608/2880 全部not_run_after_stop** |
| 模型效果评价 | 4608分支missing，4608臂内＋4608跨快照对照not_comparable；inconclusive |
| 新训练/权重/完整采样/噪声/解码/重试 | 全部0 |

绑定的原预检入口先校验UUID和free≥4096MiB，再查询compute-process列表；此次第二关停止，控制流表明两次GPU查询已尝试。没有保存原始预检stdout，**实际PID/owner/显存/利用率未知**；不能把本次停止描述为OOM、显存耗尽或要求24GiB全部空闲。没有启动模型worker这一事实由账本、失败位置及context/receipt/worker-log/diagnostic目录全部不存在共同支持。

原package人口、六份checkpoint、caption/IMU变换、随机计划、1e−6软件等式界、全部负例、600秒/24GiB预算和论文主张未改。GPU执行次数为一次；CPU mini-model fixtures不算真实实验结果。

## 实现与增量验证

[运行模块](../../../src/itm/models/mdm_training_sufficiency.py)复用已有V输入校验和MDM非链式loader。先加载共同initial的runtime view，再仅交换已核验的E/adapters；backbone/CLIP只加载一次。T严格null，S保留原O[0]静止差分。文本batch1按12case跨快照共享；控制缓存绑定checkpoint/input/config/dtype/eval，每个快照清空，432次E上限不变。

[worker](../../../scripts/run_mdm_training_sufficiency_worker.py)每组一次原始batch4；禁止训练/q_sample/后验/CFG/解码，先保存独立可读T/P/S/R数组再判断锚点，元数据发布后才登记completed。[父入口](../../../scripts/run_mdm_training_sufficiency_bounded.py)在预检前保存全部人口，使用原GPU锁、预检和硬超时supervisor，按唯一绝对期限监督并核验所有产物，不自动重试。

每次reserve/actual/completed及raw/check/状态变化写入fsync追加日志，最终写完整receipt。父入口可重放最后快照后的日志，保留被硬终止worker的准确计数和raw身份；只允许最后不完整行作为未提交尾部，序号缺失拒绝。26项检查覆盖真实冻结输入/6份CPU权重、1152fixture组/4608预测/2880锚点、12caption/432E、换权重缓存、坏身份、nan/OOM/timeout/write、预留先于调用、父入口一次调用/共同期限、产物缺失/哈希/计数错误、日志尾部/序号，以及Q两噪声先平均、3来源等权和失败人口不按0或通过计数。

[评价入口](../../../scripts/evaluate_b0_training_sufficiency.py)只读取数组/JSON，不导入模型或查询GPU；完整保留4608预测、4608臂内与4608同状态跨快照比较及各3来源计划分母。模型未执行时不生成数值效果。已有主F120行与负例仍由原完整评价保留，不从本次缺失预测改写。

CPU开发中发现：最初完整账本重写过慢，在自身临时fixture目录注入排他文件冲突停止，584次fixture调用/312.43秒；改为持久化日志后完整回放通过。随后写入故障测试发现completed登记早于元数据保存，修正顺序。最终26项全通过，无科学规则/预算变更。原CPU开发记录保留，未将人工fixture冒充GPU结果。

## 交付与下一步

见[实现manifest](../../../outputs/itm_rework/method/training_sufficiency_implementation_20261008_v1/prepared/implementation.json)、[CPU验收](../../../outputs/itm_rework/method/training_sufficiency_implementation_20261008_v1/CPU_acceptance.json)、[最终测试日志](../../../outputs/itm_rework/method/training_sufficiency_implementation_20261008_v1/CPU_tests_final.log)、[原队列状态](../../../outputs/itm_rework/method/training_sufficiency_authorized_20261008_v1/runtime/status.json)、[失败评价](../../../outputs/itm_rework/protocol/training_sufficiency_failed_evaluation_20261008_v1/summary.json)、[标准库交付核验](../../../outputs/itm_rework/method/training_sufficiency_implementation_20261008_v1/delivery_verification.json)。

逐一核验2556依赖与原包身份，重计192状态/1152组/4608分支/2880checks，独立构造全部4608＋4608＋4608评价ID和缺失分母，均一致。此次自检不称独立02/人工验收，也没有模型有效性、训练充分性、额外训练收益或新动作修复主张。

**唯一下一任务：GPU满足既有预检条件时，由主会话自主创建新执行记录/目录，沿用此实现进行一次固定诊断，随后从同状态原数组完成完整评价。** 实现已就绪，资源变化前无需重做冻结、训练、缓存或软件全量验收；有实质改动时按影响更新manifest和增量检查。无需用户再次批准GPU。
