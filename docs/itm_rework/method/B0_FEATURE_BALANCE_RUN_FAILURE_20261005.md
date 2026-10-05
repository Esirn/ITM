# 01 → 02：F 新单次运行因 GPU 已有计算进程停止

用户对 `82d28a3` 回复“批准”后，01于2026-10-05执行一次绑定的新队列。**预检返回 `Selected GPU already has a compute process`，父入口退出码1，worker未启动。** 从预检前起耗时 **0.976359秒**；模型调用、训练更新及采样均为0。本次授权已消耗，没有重试、换卡、终止其他进程或追加GPU查询。

## 身份与证据

- 接受的实现 `963a132`、软件验收 `e0999b7`，均未改动；155项CPU验收继续有效。
- 最终outputs v2 implementation：`db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`。
- F package：`a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`。
- GPU：`GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
- 原目录：`outputs/itm_rework/method/feature_balance_authorized_20261005_v1`，只有`runtime/status.json`，保持原样。
- [授权记录](B0_FEATURE_BALANCE_AUTHORIZATION_20261005.json)、[历史命令](feature_balance_run_20261005_v1/commands.md)、[父日志](feature_balance_run_20261005_v1/logs/parent.log)、[账本副本](feature_balance_run_20261005_v1/runtime/status.json)、[执行摘要](feature_balance_run_20261005_v1/execution_summary.json)。

本次单次命令经`require_escalated`批准获得设备访问权限后启动；没有先尝试一次再越权重试，没有修改运行入口或manifest。

## 预检结论与边界

保存的调用栈和既有helper控制流表明：第一项指定UUID的显存查询正常返回，通过单行UUID及可用显存≥4096 MiB检查；第二项compute-process查询正常返回，并在结果中匹配到指定UUID，触发明确拒绝。因此本次直接停止原因是**预检检测到该GPU已有计算进程**。

原始stdout/stderr仍未由既有helper持久化，不能提供具体PID、进程所有者、实际显存值或进程用途。未调查或干预该进程。两项查询均位于本次已授权预检范围内，之后没有追加查询。

此结论只适用于本次预检，不能解释2026-10-04查询退出码9的具体原因；旧失败目录和账本hash仍一致，不以本次状态覆盖旧证据。

## 完整失败保留

| 项目 | 计划 | 完成 | 停止后未执行 |
| --- | ---: | ---: | ---: |
| updates | 200 | 0 | 200 |
| exposures | 600 | 0 | 600 |
| samples | 60 | 0 | 60 |
| traces | 240 | 0 | 240 |
| software checks | 48 | 0 | 48 |

原账本无active job，completed receipts和job counts为空。没有worker context/receipt、checkpoint、损失记录、动作或trace；原始账本不存在worker counters，归档未伪造零计数。模型调用0依据父入口在首个job前停止和无worker产物；无worker显存峰值可报告。

[CPU交付核验](feature_balance_run_20261005_v1/delivery_integrity.json)通过：1,145份绑定文件hash一致，manifest/package/批准记录身份一致，原始与归档账本逐字节一致，200/600/60/240/48全部ID、顺序及失败状态与冻结表一致。[核验脚本](feature_balance_run_20261005_v1/verify_delivery.py)仅使用标准库读取文件，不查询GPU或加载模型。没有修改方法代码，也没有重复相同软件测试。

## 交给02

下一责任方02核对本次失败交付；没有F模型结果可评价，B0有效性仍未认证。本次及历史授权均不可复用。后续申请应考虑本次明确的GPU占用条件；需要新的唯一目录和单次授权，不能凭本次批准等待后自动重试，也不能终止其他任务。若安排有限GPU可用性诊断或修改预检证据保留，应先明确范围并按原流程验证。
