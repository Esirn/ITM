# 01 → 02：保存状态 V 单次运行完成

用户批准 `186d9e6` 后，01 于 2026-10-03 按 `ea7f802` 已验收实现执行一次新 V 队列。**48 状态、192 分支、144 软件锚点全部完成，退出码 0；全部锚点逐位一致。** 从预检前起耗时 **30.615823 秒**，未超时、OOM、出现非有限值或重试。本次没有训练、新 checkpoint、完整动作采样或新噪声。

本交接确认执行及产物完整性；**联合控制有效性仍未认证，下一责任方为 02 独立评价**。之前预检失败的原目录和证据保持不变；本次成功不能反推上次失败的具体原因。

## 身份与授权

| 项目 | 身份 |
| --- | --- |
| implementation | `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46` |
| V package | `6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012` |
| R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |
| 实现 / 软件验收 | `0b36422` / `ea7f802`，无实现改动 |
| GPU | `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004` |
| 新运行目录 | `outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1` |

[本次授权](B0_STATE_DIAGNOSTIC_AUTHORIZATION_20261003.json)、[执行摘要](state_diagnostic_run_20261003_v1/execution_summary.json)、[命令](state_diagnostic_run_20261003_v1/commands.md)。本次单次授权已消耗，不可复用。

## 实耗与一致性

| 项目 | 实际 | 限额 |
| --- | ---: | ---: |
| 整队，预检前起 | 30.615823 秒 | 300 秒 |
| allocated 显存峰值 | 538,674,688 bytes，约 0.502 GiB | 24 GiB |
| MDM | 48 次 batch4 / 192 examples | 48 / 192 |
| IMU encoder | 72 次 / 72 examples | 72 / 72 |
| caption | 12 次 / 12 examples | 12 / 12 |
| 软件锚点 | 144/144 通过且逐位一致 | 144 |

预检记录 total 24564 MiB、free 24124 MiB、compute processes 0。锚点 P/旧 f11、T/旧 f10、S/T 各 48 项，均 max_abs=0、RMS=0，原 atol=rtol=1e−6 未改。所有调用 reserved/actual/completed 及 calls 完整一致；0 训练更新、0 denoising 调用。receipt 中通用 `branch_examples=0` 是复用 RunGuard 的非 V 计数字段；V 实际分支数以 `counts.mdm` 的 192 和 192 条 executed 分支账本为准。

## 交付与 CPU 核验

- [运行文件索引](state_diagnostic_run_20261003_v1/run_artifacts.json)：全部 688 文件及 hash；worker receipt 绑定其中 684 个产物。
- [父账本](state_diagnostic_run_20261003_v1/runtime/status.json)、[worker receipt](state_diagnostic_run_20261003_v1/runtime/V.receipt.json)、[worker 日志](state_diagnostic_run_20261003_v1/runtime/V.log)均存档；runtime 四个文件与原目录逐字节一致。
- [CPU 核验](state_diagnostic_run_20261003_v1/delivery_integrity.json)：1,656 份绑定文件未变；48 输入描述、48 保存状态/目标/mask 与冻结来源一致；192 预测形状/dtype/描述和最终状态通过；144 锚点用 NumPy 从原始数组重算，与保存结果一致。
- 288 个 NPZ 中 864 个数值数组、10,376,160 个元素全部有限；旧失败账本 hash 未变。CPU 核验没有额外模型调用或 GPU 查询。

原始 branch sidecar 按已验收实现保留 `raw_saved_pending_anchors` / `diagnostic_only=true`，并指向 `final_group_metadata`。所有 final group、receipt 及父账本均为 completed、最终分支 diagnostic_only=false；评价按最终状态链读取，不能把原始 sidecar 的中间标记误判为失败或未执行。

## 02 下一步

按冻结评价计算 192 分支的 1,728 项块指标、192 项差值及历史 Q 并列比较；保留完整分母和原阈值。重点判断同一保存状态上的 P−S/P−R，P−T/R−S 为补充；V 与历史 Q 的状态不同，不能将差异解释为因果受控分布实验。

V 没有生成新动作，旧步行/拍手/交替弓步负例保持；6 个已暴露来源、单训练 seed、同轨迹相关状态及归一化特征单位的限制仍适用。01 本轮不代替 02 进行模型有效性认证，也不追加训练或采样。
