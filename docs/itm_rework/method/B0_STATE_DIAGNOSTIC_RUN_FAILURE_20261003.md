# 01 → 02：V 单次运行在 GPU 预检失败后停止

用户对 `25e99c1` 回复“批准”后，01 于 **2026-10-03** 执行一次已验收 V 入口。**GPU 预检失败，退出码 1；worker 未启动，没有模型输出。** 预检前起共 **1.413581 秒**，未重试、换卡或回退。目录及授权文件保留申请时的 `20260928` 名称。

实际授权：[授权记录](B0_STATE_DIAGNOSTIC_AUTHORIZATION_20260928.json)。授权已用于本次尝试，不能复用。原运行目录 `outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1` 保留，未覆盖。

## 失败证据与解释边界

父入口在 `preflight_gpu()` 的首项检查抛出 `RuntimeError: GPU identity/free-memory preflight failed`。该检查拒绝查询结果行数不为 1、UUID 不匹配或可用显存低于 4096 MiB。**已验收预检函数没有在异常前持久化原始 stdout，现有证据无法区分这三种原因**；不能断言显存不足或 GPU 被占用。后续 compute-process 查询尚未执行，未启动 worker，也没有 worker context/receipt/log 或分支预测。

[原始父日志](state_diagnostic_run_20260928_v1/logs/parent.log)、[完整失败账本副本](state_diagnostic_run_20260928_v1/runtime/status.json)、[执行摘要](state_diagnostic_run_20260928_v1/execution_summary.json)、[命令记录](state_diagnostic_run_20260928_v1/commands.md)。失败后未补查 GPU，避免把事后状态当作当时证据。

## 完整分母与预算

| 项目 | 实际结果 |
| --- | --- |
| 状态组 | 0/48 执行；48 项 `not_run_after_stop` |
| T/P/S/R 分支 | 0/192 执行；192 项 `not_run_after_stop` |
| 软件锚点 | 0/144 执行；144 项 `not_run_after_stop`，不计为通过 |
| MDM / IMU encoder / caption | reserved、actual、completed 及 calls 全为 0 |
| 训练 / 新 checkpoint / 新动作 | 全为 0 |
| 整队时间 | 1.413581 秒，限额 300 秒 |
| GPU | 仅尝试授权 UUID；无 worker 显存峰值可报告 |

软件验收 135 项通过仍是此前的 CPU 结果；本次真实 GPU 锚点未执行，模型有效性仍未认证。

## 身份与交付核验

| 项目 | SHA256 |
| --- | --- |
| implementation | `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46` |
| V package | `6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012` |
| R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |

实现 `0b36422`、独立验收 `ea7f802` 未改动。授权 GPU 为 `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。

[CPU 交付核验](state_diagnostic_run_20260928_v1/delivery_integrity.json)通过：1,656 份绑定文件 hash 一致，授权和冻结身份一致，原始/存档账本逐字节一致，完整 IDs/失败分母/零调用计数符合预检失败路径。核验脚本只读取文件，不导入模型或查询 GPU。本轮没有改实现或协议，不重复跑模型回归。

## 下一步

02 核对本次失败保留和完整分母；本批没有可评价的模型产物。后续由 01 准备新的唯一目录与单次授权申请，获批后才能重新预检或执行。若需要改进预检失败时保存原始查询的可观测性，应先完成 CPU 验证、更新 manifest 并交 02 增量验收；本轮未暗改已验收入口，也未放宽阈值。
