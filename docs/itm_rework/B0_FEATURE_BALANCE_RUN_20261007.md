# F 单次运行结果与当前状态（2026-10-07）

**本次获批队列已执行一次，在预检阶段失败，worker 未启动；本次授权已消耗。** 没有真实 F 模型结果，head/wrists 的效果判断仍为证据不足。未自动重试、追加 GPU 查询或干预其他进程。

本记录接续 [CPU 评价准备](F_EVALUATION_PREPARATION_20261007.md)。原准备报告中的“尚无授权/目录”是当时事实；当前以本次批准记录、实际失败账本和本文件为准。合并主会话完成执行和自检，没有恢复 01→02→01 传话，也不称新的独立 02 验收。

## 执行身份

- 用户在本会话批准上一轮明确列出的 [20261005_V2 单次申请](method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005_V2.json)，已记录于 [实际批准文件](method/B0_FEATURE_BALANCE_AUTHORIZATION_20261005_V2.json)。请求原文件仍是历史申请，`authorized=false` 不覆盖后来明确批准的记录；批准文件保留原授权事实，不改写为可重用授权。
- 唯一目录：`outputs/itm_rework/method/feature_balance_authorized_20261005_v2`。
- GPU：`GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
- 原 package `a4189b89…` / 最终 outputs v2 implementation `db323ef5…` 未变化；执行前后均核对 1,145 份绑定文件。HEAD 仍为 `30d1f96`，CPU 评价代码和相关新文档尚未提交，用户原有 `tasks/` 保留。
- 只调用一次已验收 bounded 入口；24 GiB / 600 秒训练 / 3,000 秒共享采样 / 3,600 秒总限额和首故障停止规则未变。

## 失败事实与边界

| 项目 | 实际结果 |
| --- | --- |
| 父退出码 / 耗时 | 1 / 1.095670270 秒（预检前 t0 至失败） |
| 错误 | `RuntimeError: GPU identity/free-memory preflight failed` |
| 失败位置 | `scripts/run_mdm_b0_bounded.py::preflight_gpu` 第 84 行 |
| 指定 UUID 查询 | 根据绑定控制流和 traceback，第一条查询正常返回，随后组合条件不通过 |
| 第二条 compute-process 查询 | 未到达 |
| worker / 模型调用 | 未启动；模型调用 0 为控制流、无 active job/receipt/产物支持的推断，没有伪造 worker counters |
| 训练 / 呈现 / 采样 / trace / 软件检查 | 200 / 600 / 60 / 240 / 48，全部 `not_run_after_stop` |
| 实际产物 | 运行目录仅 `runtime/status.json`，无 checkpoint、动作、trace 或 exposure |
| allocated 峰值 | 未测量，`null`，不是 0 |
| 账本 SHA256 | `3ab1488ee9207558424645c031b28cee8db4442d3a947d807f20c09b6e7571fe` |

失败条件是三项的逻辑或：返回行数不是 1、返回 UUID 不符、空闲显存小于 4096 MiB。**由于冻结 helper 用 capture_output 但未把原始 stdout/stderr 持久化，不能唯一确定哪项失败，不能直接宣称显存不足或 GPU 被占用。** 本次没有得到完整预检返回值，也没有查询 PID/所有者。10/04 退出码 9 和 10/05 compute-process 占用是不同失败，各自原账本 hash 保持不变。

## 已完成的后处理

原始账本已逐字节归档，完整父 traceback 已保留。文件/hash 自检验证批准与申请身份、完整计划 ID/顺序、所有未执行状态、无 worker/额外文件，以及旧失败保持不变。[自检结果](method/feature_balance_run_20261007_v1/delivery_integrity.json)与[可重放脚本](method/feature_balance_run_20261007_v1/verify_delivery.py)可追溯。

新 CPU 评价入口已读取本次实际失败账本，输出到 [本次评价目录](../../outputs/itm_rework/protocol/feature_balance_failed_evaluation_20261007_v1)。保留 F/U 120 行及全部 60/72/36 对照；F 0/60、历史 U 60/60。48 个等式槽位全部不可比较，不计通过；600 loss 槽位缺失；两个主配置均 inconclusive。没有为缺失 F 填写行为标签，也没有重复生成 50 MB 的准备审阅页面。此次后处理没有 GPU 查询或真实模型调用。

未改已验收运行实现，无新增全量测试或模型效果证据。软件验收通过、符合停止规则和失败交付完整均不能说明 F 有效或无效。

## 唯一下一步

先做一次范围明确的只读诊断，取得指定 UUID 的原始查询输出，避免再次消耗完整队列授权而仍只有通用错误。已准备 [单查询申请](method/B0_FEATURE_BALANCE_GPU_DIAGNOSTIC_REQUEST_20261007.json)，**尚未批准/执行**：

```text
nvidia-smi --id=GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 --query-gpu=uuid,memory.total,memory.free --format=csv,noheader,nounits
```

仅 1 条查询、超时 10 秒，保存 argv、时间、退出码/错误、stdout/stderr 和三项预检条件的分别解析；不查询计算进程，不轮询/重试，不加载模型/训练/采样。输出拟用 `outputs/itm_rework/method/feature_balance_gpu_diagnostic_20261007_v1`。诊断通过也不自动启动训练。

需要用户另行明确批准这项诊断：刚才获批范围是一次原队列，约定首次故障即停、无追加查询；该次授权已经消耗。后续“continue”按继续失败归档/CPU 后处理执行，不当作未展示诊断申请的批准。若未来重跑 F，须绑定另一个新目录和新单次授权，不能复用本次目录。
