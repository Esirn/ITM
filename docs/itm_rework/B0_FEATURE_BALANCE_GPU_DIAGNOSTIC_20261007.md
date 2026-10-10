# GPU 单次只读诊断结果与当前状态

**指定 GPU 在本次查询时通过第一项身份/空闲显存检查：UUID 正确、返回 1 行、空闲 7,488 MiB。** 没有查询计算进程，不能据此确认完整运行条件，也不能解释较早的复合预检失败。没有启动训练或采样。

用户批准 [单查询申请](method/B0_FEATURE_BALANCE_GPU_DIAGNOSTIC_REQUEST_20261007.json)后，已保存 [批准记录](method/B0_FEATURE_BALANCE_GPU_DIAGNOSTIC_AUTHORIZATION_20261007.json)，执行一次指定命令，超时上限 10 秒，无重试或追加查询。

| 项目 | 观测 |
| --- | --- |
| 查询时间 | 2026-10-07 18:48:14（Asia/Shanghai，UTC+8） |
| 退出码 / 查询耗时 | 0 / 0.04256 秒 |
| 返回行数 / UUID | 1 / `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`，精确匹配 |
| 总显存 | 24,564 MiB |
| 空闲显存 | 7,488 MiB（7.3125 GiB） |
| 原第一项门槛 | 单行、UUID 匹配、free≥4096 MiB，三项均通过 |
| stderr | 空，原始零字节文件已保存 |
| 计算进程 / 完整 GPU 可用性 | 未查询 / 未认证 |
| 模型调用 / 训练 / 采样 | 全部 0 |

原始 stdout：

```text
GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004, 24564, 7488
```

该观测比 [F 失败运行](B0_FEATURE_BALANCE_RUN_20261007.md)晚。较早 helper 没有保存原始 stdout，因此不能由本次结果回溯断言较早一定是显存不足、UUID 不符或多行返回；也不能由剩余显存推断具体 PID、所有者或任务。完整队列还会检查指定 GPU 无计算进程，该检查本次不在授权范围内。

精确 argv、开始/结束 UTC 时间、退出码、原始 stdout/stderr、分别解析的预检条件、请求/批准/执行脚本 hash 都保存在 [诊断输出](../../outputs/itm_rework/method/feature_balance_gpu_diagnostic_20261007_v1/status.json)，逐字节副本及[文件核验结果](method/feature_balance_gpu_diagnostic_20261007_v1/verification.json)位于 docs 下同名证据目录。本轮自检未再调用 GPU；较早失败账本 hash 未变。单查询授权已消耗，不可重用执行脚本。

## 下一步

已准备 [新的 F 单次队列申请](method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261007.json)，`authorized=false`。新目录为 `outputs/itm_rework/method/feature_balance_authorized_20261007_v1`，实际批准记录尚不存在。沿用原 package `a4189b89…`、implementation `db323ef5…` 和指定 GPU；不更改预检、损失、冻结评价、噪声或负例。

**待用户选择预计至少一小时的空闲窗口，并明确批准新申请后，才执行一次原完整队列。** 范围仍为 200 updates/600 exposures、60 条 DDPM1000 采样、240 同调用 trace/48 软件检查；单 GPU allocated≤24 GiB，训练≤600 秒、共享采样≤3000 秒、总计≤3600 秒，首次规定故障即停，不轮询/重试/换卡/补采或干预其他进程。本次诊断通过不能绕过完整预检，不能复用已消耗的训练授权。

当前仍无真实 F，主配置效果均为证据不足；不自动增加模型预算或改变研究方案。以上是合并主会话的文件自检，不称独立验收或模型效果证据。
