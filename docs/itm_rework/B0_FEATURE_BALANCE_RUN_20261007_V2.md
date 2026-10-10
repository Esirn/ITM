# F 新队列结果：指定 GPU 已有计算进程

**按用户“现在尝试一次新队列”执行了一个新单次队列，约 1.19 秒后在预检停止。指定 GPU 已有计算进程，worker 未启动，训练/采样均为 0。** 本次授权已消耗，没有自动重试或追加查询，也没有干预现有任务。

本记录是当前状态，接续 [单次只读诊断](B0_FEATURE_BALANCE_GPU_DIAGNOSTIC_20261007.md)。诊断时第一项检查通过并不代表完整预检通过；本次原入口确实通过身份/空闲显存检查，随后在第二项 compute-process 检查拒绝运行。当前阻碍是已有计算任务，不是要求先空出全部 24 GiB。

## 实际身份与结果

| 项目 | 结果 |
| --- | --- |
| 批准依据 | 本会话用户明确指令“现在尝试一次新队列” |
| 申请 / 实际批准 | [新申请](method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261007.json) / [批准记录](method/B0_FEATURE_BALANCE_AUTHORIZATION_20261007.json) |
| 唯一运行目录 | `outputs/itm_rework/method/feature_balance_authorized_20261007_v1` |
| 交付归档目录 | `docs/itm_rework/method/feature_balance_run_20261007_v2`（v2 是当日第二次交付，不与运行目录版本混同） |
| GPU | `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004` |
| package / implementation | 原 `a4189b89…` / 最终 outputs v2 `db323ef5…`，未修改 |
| 错误 | `RuntimeError: Selected GPU already has a compute process` |
| 失败位置 | `scripts/run_mdm_b0_bounded.py::preflight_gpu` 第 88 行 |
| 父退出码 / 耗时 | 1 / 1.191204145 秒 |
| 计划 updates/exposures/samples/traces/checks | 200/600/60/240/48，全部 `not_run_after_stop` |
| worker / 模型调用 | 无 worker；根据绑定控制流和无产物推断模型调用 0，没有原始 worker counters |
| allocated 峰值 | 未测量，null |
| 运行文件 | 仅 `runtime/status.json` |
| 账本 SHA256 | `763a7e0e97e7f9b102270ea54d1fa030e0a2cf5dd7f06630e84606945bf32651` |

保存的 traceback 和绑定源码支持：两项预检查询均正常返回，第一项身份/空闲显存门槛通过，第二项返回至少一行指定 GPU 的计算进程。原 helper 没有保存 stdout/stderr，故不提供具体 PID、所有者、用途或显存量；本次没有第三条 GPU 查询。不能用本次占用结果回溯解释较早那次 identity/free-memory 失败。

## 自检与评价

1,145 份运行绑定文件核验通过；批准和申请的目录、GPU、package/implementation、初始权重、损失、预算和停止规则一致。之前三次失败账本未变，本次原账本与归档逐字节一致。全部计划 ID/顺序及未执行状态都经只读自检，结果见 [delivery_integrity.json](method/feature_balance_run_20261007_v2/delivery_integrity.json)，可用 [verify_delivery.py](method/feature_balance_run_20261007_v2/verify_delivery.py)重放。这是主会话自检，不称独立 02 验收。

[CPU 评价目录](../../outputs/itm_rework/protocol/feature_balance_failed_evaluation_20261007_v2)保留 F/U 120 行和全部 60/72/36 对照。F 有效 0/60、历史 U 有效 60/60；48 项软件等式不可比较，600 loss 槽位缺失。head/wrists 均为证据不足，没有 F 行为标签或模型有效/无效结论。已验收实现未改，不重复全量回归、缓存构建或模型初始化。

## 下一步

需要该 GPU 现有计算任务结束后的可用时段，才有理由再次尝试。当前不再自动查询、等待重试、创建下一份待批准队列或更改“已有计算进程即拒绝”的规则；不干预现有进程。用户明确决定再次尝试时，再绑定唯一新目录和新单次授权，不能复用本次批准或目录。

目前没有活动 F 队列，也没有尚可使用的 GPU 授权。原研究目标和冻结负例保持不变；HEAD 仍为 `30d1f96`，本主会话的新增工作尚未提交，用户原有 `tasks/` 未改。
