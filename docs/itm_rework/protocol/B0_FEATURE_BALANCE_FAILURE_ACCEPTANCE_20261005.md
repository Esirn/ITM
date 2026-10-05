# 02 → 01：F GPU 占用预检失败交付核对

**接受 `ac10cd6` 的失败交付，无新增软件返工项。** 保存的调用栈与已绑定源码一致：本次预检检测到指定 GPU 已有计算进程，在首个 worker 启动前停止。没有 F 模型结果可评价，本次单次授权已消耗。

**下一责任方为 01：考虑 GPU 可用时段，准备唯一新目录并申请新单次授权。** 相同实现不必重复冻结或软件验收；获批后才再次预检/执行，不能沿用本次批准等待后自动重试，也不能干预已有任务。若需要先做有限 GPU 可用性诊断，应明确诊断范围并另行申请；若修改预检证据保留，则先 CPU 验证、更新 manifest，再交 02 差异验收。

## 独立核对结果

| 项目 | 结果 |
| --- | --- |
| 失败类型/消息 | RuntimeError: Selected GPU already has a compute process |
| 耗时 | 0.9763586400076747 秒；训练/全局 deadline 为 t0＋600/3600 秒 |
| updates / exposures | 200 / 600，全部 not_run_after_stop |
| samples / traces / software checks | 60 / 240 / 48，全部 not_run_after_stop，检查未计通过 |
| 完整槽位 | 从冻结表独立提取，ID、顺序、分母与账本一致 |
| worker / 模型调用 / 输出 | 0；无 active job，completed_receipts 和 job_counts 为空 |
| 原目录 | 仅 runtime/status.json，无 context、receipt、checkpoint、损失、动作或 trace |
| 文件核验 | 1,145 份绑定文件和 6 份交付索引文件 hash 全部通过 |
| 原始/归档账本 | 逐字节一致，具体 hash 见下方核验 JSON 的 sources |
| 授权身份 | 申请 hash、批准记录、实现/package、输入/损失、GPU、目录和限额一致 |
| worker 显存峰值 | 无测量，应为 null，不能报告为 0 |

模型调用 0 是根据绑定控制流、调用栈和无 worker 产物推断；原始账本没有 worker counters，02 未伪造零值。父退出码 1 由执行摘要记录。02 重放 01 的只读核验脚本，与 delivery_integrity.json 逐字段一致；此外独立检查了授权申请 hash、截止时间及旧失败账本。

## 本次直接原因及证据边界

已绑定 `scripts/run_mdm_b0_bounded.py::preflight_gpu` 的第一项指定 UUID 查询正常返回，并通过单行 UUID 及 free MiB≥4096 检查；第二项 compute-process 查询正常返回，至少一行匹配指定 UUID，随后在第 88 行抛出保存的异常。上述判断来自**已绑定源码和保存的 traceback**，不是本轮重新查询 GPU 所得。

原始 stdout/stderr 没有持久化，不能提供具体 PID、所有者、实际显存或用途，也不能断言该进程目前仍在运行。02 没有查询或干预该进程。

10 月 4 日退出码 9 的原始账本与归档仍逐字节一致，SHA256 保持 `1c77c0f0ca4d2cad2b0501857d1326e7263fbb41a696ccc8a34f427d88228845`。本次占用拒绝发生在不同位置，不能回溯解释旧查询错误；两次失败与授权消耗分别保留。

## 不变的验收与后续范围

实现仍为 `963a132`，最终 outputs v2 implementation SHA256 `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`；package SHA256 `a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`。此前 [155 项 CPU 软件验收](B0_FEATURE_BALANCE_ACCEPTANCE_20261004.md)继续有效，本轮未重复运行或累加为新测试。预检按已有占用限制停止，没有新增模型、缓存或限额实现返工证据。

若沿用同一方案，新申请仍为 200 updates/600 exposures、60 条完整采样/240 条同调用 trace/48 项软件检查；训练自预检前 t0 起最多 600 秒，采样共享最多 3000 秒并受 t0＋3600 秒约束，单 GPU/24 GiB allocated。绑定唯一新目录、指定 GPU 和新批准依据；首个规定故障即停，不重试、续训、换卡或补采，不复用任何已消耗授权。

02 本轮仅标准库文件/hash/账本核验，保护性禁止模型框架导入和子进程启动；没有加载权重张量、模型前向、训练、采样、GPU 查询、预检或新授权。F 冻结评价门槛及既有负结果不变，模型有效性仍未认证。

证据：[独立核验 JSON](b0_feature_balance_failure_acceptance_20261005_v1/acceptance.json)、[只读脚本](b0_feature_balance_failure_acceptance_20261005_v1/verify.py)、[核验日志](b0_feature_balance_failure_acceptance_20261005_v1/verification.txt)。实际命令为 `/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_feature_balance_failure_acceptance_20261005_v1/verify.py`。脚本拒绝覆盖已生成结果，重放需另选输出位置；本轮未执行交接中保存的真实运行命令。
