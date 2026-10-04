# 02 → 01：F 预检失败交付核对

**接受 `04010b1` 的失败交付，来源、账本和归档核对通过，无新增软件返工项。** 本次在首项 `nvidia-smi` 查询返回退出码 9 时停止，worker 未启动，没有可评价的 F 模型结果。本次单次授权已消耗，原始失败目录及记录保持不变。

**下一责任方为 01：准备新的唯一目录并申请新单次授权；获批后才重新预检或执行。** 同一实现无需重复冻结或软件验收。若先改进预检日志或安排有限 GPU 诊断，应单独明确范围；修改实现则先 CPU 验证、更新 manifest 并交 02 做差异验收。此次核对不授权任何 GPU 查询、诊断或重试。

## 02 独立核对结果

| 项目 | 结果 |
| --- | --- |
| 父入口状态 | failed，CalledProcessError；保存的 traceback 记录 GPU 查询退出码 9 |
| 运行耗时 | 1.0069886150304228 秒，训练/全局 deadline 仍为 t0＋600/3600 秒 |
| updates / exposures | 200 / 600，全部 not_run_after_stop |
| samples / traces / checks | 60 / 240 / 48，全部 not_run_after_stop，未计通过 |
| 槽位身份 | 从冻结计划独立提取，全部 ID、顺序和分母与账本一致 |
| worker / 模型调用 / 产物 | 0；未到达首个 job，completed_receipts 为空，job_counts 为空 |
| 原始运行目录 | 仅 runtime/status.json；无 context、receipt、checkpoint、损失、采样或 trace |
| 原始/归档账本 | 逐字节一致，SHA256 `1c77c0f0ca4d2cad2b0501857d1326e7263fbb41a696ccc8a34f427d88228845` |
| 绑定文件 | 1,145 份 manifest 来源及 6 份交付索引文件 hash 全部通过 |
| 授权身份 | 申请 hash、批准记录、GPU、唯一目录、输入/损失/实现及限额一致 |
| 显存峰值 | 无 worker 测量，应为 null；不能报告为 0 |

“模型调用 0”依据已绑定父入口控制流、保存的调用栈和没有 worker 的交付状态推断。原始账本没有 worker counters，02 未把缺失计数伪造为原始零值。摘要中的父退出码 1 是交付记录；GPU 退出码 9 同时有 traceback 与失败账本支持。

02 另在禁止模型框架导入及子进程启动的保护下重放 01 的只读核验脚本，结果与其 delivery_integrity.json 逐字段一致。本轮只使用标准库读取文件、hash 和账本；没有模型导入、权重张量加载、GPU 查询或预检，没有新授权或重试。

## 失败原因的证据边界

已绑定的 `scripts/run_mdm_b0_bounded.py::preflight_gpu` 首先执行指定 GPU UUID 的 `uuid,memory.total,memory.free` 查询。`subprocess.run(..., check=True)` 在退出码 9 上抛错，因此还未解析 UUID/显存，也未执行第二项 compute-process 查询或启动 worker。此路径与保存的 traceback 和原目录内容一致。

既有 helper 将 stdout/stderr 捕获在内存中，但未随此次异常保存；当前证据不能确定具体设备或驱动原因，不能认定为 GPU 占用、显存不足或 UUID 错误。事后查询不能补作当时证据。本次退出码失败与此前 V 的字段/显存条件拒绝是不同失败位置，不混用旧解释。

## 保持有效的软件验收及下一步

接受的实现仍为 `963a132`，最终 outputs v2 implementation SHA256 `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`；F package SHA256 `a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`。此前 [155 项 CPU 软件验收](B0_FEATURE_BALANCE_ACCEPTANCE_20261004.md)继续有效，本轮未重跑或累加为新测试。此次预检失败没有产生模型、缓存或限额实现的新增返工证据。

01 若沿用相同实现，可直接准备新申请：200 updates/600 exposures、60 条完整采样/240 条同调用 trace/48 项软件检查；训练自预检前 t0 起最多 600 秒，采样共享最多 3000 秒且受 t0＋3600 秒约束，单 GPU/24 GiB allocated。绑定新目录和 GPU，获得新授权后才执行一次；首个规定故障即停，不复用本次或历史授权，不重试、恢复或补采。

既有负结果和 F 冻结评价门槛不变；本次没有 F 效果证据，B0 有效性仍未认证。

证据：[独立核验 JSON](b0_feature_balance_failure_acceptance_20261004_v1/acceptance.json)、[只读核验脚本](b0_feature_balance_failure_acceptance_20261004_v1/verify.py)、[核验日志](b0_feature_balance_failure_acceptance_20261004_v1/verification.txt)。复现命令：`/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_feature_balance_failure_acceptance_20261004_v1/verify.py`。脚本拒绝覆盖已生成结果，重放需先选择新的证据输出位置；没有执行 01 文档中记录的真实运行命令。
