# F 评价准备：实验主会话当前状态

2026-10-07（Asia/Shanghai）。已按 `NEW_SESSION_PROMPT.md` 接管原 01＋02 的日常职责，并完成无需 GPU 授权的评价准备。**真实 F 结果仍为 0；head、wrists 均为证据不足，不能据此判断固定配重有效或无效。** 本文是主会话自检/重放记录，不是新的独立 02 验收；不改论文主张。

## 当前身份与授权

- 工作树基准 HEAD：`30d1f96`，分支 `method/itm-rework`。相比迁移前 `971d101` 仅增加交接资料。用户原有未跟踪目录 `docs/itm_rework/tasks/` 原样保留。新增评价代码和本文尚未提交。
- package SHA：`a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`。
- 最终 outputs v2 manifest SHA：`db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`；与 docs 中 v1 归档字节一致。1,145 份运行绑定文件、共同 initial、历史 R final200、历史 unit-guidance U 输出及两次旧失败身份核对通过。
- 10/04 查询退出码 9 的具体原因仍未知；10/05 为指定 GPU 已有计算进程而停止。两次 worker 均未启动，授权均已消耗。没有用后一次占用解释前一次错误。
- [最新申请](method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005_V2.json)未改，SHA `7584d8bdbb6d4c23cbb597925b326fd812ab03753caac0632ffbdbf75d6b3043`，`authorized=false`。拟运行目录和实际批准记录仍不存在。
- 本轮 GPU 查询/预检、真实模型前向、训练、采样、checkpoint 张量加载均为 0。没有缓存重建、模型初始化、随机计划重建、旧队列重试或其他进程干预。沿用既有 155 项软件验收，未重复全量回归。

## 已交付

[CPU 入口](../../scripts/evaluate_b0_feature_balance.py)与[纯数组计算模块](../../src/itm/metrics/feature_balance.py)独立于已绑定运行入口，未修改冻结协议或运行依赖，无需重建 manifest。

- 读取已保存 F 数组和运行账本；默认允许 F 尚不存在。核对 package/implementation、final200、初始化、loss、计划、输入数组、噪声元数据、receipt/产物 hash、计数和截止时间。错误/缺失/诊断输出保留分母并排除支持判定。不会执行、预检或创建运行目录。
- 复用 `canonical_b0.evaluate_motion`，保持 20 Hz、原坐标、无空间/时间拟合。保留原标量指标、GT 对照、各轴范围/根路径、完整 normalized263 的 uniform/D67/A196/balanced 与旧 5＋RIC3 分块。接触项仍只准入 val000433/000382，train 与 val000139 为 NA。
- 固定 120 行、60 个 F−U 配对、72 个臂内对照、36 个差值之差；按 arm/split/config/condition 等来源权聚合，报告计划数、有效数、缺失理由和均值/min/max。48 项软件等式不包含 F/U IMU-only。600 条训练损失槽位保留，保存四损失视图及每样本除 3 的检查；不把动态训练曲线解释为受控收敛证据。
- 主 000506 head/wrists 分别使用冻结五项门槛、原浮点松弛及 root-relative 严格改善；同时需要 F complete、U incomplete 的行为证据和自身 text/static 增量收益。其余 10 个 paired source/config 逐项列退化与行为损失，未增加成功率或调整阈值。
- 固定顺序的 120 个审阅页面隐藏 arm/config/condition，提供十个均匀姿态、全部帧播放/滑块、XY/ZY 视图与全帧 pelvis/踝/趾/膝角曲线；退化骨长记 NA。审阅文件须先以 review_id 保存全部 120 行分类、侧别/顺序/回收证据、理由及行为损失，才生成解码表。当前模板未填写，也未伪造真实 F 或新的 U 行为审阅。

详细准备产物位于 [输出目录](../../outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1)：[120 行指标](../../outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1/samples.json)、[全部对照](../../outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1/contrasts.json)、[判据结果](../../outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1/decision.json)、[审阅入口](../../outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1/review/index.html)、sources 与 artifact_index。约 50 MB，包含完整序列；不需要模型服务或网络。

## 自检/重放结果

| 项目 | 结果 |
| --- | --- |
| 定向 CPU 测试 | **36 passed**：30 项新评价测试＋6 项冻结基础指标测试，0 failed/skipped |
| 历史 U 重算 | 60/60 有效，全部动作指标（含 extent/GT/NA）与原 U 报告逐字段完全一致 |
| F 人口 | 0/60 有效，60 missing；不是 60 个模型行为失败 |
| 对照 | 60 跨臂、72 臂内、36 差值之差全部保留；只有 36 项 U 臂内对照可比较 |
| 软件等式/损失 | 48 项均 not_comparable，不计通过；600 个 F loss 槽位均 missing |
| 来源/输出 | 1,159 份来源 hash、131 份输出索引 hash 验证通过 |
| 页面程序检查 | 120 页 JS 初始绘图、末帧、ZY 切换及膝角曲线运行通过；这不是浏览器视觉检查或行为审阅 |

证据：[测试日志](feature_balance_evaluation_preparation_20261007_v1/tests.txt)、[重放日志](feature_balance_evaluation_preparation_20261007_v1/replay.txt)、[只读核验结果](feature_balance_evaluation_preparation_20261007_v1/verification.json)、[复现命令](feature_balance_evaluation_preparation_20261007_v1/commands.md)。测试内的成功 F 账本、F 解析 fixture、数值改善及行为标签全部明确标记为合成，仅存在临时测试目录，不进入上述真实评价人口。

开发时首次只读 hash 命令使用了当前 Python 不提供的 `hashlib.file_digest`，在比较前报 AttributeError；已换成分块 SHA256 后完整通过，没有写入/改变实验产物。评价测试与最终 CPU 重放无失败。

## 尚缺与唯一下一步

尚缺真实 F 队列、实际运行完整性复核及全量行为审阅；本入口的成功运行解析目前只经过合成账本验证，真实产物还须按实际内容核对。trace 保留原 240 槽位并校验 receipt 绑定产物，未新增 Q/V 或跨轨迹因果结论。已有 U 已被看过，未来单会话审阅也不称完全盲评或独立人工验证。

**唯一下一步：用户在预计至少一小时空闲窗口的开始，明确批准上述待批准申请，再执行一次原入口队列。** 本轮用户明确限定不授权 GPU 查询/预检或真实运行，因此准备工作完成不触发执行。

具体范围保持原申请：

- GPU：`GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`，当前是否空闲未知。
- 唯一目录：`outputs/itm_rework/method/feature_balance_authorized_20261005_v2`；批准后才记录 `method/B0_FEATURE_BALANCE_AUTHORIZATION_20261005_V2.json`。
- 单 GPU / allocated≤24 GiB；200 updates / 600 exposures；仅 F final200 采样 60 条 DDPM1000 完整动作，240 同调用 trace / 48 软件检查，额外 trace 调用为 0。
- t0 在预检前；训练≤600 秒，采样共享≤3,000 秒，总计≤3,600 秒；MDM≤60,600 次/240,600 examples，encoder≤1,320 examples，caption≤660 次。
- 首个规定故障即停，保存完整分母与有限产物；不自动等待、轮询、重试、续训、换卡、补采或干预已有任务。用户选择时段不能越过原预检限制。

获批后由本主会话继续执行、核验、全部 F/U 评价和研究决策，不再需要 01→02→01 传话。若 F 不足，不自动搜索其他权重或追加训练。
