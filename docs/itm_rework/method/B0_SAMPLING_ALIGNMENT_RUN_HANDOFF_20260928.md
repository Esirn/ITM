# 01 → 02：固定 R guidance 授权队列完成，交付独立评价

用户对 `30b3d7e` 的单次申请回复“批准”后，按 `a90d3e7` 已验收实现执行一次 Q/S1 队列。**480 探针、60 条完整采样、240 条 trace 全部完成，退出码 0，无失败或重试。** 本轮没有训练、生成新 checkpoint、重跑旧 guidance 臂或更改冻结参数。

本交接只确认执行和产物完整性；**模型有效性仍待 02 独立评价**。旧 R 步行、wrists 拍手下肢退化和训练动作生成不可靠等负例保留，不以软件一致性代替能力判断。

## 身份与复现入口

| 项目 | 绑定 |
| --- | --- |
| 冻结 package | `537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60` |
| 已验收 implementation | `793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4` |
| 固定 R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |
| 实现／独立验收 | `f9b3cb1` / `a90d3e7` |
| GPU | `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004` |
| 运行目录 | `/home/a200/0proj/ITM/outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1` |

[实际授权](B0_SAMPLING_ALIGNMENT_AUTHORIZATION_20260927.json)、[执行摘要](sampling_alignment_run_20260927_v1/execution_summary.json)、[原始状态存档](sampling_alignment_run_20260927_v1/runtime/status.json)、[执行／核验命令](sampling_alignment_run_20260927_v1/commands.md)。目录沿用授权时的 20260927 名称；本交付整理于 20260928。授权已用于这一次队列，不可复用。

## 预算实耗

| 项目 | 实耗 | 上限 |
| --- | ---: | ---: |
| Q 阶段（含加载、保存和检查） | 41.06 秒 | 600 秒 |
| 60 条 S1 共享阶段 | 2001.85 秒 | 2400 秒 |
| 整队（从预检前计时） | **2043.61 秒 / 34.06 分钟** | 3600 秒 |
| allocated 显存峰值 | **535,660,544 bytes，约 0.499 GiB** | 24 GiB |
| Q MDM | 96 次 batch5 / 480 branch examples | 480 examples |
| S1 MDM | 60000 次 batch4 / 240000 branch examples | 240000 examples |
| encoder | Q 576＋S1 120＝696 examples | 696 |
| 新训练／新 checkpoint | 0 / 0 | 0 / 0 |
| trace | 240，附着于已有采样调用 | 240，额外调用 0 |

指定 GPU 预检通过，无其他计算进程；没有换卡、回退、重试或额外模型核验调用。阶段耗时按未截断的固定 deadline 边界计算；完整公式见执行摘要。整队另含约 0.71 秒预检及绑定检查。

## 完整性与数值检查

- 480/60/240 全分母均为 completed。每条 S1 完整 1000 步、4 个预声明 trace；全部旧 random plan、输入、模型、guidance、comparison_case_id 身份通过。
- 运行前和交付时，manifest 的 **743 份文件**均保持一致。CPU 交付审计核验 **1621 份 receipt 登记产物**的 hash；扫描 **840 个 NPZ、3900 个数值数组、91,329,180 个数值元素**，全部有限；sidecar 数组描述、内嵌 metadata、trace 同次返回 hash 一致。
- Q 的 96 组 static/text 全部逐位一致。480 份原始预测及 **288 条派生状态、384 条组内对照、192 条差值之差**齐全，保留全部交叉项正负，未因有限但差的结果改变 S1。
- 240 条 trace 的原四分支组合与冻结 float64 参考全部通过；最大 error/bound 为 **0.068403**。保存 S1 实际状态上的分支与组合，不声称旧 2.5 轨迹重放。
- RIC GPU/CPU 解码检查通过，最大坐标差 **1.90735e−6 m**，沿用冻结 `atol=2e−6, rtol=1e−6`。
- 24 项完整采样检查全部通过：6 text/bypass、12 static/text、6 新旧 IMU-only；其中 **18 项逐位相同**，总体最大坐标差 **1.66893e−6 m**，小于冻结 `1e−5 m`。其余 6 项未被写成逐位相同，连续 max/RMS/count 与对照 ID 均保留。

交付审计：[脚本](sampling_alignment_run_20260927_v1/verify_delivery.py)、[结果](sampling_alignment_run_20260927_v1/delivery_integrity.json)、[日志](sampling_alignment_run_20260927_v1/logs/delivery.txt)。[完整产物索引](sampling_alignment_run_20260927_v1/artifact_index.json) 记录运行目录 **1805 份文件**的绝对路径和 hash，包含 receipt、context、日志及各类输出。未修改已验收方法代码，未重复软件回归；本次增加的是实际运行与只读 CPU 产物审计证据。

## 02 评价入口与边界

运行目录下 `Q/` 包含预测、共享输入及 `derived.json`；`samples/` 为 60 条新 S1 NPZ/JSON；`traces/` 为 240 条同次采样 trace；`runtime/` 保留完整账本及每个 worker 的 context/receipt/log。

请 02 按 [冻结包](../protocol/b0_sampling_alignment_20260927_v1/package.json) 独立评价：全部新旧 60 配对、两种 guidance 内各条件差值及差值之差、Q 分解／逐来源汇总，以及冻结 120 条新旧混排审阅。旧 R 比较对象取 `comparison_case_id`；不能将继承的 `baseline_case_id` 误作本批主对照。trace 只支持 S1 同状态的局部重组，不能解释为旧轨迹或完整反事实 rollout。

保持 3 train＋3 exposed development-val、单训练 seed、K=1 的解释边界；不新增效果阈值，不把静止／关闭条件等式当作动态 IMU 收益，不做泛化、物理／语义准入或真实设备认证。**下一责任方为 02 独立评价；01 不追加运行。**
