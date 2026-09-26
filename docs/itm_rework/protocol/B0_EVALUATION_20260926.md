# 02 → 01：授权 B0 的独立结果评价（2026-09-26）

读取方法交付 **`4ae11d0`**，按生成前已冻结的 [metrics.json](canonical_b0_20260925_v3/metrics.json) 评价固定全部 36 槽位。**运行/产物完整性通过；本次结果尚不支持“有效联合控制”或“IMU 带来实例信息收益”。** 没有新增通过阈值，没有把位置误差降低当作语义成功。当前仍处于 G2/B0 能力诊断，不能据此进入已验证 B/C 能力或研究结果认证。

本轮仅 CPU 读取、数值重放、作图与审阅；无训练、模型前向、扩散采样或 GPU 预检。01 的一次预算授权已经执行完毕，剩余时间不能自动用于追加试验。本报告所在提交可由 `git log -1 -- docs/itm_rework/protocol/B0_EVALUATION_20260926.md` 定位。

## 完整性和独立核对

- **36 planned / 36 valid / 0 failed**，失败分母固定 36。有效性是文件/数值/身份有效，不是动作成功。原 200 updates、36,000 去噪调用、144,000 分支样本前向记录相符；无输出过滤或替换。
- 274 个文件 hash 核对通过，包括原依赖、训练产物、逐槽位 NPZ/JSON 与 receipt。100 个 epoch checkpoint 的交付 hash 保留；全部验证只绑定最终 checkpoint，不做 epoch 选择。
- 最终 checkpoint SHA256：`44d2c3307f27bca2c79ab7d367890f968f78777168c3c37a306a7bf0560d6ac2`；来源 SHA256：`9f39bf7ffd93b2595dfdcea4ee3288761b14a6e75a6585fedb841fd490295ef1`。来源结构与已验收报告相等；重新由 3 个 train 片段的 **398 个 20 Hz 帧**核对 train-only 统计。
- 逐输出核对精确长度、20 Hz 时间、完整 mask、head/wrists sensor mask、直接 GT、caption 索引和文本/IMU presence。实际保存的 acceleration/orientation/normalized_imu 与 02 独立插值参考一致；physical-static 和 numeric-zero 按原定义核对，IMU-only 的审计 caption 不等于文本条件存在。
- 6 组各 1,000 次 PCG64 噪声，共 **6,000 个唯一 draw** 独立重建 hash；每组 6 条件的完整随机计划相等。再次 CPU 重放得到逐字节相同的评价 JSON，不是新增模型试验。
- 本地独立 RIC 解码与保存坐标的最大绝对差 **2.8610e-6 m**。使用 `rtol=1e-6, atol=2e-6` 的算术重放容差，并报告连续误差；该容差不是物理/语义成功阈值。
- **6/6 组 text-only 与原 backbone bypass 的坐标逐值相等，最大差 0**。这证实本批输出的 strict-null 恢复，不证明 paired 的控制质量。
- receipt 记录的总时长 **561.131 秒**、峰值 allocated **737,469,952 bytes（0.687 GiB）**及同一 GPU UUID 符合原预算。时间/显存来自持久化运行记录，本轮未重新观测 GPU，也不把 allocated 当作显卡总占用。

完整机器结果：[evaluation.json](b0_evaluation_20260926_v1/evaluation.json)、[最终代码/文件核对](b0_evaluation_20260926_v1/artifact_audit_replay.json)、[重放核对](b0_evaluation_20260926_v1/replay_verification.json)。首次审核的代码 hash 单独保存在 `artifact_audit.json`，最终脚本身份以 `artifact_audit_replay.json` 为准。

## 冻结口径的定量结果

所有误差使用交付坐标、完整有效序列与直接 20 Hz GT，**没有 yaw/平移拟合、Procrustes、时移或时间拉伸**。每行对 3 个来源组等权平均，均为 valid=3 / planned=3；不是按帧数加权。位置单位 m，速度单位 m/s。head/wrists 指标仅为关节 15 或 20/21 的位置代理，不能称为生成 IMU 残差。

| 配置 | 条件 | Global MPJPE | Root-relative MPJPE | Root RMSE | Head/wrists proxy | Velocity RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| head | paired | 0.5174 | 0.1939 | 0.5413 | 0.4777 | 0.6854 |
| head | text_only | 0.7546 | 0.1284 | 0.7707 | 0.7160 | 0.8753 |
| head | imu_only | 0.5894 | 0.2222 | 0.6328 | 0.5435 | 0.8687 |
| head | backbone_bypass | 0.7546 | 0.1284 | 0.7707 | 0.7160 | 0.8753 |
| head | numeric_zero | 0.5210 | 0.1936 | 0.5504 | 0.4727 | 0.6713 |
| head | physical_static | 0.5177 | 0.1950 | 0.5418 | 0.4790 | 0.6832 |
| wrists | paired | 0.5115 | 0.1817 | 0.5483 | 0.6368 | 0.5598 |
| wrists | text_only | 0.8072 | 0.1264 | 0.8086 | 0.9388 | 0.7186 |
| wrists | imu_only | 0.9808 | 0.2191 | 1.0949 | 1.0703 | 0.9932 |
| wrists | backbone_bypass | 0.8072 | 0.1264 | 0.8086 | 0.9388 | 0.7186 |
| wrists | numeric_zero | 0.5331 | 0.1928 | 0.5786 | 0.6801 | 0.5669 |
| wrists | physical_static | 0.5127 | 0.1820 | 0.5425 | 0.6364 | 0.5526 |

绝对 jerk 和接触速度另列，防止把更平滑/更静止误判为更好。jerk 是三阶前向差分 ×20³ 的平均欧氏模，单位 m/s³，3 个来源等权；GT 均值 **30.4866**。接触速度仅在 `000433` / `000382` 交付 GT 的 4 个接触标志上计算，2 个来源等权；GT 均值 **0.0456 m/s**。四脚关节顺序 `[7,10,8,11]`；只取生成序列存在的 T−1 个差分，GT 末帧的外部 endpoint 不进入评价。

| 配置 | 条件 | 绝对 jerk（n=3） | GT-contact speed proxy（n=2） |
| --- | --- | ---: | ---: |
| head | paired | 30.2822 | 0.1549 |
| head | text_only | 45.3191 | 0.1366 |
| head | imu_only | 36.1422 | 0.2821 |
| head | backbone_bypass | 45.3191 | 0.1366 |
| head | numeric_zero | 24.2699 | 0.1639 |
| head | physical_static | 27.2959 | 0.1513 |
| wrists | paired | 9.6433 | 0.0621 |
| wrists | text_only | 22.5850 | 0.0409 |
| wrists | imu_only | 58.4325 | 0.7201 |
| wrists | backbone_bypass | 22.5850 | 0.0409 |
| wrists | numeric_zero | 8.3684 | 0.0795 |
| wrists | physical_static | 6.6710 | 0.0531 |

`000139` 的 GT 有非平坦支撑/根高度上升，**flat-floor/contact speed 不适用**，逐条值为 null，未当作零或平地失败。各轴 min/max/range、pelvis path length、GT jerk、接触 stencil 数和所有逐条数值在 [per_case.csv](b0_evaluation_20260926_v1/per_case.csv) 与 `evaluation.json` 中；GT 接触 stencil 分别为 120（000433）、372（000382）。没有按生成高度另选“好脚帧”。

## 对照说明：误差下降不能支撑真实输入收益

paired 相对 text-only 的全局 MPJPE 降低 **0.2372 m（head）/0.2956 m（wrists）**，但 root-relative MPJPE 上升 **0.0655/0.0553 m**。完整动作语义也出现负例，因此不能只取全局误差下降宣称联合能力。

同一来源、配置与噪声下，paired 对各控制输入的描述性差异如下。Δ 为 paired 减对照，位置误差负值表示数值较低；输出间距为两条输出自身的平均关节欧氏距离，不是对 GT 的误差，也不是成功率。

| 配置 | 对照 | Δ global MPJPE (m) | Δ root-relative MPJPE (m) | 输出间距 (m) |
| --- | --- | ---: | ---: | ---: |
| head | text_only | −0.237169 | +0.065507 | 1.086624 |
| head | numeric_zero | −0.003570 | +0.000310 | 0.034751 |
| head | physical_static | −0.000256 | −0.001038 | 0.005114 |
| wrists | text_only | −0.295625 | +0.055331 | 1.100421 |
| wrists | numeric_zero | −0.021554 | −0.011108 | 0.070920 |
| wrists | physical_static | −0.001164 | −0.000241 | 0.029875 |

全部 **24 个逐来源 paired 对照差值**保存在 [paired_comparisons.json](b0_evaluation_20260926_v1/paired_comparisons.json)。paired 与 static 并非严格相等，不能说“模型完全不读输入”；但平均结果非常接近、速度/局部位置等指标没有一致收益，当前不能排除“开启条件分支带来固定动作偏置”的解释。numeric-zero 是输入张量零；physical-static 是零加速度、首帧姿态保持后用真实 train 统计归一化，两者没有混用。

最清楚的反例是 **000139**：GT 的 pelvis 路径长 **3.005 m**；text-only head/wrists 路径 **7.580/6.809 m**，仍能看到步态但轨迹不符。paired 路径缩为 **0.804/0.787 m**，且变为持续高抬单腿，global MPJPE 虽降为 **1.217/1.191 m**，root-relative MPJPE 却升至 **0.294/0.286 m**。static 和 numeric-zero 同样出现高抬腿。位置误差下降伴随移动缩减和动作类别偏离，不能作控制成功。

## 全部输出的条件隐藏审阅

在计算结果前固定 PCG64 seed **20260926** 对原 36 槽位排列；审阅时只显示 R01–R36 和目标粗动作，隐藏 condition/config。按六页顺序查看全部输出：每条 10 个等间距姿态快照，加逐帧手间距、腕高、踝高、根高和根 X/Z 曲线。姿态图仅为显示做逐帧 root-XZ 居中，数值评价没有居中。先落盘逐条观察，再读取 key 合并。

- [条件隐藏原始笔记](b0_evaluation_20260926_v1/semantic_notes_condition_hidden.json)：36/36 有记录。
- [揭示条件后的逐条对应](b0_evaluation_20260926_v1/semantic_review_joined.json)、[固定顺序 key](b0_evaluation_20260926_v1/review_key.json)、[图像 hash/路径](b0_evaluation_20260926_v1/review_rendering.json)。所有输出保留，没有只挑 paired 或好例。
- 图页：[1](/home/a200/0proj/ITM/outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review/page_1.png)、[2](/home/a200/0proj/ITM/outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review/page_2.png)、[3](/home/a200/0proj/ITM/outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review/page_3.png)、[4](/home/a200/0proj/ITM/outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review/page_4.png)、[5](/home/a200/0proj/ITM/outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review/page_5.png)、[6](/home/a200/0proj/ITM/outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review/page_6.png)。

具体 paired 负例：`000139-head/wrists`（R16/R14）为长时间高抬腿，无目标步行；`000433-head`（R31）抬腿后末尾仍举着，未完整站回；`000433-wrists`（R36）仅小幅抬落/转体，明确踢出不确定；`000382-head`（R13）末段张臂/抬脚而非双臂放下，`000382-wrists`（R22）仅一次宽泛靠近后降低，反复开合不明显。IMU-only 的拍手样本也出现跑步转向（wrists R11）或跳起样动作（head R25）。

这些是单一 AI 审阅者的粗骨架观察，**不是独立人工双盲实验、逐帧动画自然度认证或多评审共识**。快照对短暂动作有采样限制，手部接触没有认证；`partial_or_uncertain`/`target_not_observed` 仅为描述标签，不构造语义成功阈值或 JSR。静态快照不确定的项目保留 uncertain，未通过调阈值转成成功。

## 限制、交接和下一责任方

只有 **3 个开发验证来源、1 个训练 seed、K=1**。不作显著性/总体 CI、条件多样性、总体 FID/R-precision、同类打乱实例优势或泛化声明。骨架来源和预先统一坐标仅适用于限定离线合成基准；backbone 预训练暴露仍可能存在。严格生成 IMU 闭环、真实设备标定、相容编辑 witness 均未认证；witness 是后续 B 分支依赖，不是本次 B0 评价未完成的理由。

**02 本次独立评价已完成。下一步交给 01 做具体 B0 能力诊断和修正提案，不再重复接口验收或新增防护检查。** 建议围绕已暴露的两个问题提出可检验解释：

1. 在 `000139` 中，为什么开启任一 IMU-present 条件后，文本步行被相似高抬腿模式取代？检查条件残差对文本/骨干路径的影响、训练样本覆盖及 200-update 小样本训练的行为，区分实现问题与欠训练/过拟合。当前数据只能支持这些是待查假说，不能定性为其中某一原因。
2. 为什么 paired 与 physical-static 几乎同形，而关闭 IMU 的 text-only 与 raw backbone 完全一致？提案应明确如何检验真实时变输入的增量，而不把“开启分支就改变输出”当作成功。

01 可先读取已有 loss、权重和本报告定位，不自动追加训练、采样或 checkpoint/guidance 搜索。若需要计算试验，应交付一个范围明确的 B0 修正/诊断方案，写清数据/模型改动、训练与采样数、对照和预算；02 再冻结对应评价，真实执行取得新的相应授权。旧 36 输出、当前负结果和原冻结评价定义保持不变；不能用静止对照补充训练后仍称同一次 B0。03 目前只可引用“限定 B0 执行和独立评价完成，同时发现能力负例”，不能引用联合控制有效。

## 复现和验证

新增评价脚本 `scripts/evaluate_canonical_b0.py` 与指标模块 `src/itm/metrics/canonical_b0.py`；6 项解析测试通过，覆盖恒等、平移不拟合、三阶导数单位、欧氏速度 RMSE、GT-contact 顺序/末帧排除和坏输入拒绝。此次方法代码未改，不重复 166 项方法回归。最终脚本再次 CPU 重放，评价 JSON 和审阅排列均逐字节相同。

[执行命令](b0_evaluation_20260926_v1/commands.md) 使用独立输出目录；运行 `audit` 时拒绝覆盖已有 `evaluation.json`。旧契约/数据包、01 方法与采样代码、01 报告以及旧诊断资产均未改写。评价证据追加到公共 `evidence_index.jsonl`。
