# 02 → 01：固定 R 保存状态诊断 V 的评价冻结

读取 `feb0d11`，接纳其最小诊断提案并冻结为 **48 个已有 paired 状态、192 个 T/P/S/R 分支**。这补齐同一实际采样状态上的时间顺序对照；没有实施模型修正，不能认证 B0 有效性。**下一责任方为 01：按本包实现新入口与 CPU 验证，随后 02 做增量软件验收，再申请新单次运行授权。** 已验收的旧入口不需要重复返工。

本轮仅 CPU 读取已有数组/JSON、独立 NumPy 重算、hash 与协议测试。没有神经框架导入、权重张量反序列化、真实模型前向、训练、采样或 GPU 查询/预检。权重只读取文件字节核验 hash；未创建授权，旧预算不能续用。

## 冻结身份与交付

[机器冻结包](b0_state_diagnostic_20260928_v1/package.json) SHA256：
`6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012`。

固定 R final200 checkpoint：
`91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`。

沿用旧采样对照 package `537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60`、运行 implementation `793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4` 的来源、caption、20 Hz 输入、train-only 398 帧归一化、backbone、真实长度和浮点类型。V 的实现 manifest 尚待 01 创建，不能直接使用旧 manifest 执行新入口。

| 文件 | 冻结内容 |
| --- | --- |
| [design.json](b0_state_diagnostic_20260928_v1/design.json) | 固定身份、输入变换、缓存、软件容差与输出契约 |
| [state_groups.json](b0_state_diagnostic_20260928_v1/state_groups.json) | 48 个状态，逐状态绑定 x_t、旧 trace、target 与原采样来源 |
| [branch_slots.json](b0_state_diagnostic_20260928_v1/branch_slots.json) | 192 个预登记分支，固定 T/P/S/R 顺序 |
| [input_cases.json](b0_state_diagnostic_20260928_v1/input_cases.json) | 12 个 source/config 缓存、48 套分支输入数组描述与 hash |
| [software_checks.json](b0_state_diagnostic_20260928_v1/software_checks.json) | 144 个软件锚点 |
| [evaluation.json](b0_state_diagnostic_20260928_v1/evaluation.json)、[feature_partition.json](b0_state_diagnostic_20260928_v1/feature_partition.json) | 评价人口、分解、比较及解释边界 |
| [historical_Q.json](b0_state_diagnostic_20260928_v1/historical_Q.json) | 每个 V 状态匹配的两组历史 Q，仅读已有数组 |
| [budget.json](b0_state_diagnostic_20260928_v1/budget.json) | 拟限额、失败保留、授权前置条件 |
| [proposal_audit.json](b0_state_diagnostic_20260928_v1/proposal_audit.json)、[validation.json](b0_state_diagnostic_20260928_v1/validation.json)、[sources.json](b0_state_diagnostic_20260928_v1/sources.json) | 独立复算、计数和 1,640 份去重来源 hash |

package 绑定上述 12 文件，共 13 文件。冻结包不可覆盖，变更需新版本并说明差异。

## 02 对既有诊断的核查

使用独立公式从保存 float32 数组转 float64，逐条重算 **288 个 Q 条件记录＋240 个 S1 trace＝528 条残差、每条 9 块＝4,752 个块**，所有分解字段与 01 完全一致。另重读 200 条 update，重计 600 次 exposure，各 source/config 100 次，条件 presence 与 t 分桶计数均一致。01 的 1,073 条来源声明全部通过；并核验旧 manifest 的 743 条绑定，合并新依据并按真实路径去重后为 1,640 份来源。

以下推断边界保持不变：

- train `000506` 的 paired−同状态 text 在 Q 和实际 paired trace 上都有局部有利差值；实际 t50 条件 MSE 仍为 head 2.51408、wrists 2.34419，不能把局部改善当作完整弓步生成。输入完全没呈现的解释不成立，也没有证明呈现充分或训练收敛。
- val `000382` wrists 的有害响应也存在于实际同状态分支；t50 RIC 下肢块差值 +0.653592、上肢 +0.097541。Q 上肢也恶化，不支持仅凭该诊断关闭下肢；rotation6d 损失不能直接当作 RIC 解码的可见错误成因。
- GT 只是同时间索引的固定参考，**不是自由采样状态下的条件期望标签**。Q/实际状态误差差距不能单独识别分布偏移的因果作用。V 的 P−R 对照可补充同状态信息，仍不能回答修正后的动作是否成功。

独立临时目录重建了 01 的 7 份 JSON 和本包 13 文件，全部逐字节一致；**9 项协议测试通过，0 失败/跳过**。禁止神经框架导入及子进程的重建守卫触发 0 次。见[重建核验](b0_state_diagnostic_freeze_checks_20260928_v1/replay_verification.json)、[测试日志](b0_state_diagnostic_freeze_checks_20260928_v1/tests.txt)与[命令](b0_state_diagnostic_freeze_checks_20260928_v1/commands.md)。本轮构建器首次将仓库 symlink 指向的外部依赖误作仓库内路径，构建前停止；修正路径表示后通过，原始异常保留在 `development_symlink.txt`，没有修改 01 产物。

## V 输入、调用与软件锚点

来源完整覆盖 3 train＋3 development-val × head/wrists × t900/500/250/50，共 48 状态。它们来自旧 S1 的 12 条 paired 轨迹，按冻结槽位顺序执行，没有按误差或失败案例筛选。

每组读取已保存 float32 `x_t[1,263,1,T]`，精确复制到 batch4；一次原始 MDM 前向，顺序为：

1. **T**：真实 caption、IMU presence=false、严格 null，IMU encoder 调用为 0。
2. **P**：原 paired 20 Hz 输入。
3. **S**：物理静止，raw acceleration=0、orientation 恒为原始 O[0]。
4. **R**：倒序输入，仅在 20 Hz 输入上使用 `[0,T−1,…,1]`，时间戳不变。

这里分支 R 表示 reversed，全部分支用同一个 R final200 模型。P/S/R 均使用静止参照差分，reference 保持**原始** O[0]；传感器 mask、时间戳、frame mask、caption 不随分支改变。T 保留 paired 数组用于审计，但不得经过 IMU 编码或注入 bias。全 eval、float32、无 AMP/梯度、真实长度不 padding、`mask_frames=false`，timestep 采用原 identity map。

每个 source/config 的 caption 按旧 S1 的 batch1 编码一次，跨分支/四个 t 复用，共 **12 次 caption 调用**。P/S/R 各观测＋参照两份 IMU 编码，第一次状态登记 6 examples，后三次登记 0，共 **72 IMU encoder examples**。缓存绑定 checkpoint、来源/config、归一化、观测/参照数组、mask/time、caption、dtype/eval 身份，禁止错误复用。不能改变原编码 batch/数值路径来追求性能。

每组 3 项、共 **144 软件锚点**：P 对旧 paired trace f11，T 对旧 f10，同组 S 对 T。逐元素界固定为 `abs(new-reference) <= 1e-6 + 1e-6*abs(reference)`；报告 bitwise、max_abs、RMS、count、finite_count、pass。分支位置/人口改变，不要求预先逐位相同，失败后也不放宽界。状态、target、保存锚点、mask/IDs/hash/计数精确一致；重建输入 acceleration 精确，orientation/normalized IMU/reference 为 atol=2e-7、rtol=0，与旧输入口径一致。

先保存原始预测与状态/target/mask 身份再执行锚点判定；可引用已核验不可变文件，不能丢失独立可读数组。保留全部分支/组/check 元数据、实际与预留调用数、耗时、显存、停止原因与产物 hash。软件失败时保留有限预测并标记 diagnostic_only，不把已输出数组改写成“未执行”。

## 预先固定的评价

192 个分支各报告 9 块 MSE，共 1,728 个分支块指标；每个状态的主要比较为 **P−S、P−R**，补充比较为 **P−T、R−S**。共 192 条差值记录（主要 96＋补充 96），每条 9 块，共 1,728 个差值块。另报告原始预测差的 RMS/max/count 及预测、目标量级，避免只看有符号平均而掩盖差异。

每项比较均定义 `e=baseline−GT`、`d=prediction−baseline`，报告 `delta_MSE=2mean(e*d)+mean(d*d)`；保留交叉项正负和残差能量。静止模型预测不等于零 GT 误差。浮点恒等式使用机器包中的 float64/64eps 界，不根据结果调整。

HumanML263 的五个主分区 root4、RIC63、rotation6d126、local_velocity66、contact4 互斥且覆盖全维，按 dims/263 的贡献加和还原整体。RIC 下肢24、躯干头15、上肢24另列，不能与主 RIC 重复相加。所有数值均为预训练归一化特征单位，不是米、物理 IMU 闭环误差或语义成功。

按 source/config/t/train-val 保留单例；同一 split/config/t 内 3 来源等权均值、min/max、有效 n 和缺失原因。四个 t 是同一轨迹的相关状态，不能当四个独立生成样本，不能跨 t 汇成成功分数。48 组、192 分支、144 checks 的计划分母始终保留；失败组的有限数组只做明确标记的诊断，缺失配对记 not_comparable，不按零或通过处理。

历史 Q 使用已有 96 组的同四项比较，共 384 条差值；每个 source/config/t 先平均两次噪声，再与 V 单轨迹并列，报告 192 个 V−Q 差值。它们不是受控的状态分布因果效应。若 Q 有利但 V 不利，只说明这些状态上的条件排序不同；若 V 局部有利但旧完整动作仍失败，只说明局部收益不足。若拍手 P/R 均差于 S，保留不依赖正确顺序的有害响应线索，不直接推导关闭 IMU 或 hard mask。

沿用[旧 120 条完整输出评价](B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md)，保留 train 弓步、右腿保持、val 拍手下肢、步行与踢腿返回、numeric-zero/IMU-only 等所有负例。V 不生成新动作，不新增语义成功率或“修正通过”指标。6 个已暴露来源、固定单训练 seed/每配置单 paired 轨迹的开发诊断不支持泛化、显著性、真实设备、相容编辑或严格 IMU 闭环认证。

## 拟运行边界与 01 交付要求

| 项目 | 冻结上限 |
| --- | ---: |
| MDM batch4 调用 / branch examples | 48 / 192 |
| IMU encoder examples / batch1 caption 调用 | 72 / 12 |
| 新训练 / checkpoint / 完整采样 | 0 / 0 / 0 |
| q_sample / posterior steps / 新噪声 / CFG / motion decode | 0 / 0 / 0 / 0 / 0 |
| GPU / allocated 显存 | 1 / 24 GiB |
| 整队绝对期限 | **300 秒，从预检前起，包含加载、编码、保存和核验** |

在预检前登记全部 48 组、192 分支、144 checks。每次调用前预留计数，分别记录预留、实际调用和完成数。首次 timeout/OOM/nonfinite/身份/计数/软件锚点失败停止整队；保留 completed/failed/not_run_after_stop 与全部计划分母。有限但质量差不构成提前停止条件。无 warmup、重试、fallback、换卡、额外验证前向或重置期限。

01 实现后交付新 manifest、命令、上述完整账本和 CPU fixture 证据：同状态 batch4/固定顺序、12 caption 与72 IMU缓存、T严格null、错误身份拒绝、原始数组先落盘、故障停止/全部分母、禁止未列调用。继续复用已修复的非链式 MDM loader；无须重做旧模型有效性或旧入口验收。

**顺序固定为 01 实现和 CPU 验证 → 02 新入口增量验收 → 01 申请绑定最终 manifest/package/唯一新目录/GPU 的新授权 → 单次预检和执行 → 02 独立结果评价。** 本包 authorization、run_directory、GPU UUID 均为空；当前可以推进实现，不能启动 V 或沿用已消耗的旧授权。
