# 02 方法交接验收（第二批）

本批读取提交 `55193655b1a8e7835de7e6bd96ed21122957641e`，负责方法实现的 01 文件未被修改。
公共契约升级为 `itm-rework-v0.2-draft`，接受已核查的条件/随机流接口；不冻结物理对齐和研究评价。
v0.1 原文保存在 `contract_history/itm-rework-v0.1-draft.yaml`。旧 checkpoint/结果仍引用 v0.1，不重命名成 v0.2 实验。

## 验收通过的内容

方法与原协议联合回归：65 passed。新增验收、静止、物理时间轴与属性测试：12 passed。
共 77 个不同测试用例通过；日志来自实际命令，见本批 commands 文件。

`method_acceptance_20260924_v2/acceptance.json` 核对了 01 交付代码及共享 random_plan 的 SHA-256。
v1 先做基础核验，v2 增加随机计划地址与 seed/backbone/case 的交叉检查；不是两次独立生成实验。
对 2 条保存的 paired/null smoke 样本：

- 原始 normalized HumanML263 经相同 Mean/Std 和 RIC 解码，与保存 joints 最大差为 **0.0**。
- NPZ/JSON 元数据、checkpoint 和产物 hashes、有效帧、秒时间轴及有限值通过。
- 2×1000 个逐步噪声数组均重新生成并核对 hash；另验证保存的 null 审计 1000 次抽取，共 3000 次。
- paired/null 使用同一随机计划；单条有效长度与噪声 shape 绑定，不能跨长度声称共享相同初始噪声。
- 保存的 strict null 单步和完整样本与原始 MDM 最大差均为 0；legacy null 分别为 0.301335 和 3.204223（归一化 HumanML263 空间）。

这是保存产物的独立验收，没有再次运行完整 DDPM。它不能证明 B0 泛化或语义控制。
01 的 B0 checkpoint 只有 000004 上的一次训练 update，仍不称“验证过的 B0”。

## 逐样本 schema 的接纳边界

`method/smoke_records.jsonl` 采用明确的 method-smoke-handoff-v1，不满足正式公共结果 schema。
02 将其整理为 `accepted_diagnostics.jsonl`，保留 train_diagnostic、protocol=null、各研究指标=null 和研究评价不可用标记。
正式缺失字段见 acceptance.json；没有编造语义/自然度得分、阈值版本或将 train 改成 val。

后续正式 P/C/D 仍使用 result.schema.json；此次增加 diagnostic_result.schema.json，明确诊断和研究输出的不同准入条件。

## 物理静止补充

`itm.data.static_imu.static_imu_fixture` 生成世界系零线加速度、合法常量姿态，再按**指定 checkpoint**的统计归一化。
有效配置为 head/wrists；presence 为 true。数值全零输入不是这个物理状态。

对旧 Stage-2b 权重显式使用 strict gate，在固定 x_t、t=500、12 帧上执行 7 次 CPU forward（原始参照 1 次，每配置 static/zero/null 各 1 次）：

| 配置 | static−numeric-zero 最大差 | strict-null−原始最大差 |
| --- | ---: | ---: |
| head | 0.1616566 | 0 |
| wrists | 0.3800896 | 0 |

单位是 normalized HumanML263 prediction，不是米、不代表生成静止动作的质量。
`static_audit_20260924_v1/` 保存原始/归一化 fixture、预测数组及 hashes。
其中 imu 特征使用旧 Stage-2b normalization；供新 B0 使用时调用函数传入新 checkpoint 的 normalization，不得直接混用已归一化数组。

## 时间轴与坐标

`alignment_20260924_v1/alignment.json` 检查 3 个 train-only 来源和固定 6 条 val 候选。
9 条 HumanML263→new_joints 解码差均为 0。8 条的源 20 FPS 支持帧数比 HumanML 特征多 1，1 条相同。
HumanML 的速度特征会丢弃末帧；这不意味着特征第 0 帧必须向前或向后移一帧。

按真实秒时间插值并对 -1/0/+1 帧都报告 residual 后，9 条经同一个常量 oracle yaw 对齐的 residual 均在 **0 帧偏移**最小。
因此不批准统一的“一帧平移修复”。也不能仅由这 9 条推断所有数据均已同步。
零偏移时，root-relative 全关节原始误差均值约 0.0133–0.2020 m；oracle yaw 后约 0.0057–0.0324 m。
骨架 retarget/尺度差仍存在。

**oracle yaw 使用完整 GT，仅用于定位差异，绝不进入可部署输入预处理。**
统一 skeleton、floor、初始 root/yaw 处理在所读 HumanML 实现中存在，但当前缓存的完整历史生成来源仍需认证。
新 `resample_positions_at_times` 对显式秒时间轴采样，拒绝越界、重复时间和非有限值，不做端点夹紧或全段拉伸。
现有方法的 crop_seconds 已验收为停止全段拉伸；它不等于解决世界坐标/骨架标定。

## 张臂属性与训练候选

`arm_spread_m` 是平均水平腕间距，不能用作“摆臂”的同义指标。
另实现不参与现有 B 训练的肩/肘/髋角度检查 `shoulder_outward_angle_deg`。它是附加几何判据，仍不替代自然度/完整语义审查。

本批按预先固定的 10° sin² 包络，在 3 条 train-only 来源的肩部旋转上肢链，构造 **uncertain** 候选：

| 来源 | 腕间距变化 (m) | 肩外展角均值变化 (°) | 裁定 |
| --- | ---: | ---: | --- |
| 000004 | +0.0211505 | +4.97487 | uncertain |
| 000008 | −0.0124809 | +4.88095 | uncertain，目标腕间距方向失败 |
| 000010 | +0.0329488 | +4.92754 | uncertain |

全部保持 head joint 不变、手臂骨长及端点不变；没有证明安装顶点 IMU 不变或动作自然。
000008 的反例说明“外展角增加”不能替代实际腕间距检查。未调整角度/筛掉反例。
候选在 `arm_spread_candidates_20260924_v1/candidates.jsonl`，公共 compatible manifest 仍为空。
已实际调用 01 的 load_response_manifest，确认会拒绝这些 uncertain 候选。
候选保留完整 HumanML 时间轴；000004 有 199 帧，超过当前 MDM 196 帧上限。未来进入训练前还须显式统一 witness 和训练裁片区间，禁止静默截短来宣布协议已匹配。

若固定两腕 **joint positions**，腕间距已被固定；其变化不超过两腕位置变化范数之和（triangle inequality）。
这是当前 global-joint-proxy 辅助目标的约束，不是“wrists IMU 下张臂必定不相容”的物理结论。
因此不能把当前 arm_spread 目标默认推广为双腕相容编辑实验。

## 接下来仍需的实质工作

1. 确认可部署的设备/源世界坐标到目标表示约定；不使用目标 GT 做隐藏 canonicalization。数据修复可能改变训练分布，须版本化。
2. 从可渲染的真实运动链构造/复核相容 witness，完成自然度、caption、测量闭环和独立阈值；本批候选不具备训练资格。
3. 在数据接口验收后登记 B0 的明确训练/采样预算，先跑固定验证面板；当前没有授权新的 GPU run 或启动 B。
4. 输出仍为 RIC joints，严格 generated-IMU 通道为空；C 与 B+C 继续阻塞。

旧结果影响：旧 Text-only 的 legacy null 不等于原始 MDM，应披露基线定义；此次短 CPU 诊断不提供整套旧结果的数值校正量，也不自动否定 paired-vs-shuffled 的重算。
