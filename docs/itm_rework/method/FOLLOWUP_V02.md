# 01 接纳第二批协议交接

读取 `1bc3a41` 的 METHOD_ACCEPTANCE_20260924.md 与 `itm-rework-v0.2-draft`。
本批只修改方法入口、方法测试和本目录；没有改公共契约/数据算子、候选裁定或事实索引。
方法实现提交为 `0574491`；本文随后按 `c9cdc7b` 的独立验收更正记录，勘误提交可用 `git log -1 -- docs/itm_rework/method/FOLLOWUP_V02.md` 定位。

## 已落实

1. 方法侧 `mdm_conditions.py` 复用公共缓存加载和位置时间插值；旋转在同一目标秒时间轴上 SLERP。
   不改变世界系 acceleration/orientation 定义；不引入 GT yaw、初始根位置/骨架变换或一帧偏移。
2. 新训练前缀为 `[0,(min(motion_frames,196)-1)/20]`；IMU 支持不足时报错，不能用 min(IMU长度) 静默缩短。
   归一化输入也验证 schema、物理量和 train split。caption 区间记录下来，仍标为未认证匹配。
3. 新 `crop_seconds` 路径只在真实源支持内生成目标秒网格，明确请求越界即失败。
   超 196 帧仍遵循公开的前缀策略并记 requested_length/truncated_at_model_limit。
   `legacy_stretch` 保留历史重采样/拉伸算法，但缓存的 schema/物理量校验仍执行。
4. sampling spec 可使用 `imu_mode: static`，可选 `static_orientations[6,3,3]` 为声明的常量标定，默认 identity。
   按加载 checkpoint 的 normalization 重新生成 fixture，不复用旧 Stage-2b 归一化数组。
   metadata 保存 fixture normalization hash、presence 与来源类型；`zero`、`null` 含义不变。
5. B 的 manifest 必须 `training_admissible=true` 且无 blockers；不接受仅改标签的 uncertain 候选。
   初版 witness 文件必须包含 `timestamps_seconds`，声明时钟为 HumanML_clip_local_seconds、20Hz。
   模型更新前逐样本核对 metadata 和文件实际时间轴与训练片段完全一致；199 帧 witness 不自动变成 196 帧。
   如果未来需要裁片，须由 02 显式构建/审查该裁片、更新见证和 caption 范围。
6. 当前 arm_spread_m + active-joint-proxy 分支限定 head，避免把固定双腕 joint positions 与改变腕间距作为同时监督目标。
   paired B0 仍支持 head/wrists。这不是对惯性测量本身的相容性定理；其他属性/测量闭环需要单独协议。

## 版本与旧结果

新输入管线名为 `itm-method-seconds-v2`；新 checkpoint 记录该字段，默认 contract_version 为 v0.2。
sampler 保留旧 checkpoint 的 contract_version，另写 input_contract_version/input_pipeline_version，不能把旧训练标成 v0.2。
旧 model.pt、samples 和首批 JSON/快照未改。02 对 `5519365` 的验收仍是该版本的已保存 smoke 验收；
后续 `c9cdc7b` 已独立接纳 `0574491` 的输入正确性，见协议 INPUT_FOLLOWUP_ACCEPTANCE_20260924.md，不扩大为训练或生成效果验收。
旧附属 `_resample_imu`/`_fit_length` 等函数为历史诊断保留，新路径不调用端点夹紧。

## 实际检查与预算

```bash
conda run -n itm python scripts/check_mdm_input_handoff.py --output docs/itm_rework/method/input_handoff_20260924_v2.json
conda run -n itm python -m pytest tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_random_plan.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py -q
```

真实 dry-run 仅加载旧 smoke checkpoint 的统计与输入，不执行模型：按公共 ID 排序取前三个实际为 **000004/000006/000008**，各 head/wrists；
6 条既定 val 面板只检查 **head** 输入，static fixture 检查 head/wrists 两种配置。

**2026-09-24 勘误：**先前将训练清单误写为 000004/000008/000010，现按原始报告更正。
02 在 `c9cdc7b` 中另行显式检查了 000004/000008/000010，全部通过；那是独立补查，不能追溯计入本方法报告。
原始 `input_handoff_20260924_v2.json` 内容与 hash 不变。000006 的 caption 为 [0,5] 秒、输入前缀至 9.75 秒，
因此本检查不构成 caption 对齐认证或正式配对语义评价的依据。
全部通过；48 项回归通过，包含公共候选拒绝测试。数据/几何指标没有用于筛选，未改变训练预算或评价阈值。
CPU、1 thread（输入脚本），模型前向/训练更新/采样/GPU 作业均为 0。

## 交回 02 的待决接口

- 仍需可部署的源世界坐标/骨架到输出约定，caption 对齐认证；本批仅处理可由输入本身校验的秒轴。
- 本批没有将三个 uncertain 候选中的任何一条升为 compatible。
- 真正训练 witness 请提供明确准入标记和与训练裁片一致的实际时间数组；先审查自然度/文本/观测约束与独立阈值。
- 公共 result.schema 的正式 P/C/D 输出接入需冻结实验 manifest 与评价要求；当前诊断不得填入正式指标。
- 现有 B0 checkpoint 仍是单 update smoke；暂无新的 GPU 训练/采样许可。C 与 B+C 的输出观测阻塞不变。

03 可以引用真实入口和正确性更新，不能写 B0/B/C 生成收益或“物理同步问题已解决”。
