# 01 → 02：caption 计划消费接口

读取协议提交 `9439818`（此前 caption 提交 `e14431a`），公共契约保持 `itm-rework-v0.2-draft`。
改动提交通过 `git log -1 -- docs/itm_rework/method/CAPTION_HANDOFF_20260925.md` 定位。
本轮只改方法入口/测试/本目录，不改 data 算子、公共计划、cache、源映射、候选准入或论文。

## 实际实现

新模块 `src/itm/models/mdm_caption_conditions.py` 显式消费 02 JSONL sidecar。
新输入管线 `itm-method-caption-seconds-v3` 是 **opt-in**；无 `--caption-plan` 时保留 seconds-v2 诊断路径及既有产物复现。
不会将已有 v0.1 checkpoint 追认为 caption 管线训练。

- 绑定 motion ID、split、text/feature/cache 的 SHA256，读取明确的非空行 caption_index，使用公共严格解析器。
- 由真实特征帧数和 cache 时间支持重算公共 plan，并与 sidecar 的文本、半开 frame_slice 和两个时间数组逐项核对。
- motion 按 `[start,stop)` 读取；IMU 在 `source_times_seconds` 上插值，encoder 输入为从零开始的 `model_times_seconds`。
- 两套时间数组、源起点、原标签、caption index 和 artifact hashes 随输入 metadata 交付。
- 过长全段、越界、文本/长度覆盖、陈旧 hash 或修改过的时间数组均拒绝，不能静默变成前缀。
- 训练数据适配器与采样 loader 使用相同实现；启用计划时，归一化统计也只取显式选定片段（本轮没有重算真实训练统计）。
- 训练 CLI 在模型/设备初始化前要求明确 training_admissible 与世界/语义对齐认证；已有所有 false 计划均拒绝，即使传了旧 diagnostic override。
- 非零起点的 B 编辑分支继续拒绝复用前缀 witness，必须另行认证 witness 与 root-frame。新接口不认证或重做根位置/朝向。

没有使用 `humanml_clock.py` 的诊断映射改变运行时秒轴；不 round 源 FPS、不改变 acceleration 单位、不拟合 GT yaw。
SSM 的 13 条既有短输入仍按旧请求区间因支持不足失败。2177 条前缀候选没有被当成逐条确认的错误或新训练清单。

## CLI/接口

训练 CLI 增加 `--caption-plan <cases.jsonl>`；从中显式消费 train 行。当前公共 sidecar 无训练准入，因此预检失败是预期行为。
`_PairedDataset(..., caption_plans=...)` 允许 input-only dry-run 读取未准入行，并始终报告 uncertified_diagnostic。

采样 CLI 增加 `--caption-plan <cases.jsonl> --motion-manifest <HumanML-manifest.jsonl>`，需要 `crop_seconds`。
spec 指定 motion_id/sensor_config；默认使用计划中的文本和完整长度。若显式写 text/length，则必须与计划一致。
支持 paired/null/zero/static；caption-plan 下 shuffled donor 或编辑文本需另行明确时间映射/审查计划，当前拒绝。
旧 seconds-v2 的 shuffled 接口不受此限制变更。本轮没有调用实际采样 CLI 生成动作。

对非零起点，motion feature 只做原表示切片，不 recanonicalize；原片段 world GT 与解码后的根参考未认证，
这也是本接口不能直接升级正式训练/评价的原因。

## 实际输入检查

使用 02 固定 10 条计划，没有改 ID、索引或选取规则：

| ID / 范围 | 结果 |
|---|---|
| train 000004 | head/wrists 均 blocked：199 帧全段超过 196，不裁 caption 或 witness |
| train 000006 | head/wrists：100 帧，source `[0,4.95]`，模型时间同长度，从零开始 |
| train 000008 / 000010 | head/wrists：42 / 69 帧 |
| val 000133 | head/wrists：明确 caption_index=1，59 帧，末采样 2.9 秒 |
| 其余 5 条固定 val | head/wrists 通过原计划的完整长度与文本 |
| 13 条已锁定 SSM 短输入 | 全部继续拒绝，无插补/拉伸/时钟修正 |

共 9 个通过来源 × 2 配置、1 个阻塞来源 × 2 配置；输出 `caption_input_20260925_v1.json` 保留所有结果。
3 个成功 train 来源的训练 loader 与采样 loader IMU 数组/文本完全一致。
这次确实检查 val 两配置，不追溯改变以前仅 head 的报告。
全部真实数据检查模型前向 0、训练 update 0、生成 0、GPU 0；CPU 单线程。

## 命令与回归

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/check_mdm_caption_handoff.py --output docs/itm_rework/method/caption_input_20260925_v1.json
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_caption_time.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_method_handoff_acceptance.py tests/test_physical_time.py tests/test_static_imu.py tests/test_humanml_clock.py
```

68 passed，19 条既有 PyTorch warnings；日志 `logs/caption_input_20260925.txt`。
新测试包含非零起点双时间轴、索引 1、训练/采样一致、裁片统计、hash/文本/长度覆盖拒绝、false 准入在设备初始化前拒绝及 witness 隔离。
仅软件测试使用小型模型 fixture；与真实研究作业分开。
交付代码与公共依赖 hash 见 `code_snapshot_caption_v3.json`。旧日志、快照、报告、checkpoint 和生成样本没有覆盖。

## 仍待 02 / 后续阶段

caption 区间消费已实现，但物理时钟、部署世界坐标、骨架、非零裁片根参考与文本语义还未联合认证。
没有新的 B0 checkpoint、compatible witness、冻结阈值或 GPU 预算。B/C/B+C 和正式 P/C/D 评价继续阻塞。
03 可写输入实现/拒绝机制，不得写已解决物理同步或取得生成收益。
