# 01 → 02：缓存身份与恢复训练来源核验

读取 `2de27a6`（caption 独立验收）与 `d360e91`（前缀诊断）。公共契约仍为
`itm-rework-v0.2-draft`。本批提交可用 `git log -1 -- docs/itm_rework/method/CACHE_MIGRATION_HANDOFF_20260925.md` 定位。
接纳 02 的 109 项独立回归及其验收边界；四份加速度比较未通过原阈值的负结果保持不变。

## 修复

1. `load_method_cache` 核对 NPZ 内嵌 schema 与 manifest。旧标准缓存没有内嵌字段时仍走原公共 reader；有字段时要求标量且一致。
   将诊断 manifest 改成 `standard_imu_v1`、甚至改成 `training_admissible=true`，不能绕过文件内的 diagnostic schema。
   此检查只防止 manifest 与实际文件声明不一致，不认证缓存内容，也不防止有人同时伪造 NPZ 与 manifest。
   共享的 `src/itm/data/observation.py` 没有修改；02 可据此考虑公共 reader 的相应修复。
2. 新训练 checkpoint 增加 `training_input_provenance`，内部版本 `itm-training-input-provenance-v1`。
   绑定完整 motion/IMU manifest、实际选中 motion/text/IMU 文件、caption sidecar（启用时）、Mean/Std、原始 MDM 权重/参数文件的路径与 SHA256；
   同时绑定所选样本和配置顺序、输入管线版本、契约版本及实际计算的 IMU normalization。
   文件哈希逐块读取，每个唯一文件在一次构建内只计算一次。完整 manifest 的变化（即使发生于未选中行）也会保守拒绝恢复。
3. 归一化计算、输入来源核验和 control config 核验现在位于 MDM 导入/模型创建/设备初始化之前。
   `--resume` 必须匹配上述来源，且 checkpoint 的统计必须与当前重新计算的统计一致；移除了无条件用旧统计覆盖当前统计的行为。
   仍是 **fresh optimizer 的权重 warm start**，允许后续改变损失权重/学习率；不声称恢复 optimizer/RNG 的逐步精确续训。

权重布局与 `itm-control-v2` 外层保持兼容，新增来源字段自带版本；seconds-v2 / caption-seconds-v3 的合法输入数值变换没有改变。
没有修改公共契约 schema，新增字段是方法 checkpoint 的内部来源记录。

## 兼容边界

旧 checkpoint 采样加载不变，但旧 checkpoint 没有上述来源记录时，当前训练 CLI 的 `--resume` 会明确拒绝。
跨数据/源映射/归一化/路径的迁移也会拒绝，不自动补签来源。可在新输出路径不传 `--resume`，从原始 MDM 新建控制分支；
这仍须满足已有数据准入和预算要求。本批没有提供跨数据权重迁移机制。
精确路径也参与比较，因此复制到新目录不会被自动视作同一次训练。
哈希描述初始化时读取的文件，假定作业期间输入保持不变；没有提供运行中持续文件监控。

## 已执行检查

```bash
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_training_provenance.py tests/test_mdm_conditions.py tests/test_mdm_caption_conditions.py
/home/a200/miniconda3/envs/itm/bin/python scripts/check_mdm_cache_migration.py --output docs/itm_rework/method/cache_migration_20260925_v1.json
/home/a200/miniconda3/envs/itm/bin/python scripts/check_mdm_caption_handoff.py --output docs/itm_rework/method/caption_input_migration_20260925_v1.json
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_training_provenance.py tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py > docs/itm_rework/method/logs/cache_migration_20260925.txt 2>&1
```

- 定向 29 项通过；联合 **126 passed, 19 warnings**（既有 PyTorch warnings）。含同路径内容替换、管线/契约/caption/config 改变、统计被修改、旧 checkpoint 缺失来源的拒绝；checkpoint 序列化和相同来源 warm-start 校验通过。
- CLI 测试证明缺失来源或统计不符时，在 MDM 导入/设备检查前拒绝。测试使用小 fixture，没有启动真实训练。
- 固定 `000123/000015/000506/004728/000027`：5 条原诊断 manifest 与 5 条仅在内存中改标签的 manifest 全部拒绝；文件 SHA256 与 02 记录一致，检查前后未变化。
- 固定 caption 输入重放：18 通过、2 按计划阻塞；13 条 SSM 继续拒绝。新 JSON 与 `caption_input_20260925_v1.json` 逐字段相等。
- 真实输入检查模型前向、训练 update、生成、GPU 作业均为 0；没有重算真实训练集统计、写新研究 checkpoint 或更改历史产物。

代码与公共依赖快照见 `code_snapshot_cache_migration_v1.json`。旧快照保留为原验收对象，不应拿旧快照匹配本批修改后的代码。

## 剩余依赖

本批只解决错误缓存接入和归一化静默迁移风险。物理/语义对齐、非零裁片根参考、兼容编辑 witness、阈值与预算仍未认证。
不安装前缀补偿/SSM 时钟修正，不放宽加速度阈值，不将五份诊断缓存用于训练。
G2 无新 B0，B/C 无训练或生成收益证据。02 可独立验收本批输入和恢复逻辑；03 不应将其表述为对齐认证或模型能力提升。
