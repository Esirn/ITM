# 02 对缓存身份与恢复训练来源检查的独立验收

验收方法提交 `b4f70fcd61a07c5539bdbc5c473a5c8fe91a6256`，公共契约保持 `itm-rework-v0.2-draft`。
02 本批改动提交可用 `git log -1 -- scripts/accept_cache_migration_handoff.py` 定位。
接纳方法侧缓存身份及恢复来源检查，不代表数据或训练准入。

## 实际复核

- 方法快照的 6 个自有文件和 5 个公共依赖 hash 均匹配。
- 五份原 diagnostic manifest 和五份仅在内存中重标 schema/admission 的 manifest 均被拒绝；独立重放与 01 的保存 JSON 完全一致，原 cache 文件 hash 不变。
- caption 重放既与本批方法报告一致，也与此前已验收的原始 v3 报告一致：18 输入通过、2 输入按计划阻塞、13 条 SSM 拒绝。
- 直接加载旧 smoke checkpoint 到 CPU，确认 hash、normalization 与原模型卡一致；其缺少 `training_input_provenance`，调用恢复验证函数确实拒绝。
- 联合回归 **126 passed, 19 warnings**，warnings 为既有 PyTorch 提示。包含同路径内容替换、manifest/backbone/Mean/Std/caption/config/pipeline 变化、旧统计篡改、checkpoint 序列化及相同来源 warm start；fixture 证明错误恢复在 MDM 导入/设备检查前拒绝。

本轮真实输入检查无模型前向、训练、生成或 GPU。没有重算真实训练集统计，也没有新研究 checkpoint。

## 接纳范围与兼容性

当前 provenance 覆盖明确列出的 paired 输入文件、manifest、Mean/Std、backbone 权重/参数、可选 caption sidecar，以及样本/配置顺序、输入管线、契约版本和实际 normalization。
完整 manifest 或绝对路径变化会保守拒绝恢复；这些字段不是物理/语义认证，也不代表已绑定所有可能的后续训练监督来源。

恢复仍是重新创建 optimizer 的权重 warm start，允许有意改变学习率/损失设置；不是 optimizer/RNG 的逐步精确恢复。
旧 checkpoint 的采样加载代码未改，旧文件 CPU 加载和输入重放已核实；**没有重新执行完整采样来验证生成结果**。
旧文件不能自动补签来源来通过 `--resume`；未来新训练仍受数据准入及预算约束。

嵌入 schema 检查位于 `load_method_cache`，共享 `load_standard_cache` 未改。因此这里确认的是方法输入入口防止 manifest 单独重标，不是所有公共 reader 都具有此检查，也不声称能检测同时伪造 NPZ 与 manifest。
上一批四份 acceleration allclose 负结果、前缀/SSM 时钟及非零根参考问题不因该检查通过而消失；五份诊断缓存仍不得用于训练。

## 复现与资源

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/accept_cache_migration_handoff.py --output-dir docs/itm_rework/protocol/cache_migration_acceptance_20260925_v1
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_training_provenance.py tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py > docs/itm_rework/protocol/logs/cache_migration_acceptance_regression_20260925.txt 2>&1
```

输入重放 1 CPU thread、两个子进程各上限 120 秒。新目录 `cache_migration_acceptance_20260925_v1/` 保留验收 JSON、两份重放 JSON 和日志；验收脚本和依赖/原始产物 hash 同步记录。
本批只新增 02 验收及证据文档，没有修改方法实现、公共契约、旧文件或训练准入。
01 无需为同一验收再做确认性提交；后续交接应提供新的实质对齐/数据证据。03 可引用拒绝机制，不能写作新 B0/B/C 效果或物理闭环通过。
