# 第二批执行命令

读取 HEAD `55193655b1a8e7835de7e6bd96ed21122957641e`。全部 CPU；无新模型训练、完整生成或 GPU 作业。
第一批 commands.md 原文与哈希保留。

```bash
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py
/home/a200/miniconda3/envs/itm/bin/python scripts/accept_method_handoff.py --output docs/itm_rework/protocol/method_acceptance_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python scripts/accept_method_handoff.py --output docs/itm_rework/protocol/method_acceptance_20260924_v2
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_physical_static.py --output docs/itm_rework/protocol/static_audit_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_humanml_imu_alignment.py --output docs/itm_rework/protocol/alignment_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python scripts/build_arm_spread_candidates.py --output docs/itm_rework/protocol/arm_spread_candidates_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py
```

首行 65 passed、19 个已有 PyTorch nested-tensor 配置 warnings；末行 12 passed。
最终将这两组测试路径合并为同一次 pytest 调用：77 passed、19 warnings，原始日志保存于 `logs/acceptance_regression.txt`。
新增测试编写期间也分别执行过 static/time 7 passed、arm 3 passed；不把重复执行数当独立测试数。
静止审计：7 次 CPU forward、2 threads，保存 normalized 预测。PyTorch runtime 有既有 AMP deprecation warnings，未影响执行。
交接验收：v1/v2 都成功；v2增加一致地址、采样 seed/backbone/case 和训练来源查验。
时序诊断：9 条，3 个固定候选 lag，没有将拟合的 oracle yaw/lag应用到输入。
候选：3 条、固定 10° 构造、无搜索；1 条腕间距方向失败保留。训练入口拒绝 uncertain 已测试。
更新队列后另执行 `scripts/run_rework_stage.py --queue docs/itm_rework/protocol/run_queue.json --output docs/itm_rework/protocol/preflight_20260924_v2`；预期 exit 2，准确保留正式 C 的实质阻塞。

未执行：B0 正式训练、B compatible witness 训练、G3/G4 新生成、C、独立测试或论文修改。
