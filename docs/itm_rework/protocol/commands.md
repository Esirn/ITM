# 本批命令与结果

工作目录 `/home/a200/0proj/ITM`；Python 为 `/home/a200/miniconda3/envs/itm/bin/python`。
以下命令均已执行；数据脚本的输出目录必须为新的 run ID，重复运行需更换目录名。
旧输出未被覆盖。只读盘点还使用了 git status/rev-parse、rg、cat、文件元数据和 nvidia-smi。

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_itm_rework.py --output docs/itm_rework/protocol/audit_20260924_v1 --cache-samples-per-split 8
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_rework_sources.py --index /home/a200/mount/a40/relatedworks/mdm/HumanML3D/index.csv --amass-root /home/a200/mount/a40/datasets/AMASS --output docs/itm_rework/protocol/source_audit_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python scripts/check_rework_cache_synthesis.py --index /home/a200/mount/a40/relatedworks/mdm/HumanML3D/index.csv --amass-root /home/a200/mount/a40/datasets/AMASS --imuposer-root /home/a200/0relatedworks/IMUPoser --output docs/itm_rework/protocol/source_audit_20260924_v1/cache_synthesis.json
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_rework_cache_full.py --index /home/a200/mount/a40/relatedworks/mdm/HumanML3D/index.csv --amass-root /home/a200/mount/a40/datasets/AMASS --output outputs/itm_rework/cache_audit_20260924_v1
```

结果：重点资产 627 项；全部 8776 缓存检查通过；749 个来源组跨 split，557 对时间区间重叠；
三个源片段重合成已比较，严格 allclose 的 acceleration 未通过（详情保留，不能称完全一致）。

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_mdm_control_conditions.py --manifest outputs/mdm_control/control_test100/specs/manifest.json --results-dir outputs/mdm_control/control_test100 --output docs/itm_rework/protocol/recomputed_20260924_v1/control_test100.json
/home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_mdm_multiseed_control.py --manifest outputs/mdm_control/control_multiseed30/specs/manifest.json --results-dir outputs/mdm_control/control_multiseed30 --output docs/itm_rework/protocol/recomputed_20260924_v1/control_multiseed30.json
/home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_text_attribute_control.py --input 'stage2b=outputs/mdm_control/text_attribute_formal/results/stage2b_attributes_*.npz' --output docs/itm_rework/protocol/recomputed_20260924_v1/text_attributes.json
/home/a200/miniconda3/envs/itm/bin/python scripts/recompute_rework_statistics.py --audit docs/itm_rework/protocol/audit_20260924_v1 --recomputed docs/itm_rework/protocol/recomputed_20260924_v1 --output docs/itm_rework/protocol/statistics_20260924_v3
```

结果：旧表指标复现；136 NPZ / 1880 输出核验成功；来源组 bootstrap 和 round-trip residual 分布完成。
相同统计脚本的 v1 首次失败于历史属性 specs 未含 length；v2 失败于错误假设旧 GT 来自 HumanML263。
按真实 sampler 入口修正核验规则后 v3 完成，并额外记录每条时间拟合情况。旧实验数据没有为了通过核验而改动。

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/plan_rework_validation.py --source-intervals docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl --output docs/itm_rework/protocol/validation_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python scripts/run_rework_stage.py --queue docs/itm_rework/protocol/run_queue.json --output docs/itm_rework/protocol/preflight_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py
```

结果：6 个候选 val cases；队列 dry-run exit 2，明确阻塞于 B0/witness/thresholds；**33 passed**。
测试含 CPU runner 实际执行合成 fixture 的完整路径、拒绝 GPU job、拒绝覆盖 run 目录；fixture 不作为研究证据。
另使用 `torch.load(..., map_location='cpu', weights_only=True)` 提取 Stage-2b 的训练参数与 normalization，
写入 `source_audit_20260924_v1/checkpoint_metadata.json`；未更改 checkpoint。

## 未执行

未重建全数据集、重训 B0/其他方法、重采样任何模型、复算 FID/evaluator embeddings、重跑 SMPL fitting、
生成已认证相容 witness、估计新 JSR/Pareto、冻结最终候选、运行独立测试、修改论文、commit/push/merge 或上传数据。
这些不标为成功；原因和交接路径见 STATUS 与 HANDOFF。
