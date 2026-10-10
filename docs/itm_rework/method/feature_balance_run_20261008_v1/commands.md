# 本次完成的执行与CPU重放

已完成的历史GPU命令（目录不可复用）：

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_feature_balance_bounded.py --execute --implementation outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json --run-dir outputs/itm_rework/method/feature_balance_authorized_20261008_v1 --authorization-record docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_20261008.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004
```

CPU评价先生成v1全量表/交互页面，120 hidden-ID notes保存后生成v2最终表。重放需给`--output`另选不存在的目录：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_feature_balance.py --run outputs/itm_rework/method/feature_balance_authorized_20261008_v1 --output /tmp/itm_F_effect_replay_fresh --review-notes docs/itm_rework/method/feature_balance_run_20261008_v1/review_notes_hidden_final.json --no-render

OMP_NUM_THREADS=2 PYTHONPATH=src:scripts /home/a200/miniconda3/envs/itm/bin/python scripts/audit_b0_feature_balance_traces.py --run outputs/itm_rework/method/feature_balance_authorized_20261008_v1 --output /tmp/itm_F_trace_replay_fresh

OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/feature_balance_run_20261008_v1/verify_results.py
```

本轮相关测试：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_feature_balance_evaluation.py tests/test_canonical_b0_evaluation.py
```

图表由`scripts/render_b0_feature_balance_review.py`生成20页冻结顺序图，主000506的12个review_id额外生成每帧oblique/ZY图。渲染目录`outputs/itm_rework/protocol/feature_balance_review_images_20261008_v1`内rendering.json绑定图和源数组hash。主GT/U/F全帧曲线与all120_metrics.csv/paired_effect_table.json是最终保存数组的CPU汇总，未改变任何比较单位、坐标或时间。
