# 已执行一次的历史命令（授权已消耗，不能重跑）

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_feature_balance_bounded.py --execute --implementation outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json --run-dir outputs/itm_rework/method/feature_balance_authorized_20261007_v1 --authorization-record docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_20261007.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 > docs/itm_rework/method/feature_balance_run_20261007_v2/execute.log 2>&1
```

父进程退出码 1。完整 traceback 已保存，原运行目录只有 status.json。

已完成的 CPU 后处理：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_feature_balance.py --run outputs/itm_rework/method/feature_balance_authorized_20261007_v1 --output outputs/itm_rework/protocol/feature_balance_failed_evaluation_20261007_v2 --no-render

/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/feature_balance_run_20261007_v2/verify_delivery.py
```

评价重放须另选新输出目录；核验脚本只读，可直接重放，不调用 GPU 或模型。
