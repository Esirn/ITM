# F 已授权单次执行记录

以下真实命令仅执行一次，父入口退出码1，GPU查询退出码9。授权已消耗，不得直接重跑。

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_feature_balance_bounded.py --execute --implementation /home/a200/0proj/ITM/outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/feature_balance_authorized_20261004_v1 --authorization-record /home/a200/0proj/ITM/docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_20261004.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 > docs/itm_rework/method/feature_balance_run_20261004_v1/logs/parent.log 2>&1
```

只读CPU核验（无模型导入/GPU查询）：

```bash
python docs/itm_rework/method/feature_balance_run_20261004_v1/verify_delivery.py
```
