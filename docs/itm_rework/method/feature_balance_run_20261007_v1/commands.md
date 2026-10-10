# 本次实际执行与只读重放

以下第一条是已完成且已消耗授权的历史执行记录，**不能重跑**：

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_feature_balance_bounded.py --execute --implementation outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json --run-dir outputs/itm_rework/method/feature_balance_authorized_20261005_v2 --authorization-record docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_20261005_V2.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 > docs/itm_rework/method/feature_balance_run_20261007_v1/execute.log 2>&1
```

退出码 1，运行目录仅有失败 status.json。原始 status 的逐字节副本位于本目录 runtime/status.json。

已执行的 CPU 后处理：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_feature_balance.py --run outputs/itm_rework/method/feature_balance_authorized_20261005_v2 --output outputs/itm_rework/protocol/feature_balance_failed_evaluation_20261007_v1 --no-render

/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/feature_balance_run_20261007_v1/verify_delivery.py
```

评价入口拒绝覆盖已有目录。需要重放评价时另选新的 CPU 输出目录；自检脚本只读，可直接重放。原 CPU 准备的 verify_preparation.py 包含“当时运行目录尚不存在”的检查，授权执行后该历史假设不再成立，不应修改旧证据来使其通过；本次状态使用 verify_delivery.py。
