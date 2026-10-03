# V 已授权单次运行及交付核验

真实命令仅执行一次，退出码 0。授权已消耗，以下是历史记录，不得直接重跑。

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_state_diagnostic_bounded.py --execute --implementation /home/a200/0proj/ITM/outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1/implementation.json --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1 --authorization-record /home/a200/0proj/ITM/docs/itm_rework/method/B0_STATE_DIAGNOSTIC_AUTHORIZATION_20261003.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 > docs/itm_rework/method/state_diagnostic_run_20261003_v1/logs/parent.log 2>&1
```

只读 CPU 交付核验：

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/state_diagnostic_run_20261003_v1/verify_delivery.py
```

核验只使用标准库与 NumPy，不加载模型或查询 GPU。核验脚本首次执行的 target shape 断言误去掉了特征轴；按保存格式 (263, 1, T) 修正后完整核验通过。没有修改运行代码、原始产物或重新运行模型。
