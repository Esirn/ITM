# V 单次运行及失败交付命令

2026-10-03 执行此前已批准的 20260928 身份。以下真实运行命令只执行一次，退出码 1；授权已消耗，禁止直接重跑。

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_state_diagnostic_bounded.py --execute --implementation /home/a200/0proj/ITM/outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1/implementation.json --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1 --authorization-record /home/a200/0proj/ITM/docs/itm_rework/method/B0_STATE_DIAGNOSTIC_AUTHORIZATION_20260928.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 > docs/itm_rework/method/state_diagnostic_run_20260928_v1/logs/parent.log 2>&1
```

CPU 交付核验（无模型导入、前向或 GPU 查询）：

```bash
python docs/itm_rework/method/state_diagnostic_run_20260928_v1/verify_delivery.py
```

核验通过：1656 份绑定文件、授权/manifest/package/checkpoint 身份、原始与归档账本逐字节一致、48/192/144 全分母、全部调用计数为 0。原始预检 stdout 未由已验收入口保存，不能事后推断当时显存或身份值。
