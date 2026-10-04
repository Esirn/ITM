# 02 独立复现命令

工作目录 `/home/a200/0proj/ITM`，被验收 HEAD `963a132129b808b4c43af0c5bea30fc735ddcc67`。以下为实际执行命令；复现时使用新的审计输出文件和不存在的 prepare-dir，避免覆盖本批证据。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q \
  tests/test_mdm_feature_balance.py tests/test_b0_feature_balance_protocol.py \
  tests/test_mdm_correction.py tests/test_mdm_sampling_alignment.py \
  tests/test_mdm_run_limits.py tests/test_correction_loader_acceptance.py \
  tests/test_mdm_imu_control.py --disable-warnings \
  > docs/itm_rework/protocol/b0_feature_balance_acceptance_20261004_v1/regression.txt 2>&1

OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python \
  scripts/accept_b0_feature_balance.py \
  --output docs/itm_rework/protocol/b0_feature_balance_acceptance_20261004_v1/audit.json \
  --prepare-dir /tmp/itm02_feature_balance_acceptance_20261004_v1 \
  > docs/itm_rework/protocol/b0_feature_balance_acceptance_20261004_v1/audit.log 2>&1
```

回归 155 passed/31 warnings，157.77 秒。CPU 小型 fixture 不属于真实训练/采样；审计禁止实际模型构造/前向、CUDA 查询/初始化、预检与进程启动。重建 manifest SHA 为 `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`。
