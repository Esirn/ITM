#!/usr/bin/env bash
set -euo pipefail
cd /home/a200/0proj/ITM
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q \
  tests/test_mdm_feature_balance.py tests/test_b0_feature_balance_protocol.py \
  tests/test_mdm_correction.py tests/test_mdm_sampling_alignment.py \
  tests/test_mdm_run_limits.py tests/test_correction_loader_acceptance.py \
  tests/test_mdm_imu_control.py --disable-warnings
