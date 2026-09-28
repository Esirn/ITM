#!/usr/bin/env bash
set -eu
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_state_diagnostic.py tests/test_b0_state_diagnostic_protocol.py tests/test_mdm_sampling_alignment.py tests/test_mdm_run_limits.py tests/test_mdm_correction.py tests/test_correction_loader_acceptance.py tests/test_mdm_imu_control.py --junitxml=docs/itm_rework/method/state_diagnostic_implementation_20260928_v1/logs/regression.xml > docs/itm_rework/method/state_diagnostic_implementation_20260928_v1/logs/regression.txt 2>&1
