# Executed commands and outcomes

One authorized queue, exit 1; historical command, do not rerun or reuse this directory/authorization.

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_correction_bounded.py --execute --implementation outputs/itm_rework/method/correction_implementation_20260926_v3/implementation.json --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/correction_authorized_20260927_v1 --authorization-record docs/itm_rework/method/B0_CORRECTION_AUTHORIZATION_20260927.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004
```

CPU loader regression: before fix exit 1 (same AttributeError), after fix exit 0.

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py::test_load_model_with_upstream_nonreturning_device_and_train_methods > docs/itm_rework/method/correction_run_20260927_v1/logs/loader_before_fix.txt 2>&1
# After applying the loader fix:
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py::test_load_model_with_upstream_nonreturning_device_and_train_methods > docs/itm_rework/method/correction_run_20260927_v1/logs/loader_after_fix.txt 2>&1
```

Full CPU regression:

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py tests/test_mdm_run_limits.py tests/test_b0_correction_protocol.py tests/test_canonical_b0_evaluation.py tests/test_mdm_canonical_conditions.py tests/test_mdm_training_provenance.py tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py tests/test_canonical_pair.py --junitxml=docs/itm_rework/method/correction_run_20260927_v1/logs/regression.xml > docs/itm_rework/method/correction_run_20260927_v1/logs/regression.txt 2>&1
```

CPU-only replacement preparation, exit 0. For replay choose a fresh directory.

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_correction_bounded.py --prepare-dir outputs/itm_rework/method/correction_implementation_20260927_v1 > docs/itm_rework/method/correction_run_20260927_v1/logs/preparation.txt 2>&1
```

Read-only post-failure GPU process check, exit 0; no compute process on the authorized GPU:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

No second execute or retry command was run.
