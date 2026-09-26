# CPU implementation verification commands

Actual final regression (203 passed):

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py tests/test_mdm_run_limits.py tests/test_b0_correction_protocol.py tests/test_canonical_b0_evaluation.py tests/test_mdm_canonical_conditions.py tests/test_mdm_training_provenance.py tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py tests/test_canonical_pair.py --junitxml=docs/itm_rework/method/correction_implementation_20260926_v1/logs/regression_final.xml > docs/itm_rework/method/correction_implementation_20260926_v1/logs/regression_final.txt 2>&1
```

Final preparation; this recorded directory exists and must not be reused:

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_correction_bounded.py --prepare-dir outputs/itm_rework/method/correction_implementation_20260926_v3
```

For independent replay choose fresh output/log paths. No actual `--execute` command or user authorization record was produced this round.

Final output-status change, targeted worker replay (2 passed; subset of the 203):

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py::test_worker_probe_records_same_forward_activations_and_padding tests/test_mdm_correction.py::test_worker_complete_sampling_cpu_fixture_static_equals_null --junitxml=docs/itm_rework/method/correction_implementation_20260926_v1/logs/output_status.xml > docs/itm_rework/method/correction_implementation_20260926_v1/logs/output_status.txt 2>&1
```
