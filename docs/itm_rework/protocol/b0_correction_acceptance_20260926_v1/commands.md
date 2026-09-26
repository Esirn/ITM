# Owner02 independent CPU software acceptance

Read commit: `8e4df1e`. No real MDM forward, training, diffusion sample, GPU query or authorization. The regression includes explicitly labelled CPU model/stub fixtures.

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py tests/test_mdm_run_limits.py tests/test_b0_correction_protocol.py tests/test_canonical_b0_evaluation.py tests/test_mdm_canonical_conditions.py tests/test_mdm_training_provenance.py tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py tests/test_canonical_pair.py --junitxml=docs/itm_rework/protocol/b0_correction_acceptance_20260926_v1/regression.xml > docs/itm_rework/protocol/b0_correction_acceptance_20260926_v1/regression.log 2>&1
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/accept_b0_correction_implementation.py --output docs/itm_rework/protocol/b0_correction_acceptance_20260926_v1/independent --prepare-dir outputs/itm_rework/protocol/b0_correction_acceptance_prepare_20260926_v1 > docs/itm_rework/protocol/b0_correction_acceptance_20260926_v1/independent.log 2>&1
```

Results: 203 passed, 34 warnings, 0 failures/errors/skipped, 100.94 seconds; all120 input slots match an independent raw-NPZ interpolation/normalization reference; 60 corresponding inputs are equal across arms. Preparation completes with all module forwards and GPU operations prohibited. Initial tensor hashes and the entire initial.pt match owner01 v3.

For replay, choose fresh --output and --prepare-dir paths. Do not execute a real queue or create an authorization during acceptance.
