# Owner02 loader fix differential acceptance

Read HEAD `03e6d95`. No execute, GPU query, actual pretrained model load/forward, training, sampling or authorization.

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_correction_loader_acceptance.py tests/test_mdm_correction.py::test_load_model_with_upstream_nonreturning_device_and_train_methods --junitxml=docs/itm_rework/protocol/correction_loader_acceptance_20260927_v1/targeted.xml > docs/itm_rework/protocol/correction_loader_acceptance_20260927_v1/targeted.log 2>&1
/home/a200/miniconda3/envs/itm/bin/python scripts/accept_correction_loader_fix.py --output docs/itm_rework/protocol/correction_loader_acceptance_20260927_v1/failure_and_binding.json > docs/itm_rework/protocol/correction_loader_acceptance_20260927_v1/audit.log 2>&1
```

Four tests passed (including one expected old-loader failure reproduced in memory), both corrected reference modes passed; no model forward allowed inside loader fixtures. The audit verified four original runtime copies, full480/120 not-run denominators, zero compute counters, the new127 bound files, and unchanged initial/input/environment identities. Owner01's204-test evidence was hash/JUnit checked, not rerun or added to this four-test count. Replay the audit to a fresh output file; it refuses to overwrite evidence.
