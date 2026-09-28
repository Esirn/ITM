# V implementation CPU checks

Run from `/home/a200/0proj/ITM`. No GPU preflight or execute command ran.

Final incremental regression: **135 passed / 0 failed / 0 skipped**, 159.19 seconds; 29 new V method tests. The28 warnings are existing Transformer nested-tensor optimization notices. Exact selection and result paths:

```bash
bash docs/itm_rework/method/state_diagnostic_implementation_20260928_v1/regression.sh
```

This includes the new V method/protocol plus reused sampling-alignment, limit, correction, loader and IMU-control tests. Old runtime code is unchanged; this is a focused dependency regression, not a new effectiveness test.

CPU preparation, exit 0. Choose a fresh destination for replay; this directory must not be overwritten.

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_state_diagnostic_bounded.py --prepare-dir outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1 > docs/itm_rework/method/state_diagnostic_implementation_20260928_v1/logs/preparation.txt 2>&1
```

Preparation validates existing R tensors on CPU, 12 caches / 48 branch input sets / 48 saved-state groups, and creates only implementation.json. The manifest was copied byte for byte here; all1656 expected artifacts and the environment were rechecked. No pretrained model construction/forward, new checkpoint, GPU query or real authorization was produced.

Early development subsets passed19 and29 tests; these overlap the final suite and are not added to its count. Code review then identified the upstream native-half CLIP convention; the final fixture explicitly includes half CLIP parameters while keeping the denoiser and embeddings float32. Final regression logs cover this final implementation.
