# 01 CPU implementation checks

All commands ran from `/home/a200/0proj/ITM`. No GPU query, preflight or execute command ran.

Full regression, exit 0: **258 passed, 0 failed, 0 skipped**, 179.61 seconds; 33 are new alignment method tests. The 45 warnings are existing PyTorch Transformer nested-tensor optimization warnings. Exact selected files and output paths are in the checked-in script:

```bash
bash docs/itm_rework/method/sampling_alignment_implementation_20260927_v1/regression.sh
```

CPU preparation, completed with exit 0. The destination must not already exist; choose a new destination for independent reconstruction. Reads the fixed existing checkpoint on CPU, without constructing a pretrained model or running a forward pass.

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_sampling_alignment_bounded.py --prepare-dir outputs/itm_rework/method/sampling_alignment_implementation_20260927_v1 > docs/itm_rework/method/sampling_alignment_implementation_20260927_v1/logs/preparation.txt 2>&1
```

After preparation, verified every `expected_artifacts` entry with `itm.models.mdm_correction.verify`, reloaded `AlignmentPackage().state()` on CPU to verify final200 provenance, and copied the manifest byte for byte here. `preparation_validation.json` records the resulting manifest/hash and counts. No checkpoint was written in the preparation directory.

Fixtures replace external pretrained factories, CUDA queries and process launch; they are not real model execution. Trace parity fixtures use the actual guidance/sampler for all 1000 steps. Their high-frequency receipt `persist` is replaced in memory; queue/fault tests separately cover durable bookkeeping. Queue authorization is explicitly a temporary CPU fixture with a fake GPU ID and all GPU launch/preflight calls replaced, not a real approval record.

Early development subsets were 20 passed, then queue/auth 9 passed, then worker/trace 11 passed. These overlap and are not added to the final regression count. Final evidence is only `logs/regression.txt` and `logs/regression.xml`.
