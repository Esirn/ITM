# CPU-only proposal construction and verification

From `/home/a200/0proj/ITM`:

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/b0_state_diagnostic_20260928_v1/build_proposal.py > docs/itm_rework/method/b0_state_diagnostic_20260928_v1/build.log 2>&1
```

The builder imports only NumPy and the Python standard library. It reads saved Q/trace predictions, frozen slot specifications, owner02's Q metrics, and 200 historical training update JSON files. The checkpoint is hashed as bytes, not deserialized. It never calls a neural model, optimizer, sampler, GPU query or subprocess.

The assertions verify signed analytic residual examples, exact 263-feature partitioning, per-block float64 error identities, agreement with 288 owner02 Q MSEs, 528 saved-state decompositions, 600 unique exposures, and proposed 48-group/192-branch/72-encoder counts. This is a CPU evidence build, not execution or software acceptance of the proposed V worker.

Independent-directory reconstruction used `runpy.run_path(build_proposal.py, run_name='proposal_definition_only')`, redirected the returned main function's global `OUT` to a fresh `/tmp/itm_state_proposal_rebuild_*` directory, and invoked `main()` with imports of `torch`, `tensorflow`, `jax` and subprocess launch/query functions forbidden. All seven generated JSON files were byte-identical; all 1073 recorded sources matched their hashes. No forbidden operation was attempted. Exact directory and output hashes are retained in `rebuild_verification.json`.

The builder emits the seven data JSON files; it does not overwrite `rebuild_verification.json`, this command record, or the proposal Markdown. Rebuilding the proposal never starts V. Its budget status remains `proposed_not_frozen_not_authorized` with no GPU UUID, run directory or authorization.
