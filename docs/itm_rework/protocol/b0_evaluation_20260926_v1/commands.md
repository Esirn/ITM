# 02 B0 evaluation commands and outcomes

Working directory: `/home/a200/0proj/ITM`; read HEAD: `4ae11d0`.
No model forward, training, sampling, GPU preflight or GPU job was performed.

```bash
PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_canonical_b0_evaluation.py
PYTHONPATH=src OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_canonical_b0.py audit
PYTHONPATH=src OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_canonical_b0.py render
PYTHONPATH=src OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_canonical_b0.py compare
```

Results: 6 tests passed; audit valid_slots=36, failure_rate=0, noise_draws_replayed=6000; six condition-hidden review pages rendered (36 outputs); 24 source-paired comparisons and 36 CSV rows exported.

After reading all six pages, owner02 wrote `semantic_notes_condition_hidden.json` before reading the case key; `semantic_review_joined.json` joins those notes to `review_key.json` on `review_id`. No scalar semantic success rate was computed. Image paths and hashes are in `review_rendering.json`.

The final script simplified the backbone hash checker and added comparison/CSV export. Final verification:

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_canonical_b0.py audit --output /tmp/itm_b0_evaluation_20260926_replay_v1
PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_canonical_b0_evaluation.py
```

Again 36 valid, 6000 unique draws replayed, 6 tests passed. `evaluation.json` and `review_key.json` matched the first run byte-for-byte. Final code hashes and artifact audit are copied to `artifact_audit_replay.json`; `replay_verification.json` records the comparison. The first `artifact_audit.json` binds the earlier script version and remains historical evidence, not the final code identity.

For future reproduction, use a **fresh** `--output` directory for `audit` and pass the same directory to `compare`; audit rejects an existing evaluation. `render` writes derived images to the documented review directory; copy/archive any locally modified review images before deliberately re-rendering. Authoritative raw run outputs and frozen data are read-only inputs throughout.

The CSV export was normalized from the standard csv.writer CRLF line endings to LF for repository whitespace checks; values and column order are unchanged.
