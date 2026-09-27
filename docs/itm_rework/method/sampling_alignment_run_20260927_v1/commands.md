# Authorized single execution

Historical command; do not rerun this run directory or reuse this authorization. GPU preflight is inside the total deadline.

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_sampling_alignment_bounded.py --execute --implementation /home/a200/0proj/ITM/outputs/itm_rework/method/sampling_alignment_implementation_20260927_v1/implementation.json --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1 --authorization-record /home/a200/0proj/ITM/docs/itm_rework/method/B0_SAMPLING_ALIGNMENT_AUTHORIZATION_20260927.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004
```

Post-run CPU artifact identity/finite audit (no model calls):

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/sampling_alignment_run_20260927_v1/verify_delivery.py > docs/itm_rework/method/sampling_alignment_run_20260927_v1/logs/delivery.txt 2>&1
```
