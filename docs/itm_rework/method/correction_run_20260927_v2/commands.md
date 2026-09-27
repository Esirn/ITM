# Authorized v2 queue: completed, exit code 0

This is the single execution authorized by the user’s “批准通过”. Historical command only; do not rerun this directory or reuse the authorization.

```bash
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_correction_bounded.py --execute --implementation outputs/itm_rework/method/correction_implementation_20260927_v1/implementation.json --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/correction_authorized_20260927_v2 --authorization-record docs/itm_rework/method/B0_CORRECTION_AUTHORIZATION_20260927_V2.json --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004
```

CPU delivery verification, after the queue has stopped; reads saved artifacts only:

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/correction_run_20260927_v2/verify_delivery.py > docs/itm_rework/method/correction_run_20260927_v2/delivery_integrity.json
```

Delivery verifier completed with exit code 0. Post-run read-only GPU check also exited 0; no compute process remained on the authorized GPU:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

Actual query output:

```text
GPU-43c2f925-db05-c655-bb46-25196d0627d6, 770431, 1478 MiB
```
