#!/usr/bin/env bash
# Historical command for the one authorized attempt. Do not rerun: directory and authorization are consumed.
OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_b0_bounded.py \
  --plan docs/itm_rework/protocol/canonical_method_acceptance_20260925_v1/execution_plan/execution_plan.json \
  --run-dir /home/a200/0proj/ITM/outputs/itm_rework/method/canonical_b0_authorized_20260926_v1 \
  --execute \
  --authorization-record docs/itm_rework/method/B0_AUTHORIZATION_20260926.json \
  --gpu-uuid GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004
