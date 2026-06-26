# ITM

ITM explores **IMU-guided text-to-motion**: sparse IMU readings provide physical control signals while text supplies action semantics for full-body motion generation.

## Current Scope

This repository now contains the reproducibility and experiment scaffold for the first phase:

- local asset audit for datasets, related code, and Ego4o reproducibility risks
- path configuration kept outside source code
- a minimal IMU consistency metric module for generated-motion evaluation
- an experiment plan focused on differentiating ITM from Ego4o and text-assisted IMU reconstruction work

## Repository Layout

- `configs/paths.toml`: local path registry for datasets and related work
- `scripts/audit_assets.py`: non-mutating audit of available assets and hard-coded Ego4o paths
- `scripts/inspect_batch.py`: sanity check for padded text-motion-IMU batches
- `scripts/train_linear_baseline.py`: CPU ridge-regression text+IMU baseline
- `src/itm/baselines/linear_reconstruct.py`: linear reconstruction baseline utilities
- `src/itm/data/dataset.py`: numpy dataset and collator for manifest/cache records
- `src/itm/data/synthetic_imu.py`: minimal joint-to-IMU proxy extraction
- `src/itm/metrics/imu_consistency.py`: acceleration/orientation consistency metrics
- `docs/EXPERIMENT_PLAN.md`: implementation and evaluation roadmap
- `docs/EXPERIMENT_PROGRESS_2026-06-26.md`: latest local experiment status
- `docs/EGO4O_REPRO_AUDIT.md`: current Ego4o reproducibility assessment
- `docs/ENVIRONMENT.md`: conda environment notes

## Usage

Run the local audit:

```bash
python scripts/audit_assets.py --config configs/paths.toml
```

Build and inspect a small manifest:

```bash
conda run -n itm python scripts/build_manifest.py --split train --limit 10
conda run -n itm python scripts/inspect_sample.py --manifest outputs/manifests/train.jsonl --index 0
```

Build reusable synthetic IMU cache files from that manifest:

```bash
conda run -n itm python scripts/build_imu_cache.py --manifest outputs/manifests/train.jsonl
```

Build manifests and split-specific IMU caches together:

```bash
conda run -n itm python scripts/build_split_suite.py \
  --train-limit 1000 \
  --val-limit 200 \
  --test-limit 200
```

`build_split_suite.py` filters invalid joint arrays by default and keeps each
manifest aligned with its split-specific IMU cache. Use `--strict-invalid` to
fail fast instead.

Inspect a padded batch built from manifest and cache records:

```bash
conda run -n itm python scripts/inspect_batch.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/imu_cache.jsonl \
  --batch-size 2
```

Train a CPU smoke baseline from text and synthetic IMU to HumanML3D joint vectors:

```bash
conda run -n itm python scripts/train_linear_baseline.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/imu_cache.jsonl \
  --max-records 10
```

For a held-out smoke check, build a validation manifest/cache separately:

```bash
conda run -n itm python scripts/build_manifest.py --split val --limit 10
conda run -n itm python scripts/build_imu_cache.py \
  --manifest outputs/manifests/val.jsonl \
  --cache-manifest outputs/manifests/val_imu_cache.jsonl
conda run -n itm python scripts/train_linear_baseline.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/imu_cache.jsonl \
  --eval-manifest outputs/manifests/val.jsonl \
  --eval-imu-cache-manifest outputs/manifests/val_imu_cache.jsonl \
  --max-records 10 \
  --eval-max-records 10
```

Use `--no-text`, `--no-acceleration`, and `--no-orientation` for quick ablations.

Run a compact ablation table:

```bash
conda run -n itm python scripts/run_linear_ablation.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/imu_cache.jsonl \
  --eval-manifest outputs/manifests/val.jsonl \
  --eval-imu-cache-manifest outputs/manifests/val_imu_cache.jsonl \
  --max-records 10 \
  --eval-max-records 10
```

Summarize one or more ablation tables:

```bash
conda run -n itm python scripts/summarize_linear_results.py \
  outputs/baselines/linear_ablation/summary.csv
```

Train the optional PyTorch frame-level neural baseline after installing `torch`:

```bash
conda run -n itm python scripts/train_torch_frame_baseline.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/train_imu_cache.jsonl \
  --eval-manifest outputs/manifests/val.jsonl \
  --eval-imu-cache-manifest outputs/manifests/val_imu_cache.jsonl \
  --max-records 100 \
  --eval-max-records 50 \
  --device cuda:0
```

Run the metric smoke test:

```bash
python -m itm.metrics.imu_consistency
```

If `itm` is not installed, set:

```bash
export PYTHONPATH=/home/a200/0proj/ITM/src
```

Run tests:

```bash
conda run -n itm python -m pytest tests
```

## Data Policy

Use symlinks, hard links, or reflinks for large datasets where possible. Do not copy large datasets into this repository. Do not download multi-GB datasets or checkpoints without confirming storage location and expected size first.
