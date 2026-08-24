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
- `scripts/cache_text_embeddings.py`: frozen Transformer caption embedding cache
- `scripts/train_torch_temporal_baseline.py`: masked temporal text+IMU baseline
- `scripts/evaluate_torch_temporal_baseline.py`: held-out checkpoint evaluation
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

Cache semantic text features and train the temporal baseline:

```bash
conda run -n itm python scripts/cache_text_embeddings.py \
  --manifest outputs/manifests/train.jsonl \
  --output outputs/text_embeddings/train_distilbert.npz
conda run -n itm python scripts/cache_text_embeddings.py \
  --manifest outputs/manifests/val.jsonl \
  --output outputs/text_embeddings/val_distilbert.npz
conda run -n itm python scripts/train_torch_temporal_baseline.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/train_imu_cache.jsonl \
  --text-cache outputs/text_embeddings/train_distilbert.npz \
  --eval-manifest outputs/manifests/val.jsonl \
  --eval-imu-cache-manifest outputs/manifests/val_imu_cache.jsonl \
  --eval-text-cache outputs/text_embeddings/val_distilbert.npz \
  --device cuda:0
```

Text model loading is offline by default. Pass `--allow-download` to the cache
script only when the requested Hugging Face model is not already local.

Use the local CLIP ViT-L/14 text encoder for the domain-standard baseline:

```bash
conda run -n itm python scripts/cache_text_embeddings.py \
  --manifest outputs/manifests/train.jsonl \
  --model /home/a200/0proj/MotionLab/checkpoints/clip-vit-large-patch14 \
  --encoder clip \
  --output outputs/text_embeddings/train_clip_vitl14.npz \
  --device cuda:0
```

Export qualitative GT / text-only / IMU-only / text+IMU comparisons from
matched temporal checkpoints. The script selects examples where fusion helps
most, hurts most, and a seeded random subset; it writes animations and raw
prediction arrays:

```bash
conda run -n itm python scripts/visualize_temporal_comparison.py \
  --text-checkpoint outputs/neural/temporal_clip_vitl14_text_only_gpu0.pt \
  --imu-checkpoint outputs/neural/temporal_imu_only_wrists_gpu0.pt \
  --conditioned-checkpoint outputs/neural/temporal_clip_vitl14_wrists_gpu0.pt \
  --manifest outputs/manifests/test.jsonl \
  --imu-cache-manifest outputs/manifests/test_imu_cache.jsonl \
  --text-cache outputs/text_embeddings/test_clip_vitl14.npz \
  --output-dir outputs/visualizations/wrists_clip_vitl14_four_way \
  --device cuda:0
```

The animation also plots one synchronized IMU trace. It defaults to the first
sensor used by the conditioned checkpoint (left wrist for the two-wrist model).
Use `--imu-display-slot 5` to show the right wrist instead. The title records
all conditioned sensors plus the displayed cache slot and HumanML joint index.

Generate the single-head-IMU comparison with human-readable panel labels:

```bash
conda run -n itm python scripts/visualize_temporal_comparison.py \
  --text-checkpoint outputs/neural/temporal_clip_vitl14_text_only_gpu0.pt \
  --imu-checkpoint outputs/neural/temporal_imu_only_head_gpu0.pt \
  --conditioned-checkpoint outputs/neural/temporal_clip_vitl14_head_gpu0.pt \
  --manifest outputs/manifests/test.jsonl \
  --imu-cache-manifest outputs/manifests/test_imu_cache.jsonl \
  --text-cache outputs/text_embeddings/test_clip_vitl14.npz \
  --output-dir outputs/visualizations/head_clip_vitl14_four_way \
  --imu-display-slot 3 \
  --device cuda:0
```

New experiments use frozen CLIP as the primary text encoder. Existing
DistilBERT runs are retained as an encoder ablation but are not expanded to new
sensor configurations.

Select cached sensors by zero-based slot for sparse-IMU ablations. The default
cache order is pelvis, left ankle, right ankle, head, left wrist, right wrist:

```bash
# Pelvis and both wrists
conda run -n itm python scripts/train_torch_temporal_baseline.py \
  ... \
  --sensor-slots 0,4,5
```

The sensor slots are stored in the checkpoint and automatically restored by
`evaluate_torch_temporal_baseline.py`.

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

## Generative Baselines

Build an SMPL-based standard IMU cache (`30` FPS for IMUPoser, `20` FPS for
MDM control):

```bash
conda run -n itm python scripts/build_standard_imu_cache.py \
  --manifest outputs/manifests/train.jsonl \
  --index /home/a200/mount/a40/relatedworks/mdm/HumanML3D/index.csv \
  --amass-root /home/a200/0proj/datasets/AMASS \
  --cache-manifest outputs/manifests/train_standard_imu.jsonl \
  --device cuda:0
```

Run and render the official MDM checkpoint without editing its repository:

```bash
conda run -n itm python scripts/run_mdm.py sample \
  --model_path outputs/mdm/checkpoints/humanml_trans_enc_512/model000475000.pt \
  --text_prompt "a person walks forward, turns left, and sits down" \
  --motion_length 6 --num_repetitions 3 --guidance_param 2.5 --device 0 \
  --output_dir outputs/mdm/text_only
conda run -n itm python scripts/render_mdm_results.py \
  --results outputs/mdm/text_only/results.npy \
  --output outputs/mdm/text_only/comparison.gif
```

Reproduce MotionLab MotionFlow Text-to-Motion in its isolated `rfmotion`
environment without loading its renderer or unrelated task encoders:

```bash
conda run --no-capture-output -n rfmotion python scripts/sample_motionlab_text.py \
  --motionlab-root /home/a200/mount/a40/relatedworks/MotionLab \
  --checkpoint /home/a200/mount/a40/relatedworks/MotionLab/checkpoints/motionflow/motionflow.ckpt \
  --request /tmp/motionlab_request.json \
  --output outputs/motionlab/text_only/results.npz \
  --metadata outputs/motionlab/text_only/metadata.json \
  --device cuda:1
conda run --no-capture-output -n itm python scripts/render_motionlab_results.py \
  --results outputs/motionlab/text_only/results.npz \
  --metadata outputs/motionlab/text_only/metadata.json \
  --output outputs/motionlab/text_only/motion.gif
```

The request JSON contains `text`, `lengths`, and `seed`. See
`docs/MOTIONLAB_MIGRATION.md` for the reproduced path and the distinction
between MotionLab trajectory hints and raw IMU control.

Train and evaluate the flexible IMUPoser baseline:

```bash
conda run -n itm python scripts/train_flexible_imu_poser.py \
  --train-cache-manifest outputs/manifests/train_standard_imu.jsonl \
  --eval-cache-manifest outputs/manifests/val_standard_imu.jsonl \
  --sensor-configs head,wrists --device cuda:0
conda run -n itm python scripts/evaluate_flexible_imu_poser.py \
  --checkpoint outputs/imu_poser/flexible_imu_poser.pt \
  --cache-manifest outputs/manifests/val_standard_imu.jsonl \
  --output outputs/imu_poser/eval.json --device cuda:0
```

Train the frozen-MDM IMU adapters on GPU 1 (change `--device` after checking
current utilization):

```bash
conda run -n itm python scripts/train_mdm_imu_control.py \
  --mdm-checkpoint /home/a200/mount/a40/relatedworks/mdm/motion-diffusion-model/save/humanml_trans_enc_512/model000475000.pt \
  --mdm-args /home/a200/mount/a40/relatedworks/mdm/motion-diffusion-model/save/humanml_trans_enc_512/args.json \
  --manifest outputs/manifests_full/train.jsonl \
  --standard-imu-manifest outputs/manifests_full/train_standard_imu.jsonl \
  --sensor-configs head,wrists --epochs 5 --batch-size 16 \
  --device cuda:1 --output outputs/mdm_control/stage1_full_pilot_v2.pt
```

Counterfactual sampling uses a JSON case list and shared diffusion noise:

```bash
conda run -n itm python scripts/sample_mdm_imu_control.py \
  --control-checkpoint outputs/mdm_control/stage1_full_pilot_v2.pt \
  --mdm-args /home/a200/mount/a40/relatedworks/mdm/motion-diffusion-model/save/humanml_trans_enc_512/args.json \
  --standard-imu-manifest outputs/manifests_full/train_standard_imu.jsonl \
  --spec outputs/mdm_control/qualitative/same_text_different_imu/head_spec.json \
  --output outputs/mdm_control/qualitative/same_text_different_imu/head_results.npz \
  --seed 1234 --text-scale 2.5 --imu-scale 1.0 --device cuda:1
```

Render with `scripts/render_mdm_imu_control.py`; use
`scripts/render_mdm_imu_four_way.py` for GT / MDM / IMUPoser / ITM. See
`docs/BASELINE_RESET_PROGRESS_2026-07-02.md` for pilot results and limitations.

## Interactive Demo

Start the local/LAN condition lab:

```bash
conda run --no-capture-output -n itm uvicorn itm.demo.app:app \
  --host 0.0.0.0 --port 8001
```

Open `http://localhost:8001` on this machine or
`http://<server-lan-ip>:8001` from another machine. The demo supports a
GT/Text-only/IMU-only/Text+IMU comparison and the eight non-empty combinations
of Text `{None,A,B}` × IMU `{None,A,B}`. Runs are cached under
`outputs/demo_runs/<run_id>/` and can be reopened by run ID without inference.
