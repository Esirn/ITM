# ITM Experiment Progress - 2026-06-26

## Implemented

- Initialized the repository and committed the phase-1 scaffold.
- Built HumanML3D-style manifest utilities and synthetic sparse-IMU cache tools.
- Added numpy dataset/collator for `text + joint_vec + joints + IMU`.
- Added IMU consistency metrics for acceleration and orientation.
- Added CPU linear reconstruction baselines with ablations:
  - full
  - text only
  - IMU only
  - acceleration only
  - orientation only
  - time only
- Added split-suite and result-summary scripts.
- Added an optional PyTorch frame-level MLP baseline script.

## Data Status

Current medium-scale smoke split:

- `train`: requested 1000, built 999 valid records
- `val`: requested 200, built 200 records
- `test`: requested 200, built 200 records

One invalid training record was filtered:

- `000990`: joints shape `(22, 3)`, expected `(T, J, 3)`

The filtered manifests are aligned with split-specific cache manifests:

- `outputs/manifests/train.jsonl`
- `outputs/manifests/train_imu_cache.jsonl`
- `outputs/manifests/val.jsonl`
- `outputs/manifests/val_imu_cache.jsonl`
- `outputs/manifests/test.jsonl`
- `outputs/manifests/test_imu_cache.jsonl`

## Linear Baseline Results

Train 999, evaluate val 200:

| variant | eval_mse | eval_mae |
| --- | ---: | ---: |
| imu_only | 0.024062 | 0.080926 |
| orientation_only | 0.024486 | 0.081705 |
| full | 0.025112 | 0.085994 |
| text_only | 0.057667 | 0.132711 |
| acceleration_only | 0.057778 | 0.127983 |
| time_only | 0.058600 | 0.129147 |

Train 999, evaluate test 200:

| variant | eval_mse | eval_mae |
| --- | ---: | ---: |
| imu_only | 0.024277 | 0.082120 |
| orientation_only | 0.024697 | 0.082876 |
| full | 0.025556 | 0.087224 |
| acceleration_only | 0.060430 | 0.131076 |
| time_only | 0.061277 | 0.132269 |
| text_only | 0.062518 | 0.137824 |

Interpretation: synthetic IMU is generated directly from ground-truth joints, so
IMU-heavy baselines are expected to be strong. Text hash features do not help
the linear model yet, which makes the next neural fusion model important.

## Torch Status

PyTorch is not currently installed in the `itm` environment.

Attempts to install from the official PyTorch CUDA 12.1 and CPU wheel indexes
stalled without progress and were interrupted to avoid leaving a hanging
install process. The optional neural baseline code is present and syntax-checked,
but cannot be executed until `torch` is installed.

## Next Recommended Step

Install PyTorch through a reliable local mirror or conda package source, then
run:

```bash
conda run -n itm python scripts/train_torch_frame_baseline.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/train_imu_cache.jsonl \
  --eval-manifest outputs/manifests/val.jsonl \
  --eval-imu-cache-manifest outputs/manifests/val_imu_cache.jsonl \
  --max-records 100 \
  --eval-max-records 50 \
  --device cpu
```

Use `--device cuda:0` after confirming GPU 0 has enough free memory. Do not use
GPU 1 while PID `2017173` is still running there.
