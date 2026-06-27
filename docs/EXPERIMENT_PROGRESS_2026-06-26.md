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

PyTorch is now installed in the `itm` environment and verified:

- `torch 2.5.1`
- CUDA available: `True`
- CUDA runtime: `12.1`
- GPU count: `2`

The initial install hit `undefined symbol: iJIT_NotifyEvent`; downgrading
`mkl` and `intel-openmp` below 2025 fixed it.

## Neural Baseline Smoke

A CPU smoke run completed:

```bash
conda run -n itm python scripts/train_torch_frame_baseline.py \
  --manifest outputs/manifests/train.jsonl \
  --imu-cache-manifest outputs/manifests/train_imu_cache.jsonl \
  --eval-manifest outputs/manifests/val.jsonl \
  --eval-imu-cache-manifest outputs/manifests/val_imu_cache.jsonl \
  --max-records 20 \
  --eval-max-records 10 \
  --epochs 2 \
  --hidden-dim 64 \
  --num-layers 1 \
  --device cpu
```

Result:

- train MSE: `0.224086`
- eval MSE: `0.223038`

This only verifies the training loop. Use `--device cuda:0` after confirming GPU
0 has enough free memory. Do not use GPU 1 while PID `2017173` is still running
there.

## Neural Baseline GPU Run - 2026-06-27

The frame-level MLP was trained on all 999 training records and evaluated on
all 200 validation records. All variants used two 256-wide hidden layers, 30
epochs, batch size 1024, and `cuda:0`.

| variant | train_mse | eval_mse | eval_mae |
| --- | ---: | ---: | ---: |
| imu_only | 0.009864 | 0.014230 | 0.061955 |
| full | 0.006732 | 0.020078 | 0.078787 |
| text_only | 0.014815 | 0.081496 | 0.154478 |

The IMU-only MLP improves validation MSE over the linear IMU-only baseline from
`0.024062` to `0.014230`. Adding hashed text features lowers training error but
worsens validation error, while text-only generalization is poor. This is
evidence that the current hash representation memorizes training captions
rather than providing reusable language semantics; it is not evidence that
text guidance is intrinsically unhelpful.

Next steps:

1. Replace hash features with a frozen pretrained text encoder and cache the
   caption embeddings.
2. Replace independent frame regression with a temporal sequence model and
   variable-length masking.
3. Evaluate checkpoints on the held-out test split only after model selection
   on validation data.
4. Add sensor-count and sensor-location ablations after the temporal baseline
   is stable.

## Semantic Temporal Baseline - 2026-06-27

Implemented:

- offline frozen DistilBERT caption embedding caches (768 dimensions)
- a masked Transformer encoder for variable-length IMU sequences
- broadcast text conditioning, sinusoidal positions, and padded-loss masking
- validation-best checkpoint selection and independent held-out evaluation

Both models used four Transformer layers, model dimension 256, eight attention
heads, and 30 epochs on the 999/200 train/validation split.

| variant | best epoch | val MSE | test MSE | test MAE |
| --- | ---: | ---: | ---: | ---: |
| temporal IMU-only | 29 | 0.017614 | 0.018271 | 0.075857 |
| temporal DistilBERT + IMU | 29 | 0.017996 | 0.018774 | 0.078116 |

The pretrained text representation removes most of the large generalization
failure observed with hashed text, but still does not improve over the matched
IMU-only temporal model. The validation and held-out test rankings agree. The
current task is deterministic reconstruction from synthetic IMU, so frame MSE
rewards copying the strong physical signal and gives text little opportunity to
resolve ambiguity.

The next model should therefore create a deliberately underdetermined setting
(fewer sensors, masked IMU spans, or noisy sensors) and use a generative motion
objective. Sensor-count/location ablations should be established before adding
a diffusion or masked-token motion decoder.
