# ITM Experiment Progress - 2026-06-26

> The project route was reset on 2026-07-02 after qualitative review of the
> regression baseline. Current MDM, standard-IMU, IMUPoser, and control-adapter
> status is recorded in `BASELINE_RESET_PROGRESS_2026-07-02.md`.

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

## Sparse Sensor Ablation - 2026-06-27

The temporal pipeline now selects sensor subsets by cache slot and records the
selection in each checkpoint. Cache slot order is pelvis, left ankle, right
ankle, head, left wrist, and right wrist. The following matched text+IMU and
IMU-only runs use the same architecture and seed as the six-sensor experiment.

| sensors | text | val MSE | test MSE | test MAE |
| --- | --- | ---: | ---: | ---: |
| pelvis (1) | no | 0.030690 | 0.031836 | 0.099072 |
| pelvis (1) | yes | 0.030185 | 0.032504 | 0.099689 |
| head (1) | no | 0.028923 | 0.030411 | 0.097918 |
| head (1) | yes | 0.028442 | 0.030593 | 0.098537 |
| wrists (2) | no | 0.029189 | 0.030797 | 0.098079 |
| wrists (2) | yes | 0.028406 | 0.029610 | 0.096633 |
| pelvis + wrists (3) | no | 0.023170 | 0.023793 | 0.087235 |
| pelvis + wrists (3) | yes | 0.022950 | 0.023650 | 0.088568 |
| all (6) | no | 0.017614 | 0.018271 | 0.075857 |
| all (6) | yes | 0.017996 | 0.018774 | 0.078116 |

The two-wrist setting is the only current configuration where text improves
both MSE and MAE on validation and test. The three-sensor test MSE improves by
only about 0.6%, while MAE worsens. The pelvis-only validation improvement does
not reproduce on test, and six sensors favor IMU-only. This suggests sensor
location and ambiguity matter more than sensor count alone.

Like pelvis-only, head-only shows a small text gain on validation (about 1.7%)
that does not reproduce on test: test MSE worsens by about 0.6%, and MAE also
worsens. Under the current single-seed protocol, there is no evidence that the
head-only setting benefits from text conditioning.

These are single-seed exploratory results. They are not yet paper-grade
evidence: matched runs need at least three seeds, confidence intervals, and
semantic/physical generation metrics beyond joint-vector reconstruction error.

## CLIP Baseline and Qualitative Export - 2026-07-02

Implemented:

- offline CLIP ViT-L/14 text caches using the standard EOS-pooled text feature
- exact NumPy recovery from raw HumanML3D 263D features to 22 global joints
- synchronized GT / text-only / IMU-only / text+IMU animation export
- synchronized acceleration and orientation traces for one named IMU location
- unbiased qualitative selection containing text-helped, text-hurt, and seeded
  random examples

The recovery implementation was checked against `new_joints` on a real test
sample (MAE `2.25e-9`, maximum error `5.96e-8`). The matched two-wrist CLIP run
used the same architecture, seed, and training protocol as DistilBERT.

| condition | val MSE | test MSE | test MAE |
| --- | ---: | ---: | ---: |
| wrists IMU only | 0.029189 | 0.030797 | 0.098079 |
| DistilBERT + wrists IMU | 0.028406 | 0.029610 | 0.096633 |
| CLIP ViT-L/14 + wrists IMU | 0.031065 | 0.033875 | 0.104977 |
| DistilBERT text only | 0.052178 | 0.056288 | 0.130745 |
| CLIP ViT-L/14 text only | 0.054305 | 0.057097 | 0.136571 |

CLIP is the field-standard text encoder and should remain a named baseline, but
it is weaker under the current broadcast-add fusion. Per-sample results contain
both large improvements and regressions, so the failure is not evidence that
CLIP lacks motion semantics. It instead shows that deterministic 263D MSE and a
single broadcast text vector are a poor final formulation for controllable
generation.

Qualitative artifacts are in:

- `outputs/visualizations/wrists_distilbert/`
- `outputs/visualizations/wrists_clip_vitl14/`
- `outputs/visualizations/wrists_clip_vitl14_four_way/`

The temporal text-only baseline receives the ground-truth sequence length and
normalized time coordinates. It is therefore a conditional reconstruction
diagnostic, not a free-length text-to-motion generator. The future diffusion
baseline must replace it for paper-level text-only comparisons.

The two-wrist animations identify both conditioned sensors and display the left
wrist trace by default (`slot 4`, HumanML joint `20`). Acceleration is the
synthetic second-difference proxy in `position/frame^2`, not calibrated physical
`m/s^2`; orientation is the synthetic limb-direction unit vector. Both traces
use a moving time cursor synchronized with the skeleton animation.

## Head IMU Qualitative Export - 2026-07-02

The existing DistilBERT head-only checkpoints were exported as synchronized
four-way comparisons: ground truth, text only, Head IMU only, and text + Head
IMU. The public animation labels use `Head IMU`; internal provenance remains in
`index.json` as cache slot `3`, HumanML joint `15`. Acceleration and head
orientation traces are synchronized below the skeleton panels.

Artifacts are in `outputs/visualizations/head_distilbert_four_way/`, containing
one text-helped, one text-hurt, and one seeded-random test example. The strongest
per-sample improvement over Head-IMU-only is `0.062434` MSE, while the strongest
regression is `-0.055020` MSE. These selected cases illustrate behavior and are
not aggregate performance claims.

The matched CLIP ViT-L/14 + head-IMU run reached validation MSE `0.030225` and
test MSE/MAE `0.032854` / `0.104750`. For comparison, head IMU-only reached
test MSE `0.030411`, while DistilBERT + head IMU reached `0.030593`. CLIP is
therefore not better under the current broadcast-add reconstruction model.

CLIP head artifacts are in
`outputs/visualizations/head_clip_vitl14_four_way/`. The selected text-helped
case improves over head IMU-only by `0.024623` MSE; the selected text-hurt case
regresses by `0.037213` MSE. The plots show head acceleration and orientation
for cache slot `3`, HumanML joint `15`.

Going forward, frozen CLIP is the primary text encoder because it is the more
standard text-to-motion baseline and will match the planned diffusion backbone.
Existing DistilBERT results remain as an encoder ablation; new sensor/model
experiments do not need corresponding DistilBERT runs unless a reviewer-facing
ablation later requires them.

The current temporal models are reconstruction sanity checks, not the paper's
final generative model. The next core baseline should use a pretrained
text-to-motion diffusion model with a separate temporal IMU control branch.
