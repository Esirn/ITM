# Baseline Reset Progress - 2026-07-02

## Status

The previous joint-vector regression Transformer remains a data-flow lower
bound. New experiments use pretrained MDM for text generation and SMPL
rotations for IMU-only pose estimation.

## Text-only MDM

- Official checkpoint: `humanml_trans_enc_512/model000475000.pt`
- Text encoder: frozen OpenAI CLIP ViT-B/32 from the official checkpoint
- Compatibility wrapper: `scripts/run_mdm.py`
- Maintained renderer: `scripts/render_mdm_results.py`
- Smoke artifact: `outputs/mdm/text_only_smoke/comparison.gif`

Three 1000-step samples were generated for “a person walks forward, turns left,
and sits down”. The official checkpoint's bundled 20-rep reference log reports
FID `0.5443`, R-precision top-1/2/3 `0.3195/0.4978/0.6110`, and Diversity
`9.5595`. A new 5-rep official debug evaluation is implemented but not launched
because the upstream script estimates about three GPU-hours.

## Standard IMU Bridge

- HumanML IDs map back to AMASS through `index.csv`.
- Standard cache schema stores six slots of 3D acceleration and 3x3 global
  orientation, SMPL local rotations, joints, translation, shape, FPS, and
  provenance.
- Translation uses linear interpolation; rotations use per-joint Slerp,
  including non-60/120 FPS AMASS sources.
- 30 FPS caches support IMUPoser; 20 FPS caches align with MDM control training.

The full training cache contains 7009 aligned HumanML/AMASS records at 30 FPS.
MDM control resamples acceleration linearly and orientation with Slerp to 20
FPS at load time. Full aligned validation and test caches contain 434 and 1333
records respectively.

## Flexible IMUPoser

The baseline matches the core IMUPoser architecture: fixed 60D five-device
input, sensor masking, two-layer 512-wide bidirectional LSTM, and 24 SMPL joint
rotations in 6D. Head-only and wrists share one checkpoint. Pose velocity and
acceleration losses provide initial temporal regularization.

Medium-data run (604 train / 115 validation):

| configuration | MPJPE | rotation error | jerk ratio | foot skating |
| --- | ---: | ---: | ---: | ---: |
| head | 14.75 cm | 0.275 rad | 0.314 | 0.142 m/s |
| wrists | 10.59 cm | 0.233 rad | 0.408 | 0.324 m/s |

Wrists pass the 12 cm gate; head-only remains above it. Low jerk indicates
over-smoothing, not necessarily superior reconstruction.

## MDM IMU Control

Implemented and exercised as a training-set qualitative pilot:

- six-slot 12D temporal IMU encoder with sensor IDs and masks
- independent IMU condition dropout
- zero-initialized residual adapters at every MDM Transformer layer
- independent text and incremental IMU classifier-free guidance helper
- forward hook that preserves the official MDM diffusion call signature
- stage-1 trainer that freezes MDM/CLIP and saves only control weights
- acceleration normalization statistics stored in each checkpoint
- shared-noise counterfactual sampling and per-panel IMU trace rendering

A five-epoch run over all 7009 aligned training motions and both head/wrists
sensor masks completed on GPU 1. After fixing strict post-encoder condition
dropout, losses were `0.07791`, `0.08080`, `0.07893`, `0.07619`, and `0.07642`;
the final checkpoint is `outputs/mdm_control/stage1_full_pilot_v2.pt`.

Shared-noise pilot comparisons are under `outputs/mdm_control/qualitative/`:

- `same_text_different_imu`: vague walking text with slow, brisk, and limping IMU
- `same_imu_different_text`: fixed head IMU with disabled, vague, and arm-swing text
- `four_way`: GT, official MDM, Flexible IMUPoser, and ITM

For the selected training examples, changing head IMU produced 15.9--20.1 cm
mean root-relative joint differences. With fixed head IMU, a HumanML-style
arm-swing caption at text scale 2.5 increased the wrist-relative-to-shoulder
motion proxy from 0.155 m to 0.298 m (about 92%) relative to vague walking text.
Increasing text scale to 4.0 reduced it to 0.110 m, so stronger guidance is not
monotonic. These are mechanism checks on seen samples, not paper results. Validation
and test caches, multiple seeds, guidance sweeps, and formal metrics remain
required.

An initial unseen-test pilot was also generated. Changing IMU under the same
vague walking text still produced 9.0--20.1 cm mean root-relative differences,
with slow versus quick/struggling examples showing distinct root travel.
However, the fixed-head-IMU arm-swing prompt did not generalize: the motion
proxy decreased from 0.199 m for vague walking to 0.100 m. The first
complementarity direction is therefore promising; the text-completes-missing-
body direction remains unproven and needs broader caption/seed evaluation or a
stronger text-control training objective.
