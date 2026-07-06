# ITM Experiment Plan

> Implementation now follows a gated baseline reset: official MDM must serve as
> Text-only, and the flexible SMPL IMU-only model must pass its quality gate
> before formal Text+IMU training. The earlier regression Transformer is a
> lower-bound sanity check only.

## Positioning

ITM should be framed as **IMU-guided text-to-motion generation**, not as another text-assisted sparse-IMU reconstruction method. Text controls semantic intent; sparse IMU controls physical and temporal details. This avoids directly colliding with Spatial-Related Sensors Matters and Ego4o.

## Phase 0: Asset and Reproducibility Audit

- Run `scripts/audit_assets.py --config configs/paths.toml`.
- Confirm local availability of AMASS, SMPL, HumanML3D text/splits, and related code.
- Treat Ego4o as an architecture/config reference only; do not spend project time forcing full reproduction unless a specific baseline comparison needs it.
- Do not download large datasets or checkpoints without confirming storage location and expected size.

## Phase 1: Minimal Data Pipeline

- Build `motion + text` examples from HumanML3D/BABEL-compatible data.
- Generate synthetic sparse IMU from motion using acceleration and orientation proxies.
- Store derived artifacts outside the repo, with configurable paths and symlinks into the workspace only when needed.
- Start with 6 IMU positions for compatibility with TransPose/Ego4o-style setups, then ablate 1-3 IMU settings.
- Use `/home/a200/0proj/datasets/all` as the first HumanML3D-style source: `texts`, `new_joints`, `new_joint_vecs`, and `train/val/test.txt`.

## Phase 2: Model Prototype

- Use an existing text-to-motion backbone where possible.
- Add an IMU temporal encoder that consumes acceleration and 6D orientation features.
- Fuse text and IMU conditions before or inside the motion generator.
- Train with:
  - motion reconstruction/generation loss
  - text-motion alignment objective from the chosen backbone
  - IMU consistency loss computed by regenerating IMU-like signals from output motion

## Phase 3: Evaluation

- Text-to-motion quality: FID, R-Precision, MM-Dist, Diversity, MModality.
- Physical consistency: generated-vs-input IMU acceleration and orientation errors.
- Pose accuracy when ground truth exists: MPJPE, PA-MPJPE, jitter.
- Control tests:
  - same text, different IMU
  - same IMU, different text
  - ambiguous actions such as sit, stand, squat, crouch, and transition

## Baselines

- Text-only: MotionDiffuse, MotionGPT, MotionLab.
- IMU-only: DIP, TransPose, IMUPoser-style models.
- Multimodal reference: Ego4o paper/code ideas, with direct numerical comparison only if its local dependencies become available.
- Simple fusion: concatenate text and IMU features into a transformer as a lower-bound multimodal baseline.

## Acceptance Criteria for a Paper-Grade V1

- Text+IMU improves over text-only on IMU consistency without collapsing text-motion quality.
- Text+IMU improves over IMU-only in ambiguous/static-posture cases.
- The method supports at least 1, 3, and 6 IMU configurations.
- Qualitative demos clearly show physical control: replacing IMU changes motion details under the same text.
## Evaluation Reframing

ITM is an IMU-controllable **text-to-motion generation** project, not primarily
an inertial pose-estimation project. The paper-level evaluation therefore uses
three complementary axes:

1. Text alignment: official HumanML3D Matching Score and R-precision.
2. Motion generation: official HumanML3D FID, Diversity, and Multimodality.
3. IMU control: acceleration/orientation consistency between the conditioning
   signal and IMU re-synthesized from generated motion, reported per sensor and
   under sensor noise/dropout.

MPJPE and paired 263D reconstruction errors remain supplementary diagnostics.
They are useful when paired ground truth exists, but must not be the headline
objective because they reward one deterministic answer and do not measure
semantic quality or sample diversity. Multimodality is only meaningful after a
stochastic generator is implemented.

The primary comparisons should include text-only generation, IMU-only control,
and text+IMU generation with independently adjustable condition guidance. The
same protocol should cover all six sensors, wrists only, and head only. Report
the text-quality/control-adherence trade-off rather than collapsing it into one
custom score.
