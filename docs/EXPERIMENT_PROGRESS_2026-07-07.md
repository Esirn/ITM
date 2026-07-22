# Experiment Progress - 2026-07-07

## Status

The next-stage counterfactual suite has been implemented and executed on the
test split with `outputs/mdm_control/stage1_full_pilot_v2.pt`.

Completed outputs:

- `outputs/mdm_control/experiments/same_text_different_imu/test_walk_head_wrists/`
- `outputs/mdm_control/experiments/same_head_imu_different_text/test_prompts/`
- `outputs/mdm_control/experiments/guidance_sweep/`
- `outputs/mdm_control/experiments/matrix_test_head/`
- `outputs/mdm_control/experiments/matrix_test_wrists/`
- `outputs/mdm_control/experiments/RESULTS_SUMMARY.md`
- `outputs/mdm_control/experiments/RESULTS_SUMMARY.json`

The suite contains 60 completed runs and 770 generated cases.

## Main Observations

### Same vague text, different IMU

Fixed text:

```text
a person walks
```

This experiment generated 30 test samples with MDM Text-only, ITM Text+head IMU,
and ITM Text+wrist IMUs.

Aggregate results:

| condition group | active sensor error | jerk ratio | arm swing | root travel |
| --- | ---: | ---: | ---: | ---: |
| all | 0.2871 m | 4.2072 | 0.1909 m | 2.1561 m |
| head | 0.1733 m | 4.5210 | 0.2021 m | 2.1826 m |
| wrists | 0.5147 m | 3.5796 | 0.1686 m | 2.1030 m |

The first complementarity direction remains promising: changing IMU under the
same vague text changes travel, step frequency, and full-body motion proxies.
However, jerk is high, so the generated motions need visual inspection and
likely a smoother second training stage.

### Same head IMU, different text

This experiment fixed head-only IMU and varied six walking prompts.

Prompt-level summary:

| prompt | head error | arm swing | jerk | root travel |
| --- | ---: | ---: | ---: | ---: |
| walking with swinging arms | 0.1662 m | 0.2615 m | 6.2430 | 3.3377 m |
| walking quickly | 0.1824 m | 0.2304 m | 5.6510 | 2.4707 m |
| turning while walking | 0.2088 m | 0.2153 m | 4.1204 | 1.4661 m |
| walking | 0.1731 m | 0.2088 m | 5.1645 | 2.4170 m |
| walking slowly | 0.1776 m | 0.1742 m | 4.1510 | 2.1081 m |
| walking with large steps | 0.1948 m | 0.1582 m | 3.7824 | 1.8873 m |

This is better than the earlier single-sample pilot: the arm-swing prompt has
the highest arm-swing proxy while retaining a low head trajectory error. The
effect is not clean enough to claim final success because jerk remains high and
some prompts trade semantic change for unstable motion.

### Guidance sweep

The best practical setting for head-only qualitative inspection is currently:

```text
text_scale = 2.5
imu_scale = 0.5
```

It gives the lowest head active-sensor error in the sweep while keeping jerk
near the low end:

| sensor | text scale | IMU scale | active sensor error | jerk |
| --- | ---: | ---: | ---: | ---: |
| head | 2.5 | 0.5 | 0.2003 m | 1.6134 |
| head | 1.0 | 0.5 | 0.2031 m | 1.6292 |
| head | 4.0 | 0.5 | 0.2008 m | 1.6855 |

For wrist IMUs, `text_scale=2.5, imu_scale=0.5` is a reasonable default:

| sensor | text scale | IMU scale | active sensor error | jerk |
| --- | ---: | ---: | ---: | ---: |
| wrists | 2.5 | 0.5 | 0.4794 m | 1.5036 |
| wrists | 2.5 | 1.0 | 0.4847 m | 1.5066 |
| wrists | 4.0 | 1.0 | 0.5037 m | 1.4209 |

## Useful Matrix Examples

Low-jerk candidates for qualitative inspection:

- `matrix_test_wrists/pair_001_004488_004222`
- `matrix_test_wrists/pair_007_008238_006378`
- `matrix_test_head/pair_001_004488_004222`
- `matrix_test_wrists/pair_011_003437_012883`
- `matrix_test_wrists/pair_017_005033_007286`

High contrast candidates where text/IMU combinations produce larger
root-relative changes:

- `matrix_test_wrists/pair_013_007089_006756`
- `matrix_test_head/pair_013_007089_006756`
- `matrix_test_wrists/pair_004_002572_010157`
- `matrix_test_head/pair_004_002572_010157`
- `matrix_test_wrists/pair_014_009600_010068`

Good head-IMU tracking candidates:

- `matrix_test_head/pair_001_004488_004222`
- `matrix_test_head/pair_002_008803_009184`
- `matrix_test_head/pair_008_002888_002382`
- `matrix_test_head/pair_006_007767_003116`
- `matrix_test_head/pair_018_010233_008755`

## Current Interpretation

The current frozen-MDM adapter is good enough to support a first qualitative
story:

- IMU changes can visibly affect motion under vague text.
- Text changes can affect arm swing and walking style under fixed head IMU.
- The current model is still temporally unstable; jerk is the main weakness.

The next model iteration should not add a new backbone yet. The more direct
next step is to keep the frozen MDM adapter setup and add trajectory/velocity
consistency plus jerk regularization in a second-stage training objective.

## Validation

Code tests after the experiment suite:

```text
37 passed, 5 warnings
```

## Stage-2 Consistency/Smooth Training

Implemented a second-stage training pass on top of
`outputs/mdm_control/stage1_full_pilot_v2.pt`. The model still freezes MDM and
CLIP and only updates the IMU encoder plus residual adapters. The new training
objective adds active-sensor trajectory consistency, active-sensor velocity
consistency, and an excess-jerk regularizer.

Checkpoint outputs:

- `outputs/mdm_control/stage2_consistency_smooth_epoch006.pt`
- `outputs/mdm_control/stage2_consistency_smooth_epoch007.pt`
- `outputs/mdm_control/stage2_consistency_smooth_epoch008.pt`
- `outputs/mdm_control/stage2_consistency_smooth.pt`

Training losses:

| epoch | total | diffusion | trajectory | velocity | jerk |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.203785 | 0.074494 | 0.129022 | 0.001179 | 0.003347 |
| 7 | 0.186302 | 0.076941 | 0.109098 | 0.001148 | 0.003321 |
| 8 | 0.195525 | 0.077188 | 0.118068 | 0.001175 | 0.003319 |

The full test counterfactual suite was rerun with the stage-2 checkpoint:

- `outputs/mdm_control/experiments_stage2/RESULTS_SUMMARY.md`
- `outputs/mdm_control/experiments_stage2/RESULTS_SUMMARY.json`
- `outputs/mdm_control/experiments_stage2/STAGE1_VS_STAGE2.md`

Main comparison against stage-1:

| experiment | active sensor error change | jerk change | arm swing change |
| --- | ---: | ---: | ---: |
| same text, different IMU | -1.4% | -58.1% | -15.0% |
| matrix | -3.4% | -31.0% | -9.6% |
| same head IMU, different text | +13.7% | +19.8% | -23.8% |

Interpretation:

- Stage-2 clearly improves the first core phenomenon: under the same vague text,
  different IMUs still change the generated motion while jerk is much lower.
- Stage-2 also produces smoother matrix examples, useful for qualitative browsing.
- The second core phenomenon is weaker after smoothing: fixed head IMU plus
  different text has higher jerk, worse head active-sensor error, and lower arm
  swing proxy. This should be treated as a failure mode rather than hidden.
- The next model iteration should preserve the stage-2 active-sensor smoothing
  gains while adding a stronger text-preservation or arm-motion objective for
  head-only prompts.

## Stage-2b Balanced Variants

Three lower-weight consistency/smoothness variants were trained from
`stage1_full_pilot_v2.pt` and evaluated with the same 60-run test suite:

| checkpoint | trajectory | velocity | jerk | output root |
| --- | ---: | ---: | ---: | --- |
| `stage2b_balanced_a.pt` | 0.5 | 0.1 | 0.003 | `outputs/mdm_control/experiments_stage2b_balanced_a/` |
| `stage2b_balanced_b.pt` | 1.0 | 0.1 | 0.003 | `outputs/mdm_control/experiments_stage2b_balanced_b/` |
| `stage2b_balanced_c.pt` | 0.5 | 0.2 | 0.001 | `outputs/mdm_control/experiments_stage2b_balanced_c/` |

Each variant produced 60 completed runs and 770 generated cases. The combined
model-selection summary is:

- `outputs/mdm_control/comparisons/itm_stage1_stage2_stage2b/MODEL_SELECTION.md`
- `outputs/mdm_control/comparisons/itm_stage1_stage2_stage2b/MODEL_SELECTION.json`

Key results:

| model | same-text jerk delta | same-head arm swing delta | same-head active error | matrix jerk delta |
| --- | ---: | ---: | ---: | ---: |
| stage2b_balanced_a | -57.6% | -22.1% | 0.2098 m | -27.9% |
| stage2b_balanced_b | -58.2% | -24.0% | 0.2152 m | -33.4% |
| stage2b_balanced_c | -57.9% | -21.6% | 0.2060 m | -27.1% |

No checkpoint passes all four planned gates. The role-based selection is:

- Main checkpoint if a single model is required:
  `outputs/mdm_control/stage2b_balanced_b.pt`.
- Best IMU-control/smoothness checkpoint:
  `outputs/mdm_control/stage2b_balanced_b.pt`.
- Best head-only text-completion diagnostic checkpoint:
  `outputs/mdm_control/stage1_full_pilot_v2.pt`.

Interpretation:

- The stage-2/2b family robustly solves the high-jerk problem for the
  **same-text-different-IMU** phenomenon.
- None of the tested consistency/smoothness weights recover the stage-1
  **same-head-IMU-different-text** arm-swing behavior.
- This is now a clear modeling trade-off rather than noise: direct
  active-sensor consistency helps IMU control but weakens text-driven completion
  of unobserved body parts.
- The next model change should add a text-preservation or upper-body semantic
  objective instead of only retuning the same trajectory/velocity/jerk weights.

## Baseline/Diagnostic Comparisons

The current comparison status is:

| baseline | status | output |
| --- | --- | --- |
| Official MDM Text-only | 5-rep debug evaluator completed; bundled 20-rep reference log available | `outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/eval_humanml_humanml_trans_enc_512_000475000_gscale2.5_debug.log` |
| Flexible IMUPoser | evaluated on test split | `outputs/mdm_control/comparisons/imu_only_imuposer/test_eval.json` |
| Stage-1 ITM | evaluated on test split | `outputs/mdm_control/experiments/` |
| Stage-2 ITM | evaluated on test split | `outputs/mdm_control/experiments_stage2/` |
| Stage-2b ITM | three variants evaluated on test split | `outputs/mdm_control/experiments_stage2b_balanced_{a,b,c}/` |
| Stage-3 ITM | upper-body objective variant evaluated on test split | `outputs/mdm_control/experiments_stage3_upper_body/` |

Flexible IMUPoser test results:

| sensor config | records | MPJPE | rotation error | jerk ratio | foot skating |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 1333 | 14.91 cm | 0.278 rad | 0.294 | 0.123 m/s |
| wrists | 1333 | 10.74 cm | 0.233 rad | 0.455 | 0.308 m/s |

Interpretation:

- Wrists remain a usable IMU-only reconstruction baseline.
- Head-only remains above the 12 cm gate, so it should not be treated as a
  strong solved baseline.
- IMUPoser is useful for showing the distinction between IMU-only reconstruction
  and ITM generation/control, but MPJPE should stay a diagnostic metric rather
  than the main ITM success criterion.
- The official MDM evaluator now runs with the local MDM root. The 5-rep debug
  evaluation completed on GPU 1. Treat it as a fast environment/baseline check;
  the bundled 20-rep reference log remains the stronger paper-level reference
  until we run a full 20-rep evaluation under the current environment.

Official MDM Text-only evaluator results:

| run | reps | matching score | R@1 | R@2 | R@3 | FID | diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current debug | 5 | 3.6625 +/- 0.0615 | 0.4004 | 0.5889 | 0.7068 | 0.4276 +/- 0.0508 | 9.4112 +/- 0.1584 |
| bundled reference | 20 | 5.5659 +/- 0.0270 | 0.3195 | 0.4978 | 0.6110 | 0.5443 +/- 0.0442 | 9.5595 +/- 0.0857 |

The current debug run is numerically reasonable and confirms the evaluator
stack, but it should not be over-interpreted because it uses only 5
replications.

Official MDM evaluator runtime setup:

- `scripts/run_mdm.py` injects a minimal `wandb` stub because upstream MDM
  imports WandB at module import time even when `NoPlatform` is used.
- Local MDM root:
  `/home/a200/0relatedworks/motion-diffusion-model`.
- Symlinked evaluator assets:
  - `glove -> /home/a200/0proj/MotionLab/checkpoints/glove`
  - `t2m -> /home/a200/0proj/MotionLab/checkpoints/t2m`
  - `dataset/HumanML3D -> /home/a200/0proj/datasets/all`
- Completed debug command:

```text
conda run --no-capture-output -n itm python scripts/run_mdm.py evaluate \
  --mdm-root /home/a200/0relatedworks/motion-diffusion-model \
  --model_path /home/a200/0proj/ITM/outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt \
  --eval_mode debug --guidance_param 2.5 --device 1
```

## ITM Generation-Quality Evaluator Bridge

Added a project-side evaluator bridge for ITM:

- `scripts/evaluate_mdm_imu_control_generation.py`

The script samples ITM Text+IMU motions on the IMU-mapped HumanML3D subset and
then feeds the generated HumanML3D 263D motions into the official HumanML
text-motion evaluator wrapper. It computes Matching Score, R-precision, FID and
Diversity without modifying upstream MDM code.

Important protocol note:

- This is not the full official MDM 1000-sample/5-rep or 20-rep protocol.
- It uses only samples with available standard IMU, so it is a diagnostic
  generation-quality check for ITM rather than a final paper table.
- Generated motions are converted from MDM normalization to the T2M evaluator
  normalization before scoring, matching the convention used by MDM's generated
  dataset loader.

The test split contains 682 IMU-mapped samples with usable HumanML3D motion
lengths; with batch size 32, the evaluator scores 672 samples.

Full-available-subset diagnostic results:

| model | sensor | samples | matching score | R@1 | R@2 | R@3 | FID | diversity | GT diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDM Text-only subset | none | 672 | 3.4116 | 0.4464 | 0.6354 | 0.7530 | 0.5280 | 10.0042 | 9.7165 |
| `stage1_full_pilot_v2` | head | 672 | 3.3771 | 0.4449 | 0.6339 | 0.7366 | 0.9648 | 9.3553 | 9.7165 |
| `stage1_full_pilot_v2` | wrists | 672 | 3.3770 | 0.4598 | 0.6473 | 0.7500 | 0.9268 | 9.5862 | 9.7165 |
| `stage2b_balanced_b` | head | 672 | 3.2564 | 0.4539 | 0.6592 | 0.7783 | 0.6076 | 10.0631 | 9.7165 |
| `stage2b_balanced_b` | wrists | 672 | 3.3592 | 0.4435 | 0.6533 | 0.7708 | 0.8540 | 9.8406 | 9.7165 |

Five-rep full-available-subset results:

| model | sensor | samples | reps | matching score | R@1 | R@2 | R@3 | FID | diversity | GT diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MDM Text-only subset | none | 672 | 5 | 3.4152 +/- 0.0196 | 0.4530 +/- 0.0178 | 0.6506 +/- 0.0157 | 0.7586 +/- 0.0116 | 0.5089 +/- 0.0441 | 10.0517 +/- 0.1657 | 9.8342 +/- 0.1421 |
| `stage2b_balanced_b` | head | 672 | 5 | 3.2680 +/- 0.0519 | 0.4580 +/- 0.0131 | 0.6682 +/- 0.0187 | 0.7780 +/- 0.0139 | 0.6526 +/- 0.0521 | 10.1686 +/- 0.1396 | 9.8342 +/- 0.1421 |
| `stage2b_balanced_b` | wrists | 672 | 5 | 3.3546 +/- 0.0378 | 0.4551 +/- 0.0150 | 0.6598 +/- 0.0218 | 0.7673 +/- 0.0103 | 0.8167 +/- 0.0357 | 10.0319 +/- 0.1746 | 9.8342 +/- 0.1421 |

Outputs:

- `outputs/mdm_control/comparisons/itm_generation_eval/mdm_text_only_subset_672.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage1_head_672.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage1_wrists_672.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage2b_head_672.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage2b_wrists_672.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/mdm_text_only_subset_672_5rep.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage2b_head_672_5rep.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage2b_wrists_672_5rep.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/mdm_text_only_subset_320.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage2b_head_320.json`
- `outputs/mdm_control/comparisons/itm_generation_eval/stage2b_wrists_320.json`

Interpretation:

- The evaluator bridge is now functional for ITM outputs.
- On the IMU-mapped subset, Stage-2b keeps text-motion matching and diversity in
  a plausible range relative to the text-only subset baseline.
- Stage-2b head improves R-precision slightly on this subset while increasing
  FID only modestly over the same-subset text-only baseline.
- Stage-2b wrists has a larger FID increase, suggesting stronger IMU control can
  disturb the generated distribution more.
- Stage-1 remains useful for the head-only text-completion qualitative
  phenomenon, but it is worse than Stage-2b on this generation-quality
  diagnostic: lower diversity and substantially higher FID.
- This diagnostic supports using the official evaluator stack for ITM. It is
  now a repeatable 5-rep subset table, but still not the full official
  1000-sample/20-rep protocol because it is restricted to IMU-mapped samples.
- The five-rep run strengthens the same conclusion: head-IMU ITM keeps or
  slightly improves R-precision on the IMU-mapped subset, while wrists-IMU ITM
  shows a clearer FID cost.

## Stage-3 Upper-Body Objective

To address the Stage-2/2b failure mode where active-sensor smoothing weakens
head-only text completion, the training script now supports an optional
head-only upper-body objective:

- `--upper-body-loss-weight`
- `--upper-body-velocity-loss-weight`

The loss is only applied to head-only samples. It supervises the HumanML
upper-body joints `(16, 17, 18, 19, 20, 21)` relative to the root, plus their
relative velocities. The intent is to preserve shoulder/elbow/wrist motion for
unobserved upper-body parts without changing the active-sensor trajectory loss.

The first candidate was trained from `stage1_full_pilot_v2.pt`:

```text
output = outputs/mdm_control/stage3_upper_body.pt
resume = outputs/mdm_control/stage1_full_pilot_v2.pt
trajectory = 0.5
velocity = 0.1
jerk = 0.003
upper_body = 0.25
upper_body_velocity = 0.05
```

Runtime note:

- The `/home/a200/mount/a40/...` sshfs/FUSE MDM root became unreliable during
  smoke training, so subsequent runs used the local MDM root
  `/home/a200/0relatedworks/motion-diffusion-model`.
- The local root needed symlinks to the existing SMPL assets before MDM sampling
  could run.

Checkpoint outputs:

- `outputs/mdm_control/stage3_upper_body_smoke.pt`
- `outputs/mdm_control/stage3_upper_body_epoch006.pt`
- `outputs/mdm_control/stage3_upper_body_epoch007.pt`
- `outputs/mdm_control/stage3_upper_body_epoch008.pt`
- `outputs/mdm_control/stage3_upper_body.pt`

Training losses:

| epoch | total | diffusion | trajectory | velocity | jerk | upper body | upper body velocity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.139364 | 0.074313 | 0.124949 | 0.001186 | 0.003334 | 0.009630 | 0.000794 |
| 7 | 0.132593 | 0.076555 | 0.107076 | 0.001145 | 0.003311 | 0.009340 | 0.000808 |
| 8 | 0.138904 | 0.076729 | 0.119339 | 0.001181 | 0.003319 | 0.009353 | 0.000784 |

The full 60-run test suite completed:

- `outputs/mdm_control/experiments_stage3_upper_body/RESULTS_SUMMARY.md`
- `outputs/mdm_control/experiments_stage3_upper_body/RESULTS_SUMMARY.json`

Key comparison against stage-1:

| model | same-text jerk delta | same-head arm swing delta | same-head active error | matrix jerk delta |
| --- | ---: | ---: | ---: | ---: |
| stage2b_balanced_b | -58.2% | -24.0% | 0.2152 m | -33.4% |
| stage3_upper_body | -57.9% | -22.2% | 0.2059 m | -26.3% |

Interpretation:

- Stage-3 keeps the main same-text-different-IMU smoothing gain.
- The upper-body objective slightly improves head-only arm swing relative to
  `stage2b_balanced_b`, but it is still far below stage-1 and fails the planned
  "arm swing drop within 10%" gate.
- Matrix jerk is worse than `stage2b_balanced_b`, so Stage-3 is not the current
  main checkpoint.
- The current role-based selection remains:
  `stage2b_balanced_b` for the main/IMU-control model and
  `stage1_full_pilot_v2.pt` for head-only text-completion diagnostics.

## Paper Table Cleanup - 2026-07-09

Added a joint-derived active-sensor acceleration consistency proxy to the
counterfactual suite metrics:

- `active_sensor_acceleration_error_mps2`
- `active_sensor_acceleration_ratio`

Implementation note:

- The proxy computes second differences from generated HumanML 22-joint active
  sensor joints and compares them with the target virtual-IMU acceleration.
- It is useful as a control-quality diagnostic, but it is not a full
  generated-IMU orientation metric because ITM currently outputs HumanML3D 263D
  rather than SMPL sensor frames.

Recomputed per-run `metrics.json` and `summary.md` for:

- `outputs/mdm_control/experiments/`
- `outputs/mdm_control/experiments_stage2b_balanced_b/`

Refreshed summaries:

- `outputs/mdm_control/experiments/RESULTS_SUMMARY.json`
- `outputs/mdm_control/experiments/RESULTS_SUMMARY.md`
- `outputs/mdm_control/experiments_stage2b_balanced_b/RESULTS_SUMMARY.json`
- `outputs/mdm_control/experiments_stage2b_balanced_b/RESULTS_SUMMARY.md`
- `outputs/mdm_control/comparisons/itm_stage1_stage2_stage2b/STAGE1_VS_STAGE2B_WITH_ACCEL_PROXY.md`

Key Stage-1 vs Stage-2b changes:

| experiment | active trajectory error | active acceleration error | acceleration ratio | jerk |
| --- | ---: | ---: | ---: | ---: |
| same-text Stage-1 | 0.2871 m | 4.7340 m/s^2 | 3.5541 | 4.2072 |
| same-text Stage-2b | 0.2833 m | 4.5360 m/s^2 | 2.0936 | 1.7599 |
| matrix Stage-1 | 0.3937 m | 5.2004 m/s^2 | 2.5815 | 2.1978 |
| matrix Stage-2b | 0.3835 m | 4.5218 m/s^2 | 1.8963 | 1.4646 |

Interpretation:

- Stage-2b's same-text smoothing gain is supported by both jerk and active
  acceleration proxy: jerk drops 58.2%, acceleration error drops 4.2%, and the
  acceleration magnitude ratio moves closer to the target.
- Matrix runs show the same direction: jerk drops 33.4% and active acceleration
  error drops 13.1%.
- Same-head-IMU-different-text remains a limitation: active acceleration error
  drops, but jerk and arm-swing/text-completion behavior are still worse than
  Stage-1.

Paper-facing table draft:

- `docs/PAPER_EXPERIMENT_TABLES_2026-07-09.md`

## Stage-4 Text-Anchor Objective

Stage-4 tested whether an explicit frozen-MDM text-only anchor can recover
head-only text completion while preserving Stage-2b smoothness.

Training setup:

```text
output = outputs/mdm_control/stage4_text_anchor.pt
resume = outputs/mdm_control/stage1_full_pilot_v2.pt
trajectory = 0.5
velocity = 0.1
jerk = 0.003
text_anchor = 0.2
upper_body_text_anchor = 0.4
```

Implementation note:

- The text anchor uses a frozen MDM Text-only predicted-x0 branch.
- The anchor call removes `imu`, `sensor_mask`, and `imu_frame_mask` from
  `model_kwargs`, rather than using `imu_uncond=True`, because trained adapter
  biases would otherwise still affect the forward pass.
- The anchor losses compare root-relative 22-joint predictions. The general
  text anchor excludes active sensor joints; the upper-body text anchor focuses
  on upper-body joints not occupied by active sensors.

Training losses:

| epoch | total | diffusion | trajectory | velocity | jerk | text anchor | upper-body text anchor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.136448 | 0.074221 | 0.120122 | 0.001167 | 0.003331 | 0.002268 | 0.003965 |
| 7 | 0.131718 | 0.076507 | 0.106387 | 0.001135 | 0.003301 | 0.002116 | 0.003678 |
| 8 | 0.136189 | 0.076729 | 0.115053 | 0.001169 | 0.003311 | 0.002032 | 0.003502 |

The full 60-run test suite completed:

- `outputs/mdm_control/experiments_stage4_text_anchor/RESULTS_SUMMARY.md`
- `outputs/mdm_control/experiments_stage4_text_anchor/RESULTS_SUMMARY.json`
- `outputs/mdm_control/comparisons/itm_stage1_stage2_stage2b/STAGE1_VS_STAGE4_TEXT_ANCHOR.md`

Key comparison:

| model | same-text jerk | same-head active error | same-head arm swing | matrix jerk |
| --- | ---: | ---: | ---: | ---: |
| Stage-1 | 4.2072 | 0.1838 m | 0.2080 m | 2.1978 |
| Stage-2b | 1.7599 | 0.2152 m | 0.1581 m | 1.4646 |
| Stage-3 | 1.7724 | 0.2059 m | 0.1619 m | 1.6196 |
| Stage-4 | 1.8050 | 0.2103 m | 0.1666 m | 1.5497 |

Interpretation:

- Stage-4 keeps the main smoothing gain: same-text jerk is 57.1% below
  Stage-1 and matrix jerk is 29.5% below Stage-1.
- Stage-4 partially recovers head-only arm swing relative to Stage-2b
  (`0.1666 m` vs `0.1581 m`), but it is still 19.9% below Stage-1 and misses
  the planned 10% gate.
- Stage-4 active head error is slightly better than Stage-2b (`0.2103 m` vs
  `0.2152 m`) but still fails the Stage-1 110% gate.
- Stage-4 does not replace Stage-2b as the main checkpoint. It is useful as a
  negative/diagnostic ablation: a simple MDM text-anchor helps a little but
  does not solve the head-only semantic completion weakness.
- Current role-based selection remains:
  `stage2b_balanced_b` for the main/IMU-control model,
  `stage1_full_pilot_v2.pt` for text-completion diagnostics, and
  `stage4_text_anchor.pt` as a text-anchor ablation.

Future stronger directions:

- Stronger Text-to-Motion backbone or baseline: LGTM, MLD, MoMask, MotionGPT.
- Stronger IMU-only baseline: MobilePoser.
- More formal generated-IMU consistency via generated motion -> SMPL/sensor
  frames, instead of the current HumanML joint acceleration proxy.
