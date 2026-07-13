# ITM Paper Experiment Tables - 2026-07-09

## Model Choice

Current paper default:

- Main model: `outputs/mdm_control/stage2b_balanced_b.pt`.
- Text-completion diagnostic model: `outputs/mdm_control/stage1_full_pilot_v2.pt`.
- Stage-3 is not selected because it does not improve over Stage-2b overall.

## Generation Quality on IMU-Mapped Test Subset

Protocol:

- Test subset: 682 usable IMU-mapped HumanML3D samples.
- Evaluated samples: 672, due to batch size 32.
- Repetitions: 5.
- Metrics: official HumanML text-motion evaluator wrapper.
- This is a subset protocol, not the full official 1000-sample/20-rep protocol.

| model | sensor | Matching ↓ | R@1 ↑ | R@2 ↑ | R@3 ↑ | FID ↓ | Diversity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDM Text-only subset | none | 3.4152 ± 0.0196 | 0.4530 ± 0.0178 | 0.6506 ± 0.0157 | 0.7586 ± 0.0116 | 0.5089 ± 0.0441 | 10.0517 ± 0.1657 |
| ITM Stage-2b | head | 3.2680 ± 0.0519 | 0.4580 ± 0.0131 | 0.6682 ± 0.0187 | 0.7780 ± 0.0139 | 0.6526 ± 0.0521 | 10.1686 ± 0.1396 |
| ITM Stage-2b | wrists | 3.3546 ± 0.0378 | 0.4551 ± 0.0150 | 0.6598 ± 0.0218 | 0.7673 ± 0.0103 | 0.8167 ± 0.0357 | 10.0319 ± 0.1746 |

Interpretation:

- Stage-2b head preserves text-motion matching and slightly improves R-precision on the IMU-mapped subset.
- Stage-2b wrists preserves retrieval metrics but has a larger FID cost.
- This supports the claim that IMU control can be added without collapsing text generation quality, while stronger wrist control perturbs the generated distribution more.

## Control Quality: Stage-1 vs Stage-2b

Metric note:

- `active sensor error` is root-relative active-joint trajectory error.
- `active acceleration error` is a joint-derived acceleration proxy computed from generated active joints and target virtual-IMU acceleration.
- This is not a full generated-IMU orientation metric because the current ITM output is HumanML3D 263D/22 joints, not decoded SMPL sensor frames.

Same vague text, different IMU:

| model | active sensor error ↓ | active acceleration error ↓ | acceleration ratio | jerk ratio ↓ | arm swing | root travel | step frequency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-1 | 0.2871 m | 4.7340 m/s^2 | 3.5541 | 4.2072 | 0.1909 m | 2.1561 m | 1.5505 Hz |
| Stage-2b | 0.2833 m | 4.5360 m/s^2 | 2.0936 | 1.7599 | 0.1615 m | 1.4666 m | 1.4815 Hz |

Same text, sensor breakdown:

| model | sensor | active sensor error ↓ | active acceleration error ↓ | acceleration ratio | jerk ratio ↓ | root travel | step frequency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-1 | head | 0.1733 m | 3.2895 m/s^2 | 3.4004 | 4.5210 | 2.1826 m | 1.5833 Hz |
| Stage-1 | wrists | 0.5147 m | 7.6230 m/s^2 | 3.8615 | 3.5796 | 2.1030 m | 1.4848 Hz |
| Stage-2b | head | 0.1663 m | 3.1365 m/s^2 | 1.8083 | 1.7407 | 1.5127 m | 1.4786 Hz |
| Stage-2b | wrists | 0.5172 m | 7.3352 m/s^2 | 2.6641 | 1.7981 | 1.3744 m | 1.4872 Hz |

Same head IMU, different text:

| model | active head error ↓ | active acceleration error ↓ | acceleration ratio | jerk ratio ↓ | arm swing | root travel | step frequency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-1 | 0.1838 m | 3.7829 m/s^2 | 3.8952 | 4.8521 | 0.2080 m | 2.2811 m | 1.4899 Hz |
| Stage-2b | 0.2152 m | 3.2269 m/s^2 | 4.9469 | 5.6981 | 0.1581 m | 1.4263 m | 1.4094 Hz |

Interpretation:

- Stage-2b strongly reduces jerk for the same-text-different-IMU phenomenon.
- Stage-2b also lowers the active acceleration proxy in same-text and matrix runs, so the smoother motion is not only a smaller jerk number.
- Stage-2b head improves or preserves active sensor tracking in same-text experiments.
- Stage-1 remains better for the head-only text-completion/arm-swing diagnostic.
- This is the central trade-off to report honestly: smoothing and IMU control improve, but unobserved upper-body semantic completion weakens.

## IMU-Only Baseline

Flexible IMUPoser on the test split:

| sensor | records | MPJPE ↓ | rotation error ↓ | jerk ratio ↓ | foot skating ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 1333 | 14.91 cm | 0.278 rad | 0.294 | 0.123 m/s |
| wrists | 1333 | 10.74 cm | 0.233 rad | 0.455 | 0.308 m/s |

Use this as an IMU-only reconstruction baseline, not as a direct text-generation competitor.

## Qualitative Figure Shortlist

Use these as candidates, then manually inspect in the demo before putting them in a paper figure.

Primary condition-swapping examples:

- Same text, different IMU:
  `outputs/mdm_control/experiments_stage2b_balanced_b/same_text_different_imu/test_walk_head_wrists/`
- Same head IMU, different text:
  `outputs/mdm_control/experiments/same_head_imu_different_text/test_prompts/`

Matrix examples:

- Low-jerk wrists matrix:
  `outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_wrists/pair_001_004488_004222/`
- High-contrast matrix:
  `outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_wrists/pair_013_007089_006756/`
- High-contrast head matrix:
  `outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_head/pair_013_007089_006756/`
- Good head tracking:
  `outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_head/pair_001_004488_004222/`

Failure/limitation examples:

- Head-only text completion weakness:
  `outputs/mdm_control/experiments_stage2b_balanced_b/same_head_imu_different_text/test_prompts/`
- Stage-1 high-jerk but stronger arm-swing diagnostic:
  `outputs/mdm_control/experiments/same_head_imu_different_text/test_prompts/`

Figure design recommendation:

- Show GT, MDM Text-only, ITM Text+IMU, and synchronized target IMU curves.
- For head-only examples, include head trajectory and arm-swing proxy.
- For same-text examples, emphasize root travel, step frequency, and trajectory differences.
