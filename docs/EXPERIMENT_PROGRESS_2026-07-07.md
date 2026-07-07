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
