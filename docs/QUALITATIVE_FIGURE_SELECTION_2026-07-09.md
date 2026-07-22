# ITM Qualitative Figure Selection - 2026-07-09

This note selects candidate qualitative examples from existing test-split
results. The goal is to make paper/PPT figure inspection fast, not to replace
manual visual judgment.

## Figure 1: Same Vague Text, Different IMU

Use Stage-2b:

```text
outputs/mdm_control/experiments_stage2b_balanced_b/same_text_different_imu/test_walk_head_wrists/
```

Fixed text:

```text
a person walks
```

Recommended motion IDs for visual inspection:

| motion id | why inspect | head active error | head jerk | head root travel | wrists active error | wrists jerk | wrists root travel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `004817` | low jerk and clear head/wrists travel difference | 0.0952 m | 0.5505 | 0.7223 m | 0.2375 m | 0.8004 | 1.3903 m |
| `011608` | strong head-vs-wrists root travel contrast | 0.0924 m | 1.3609 | 0.0941 m | 0.3588 m | 1.3893 | 2.3597 m |
| `009477` | strongest root travel contrast, useful if visual looks clean | 0.1106 m | 1.4871 | 2.8275 m | 0.4880 m | 1.1255 | 0.1222 m |
| `000903` | head tracking improves over text-only and wrist motion differs strongly | 0.1142 m | 1.7200 | 2.1905 m | 0.4520 m | 0.5474 | 0.5377 m |
| `005300` | low wrist jerk with large head-vs-wrists travel contrast | 0.1642 m | 1.0535 | 2.3280 m | 0.5779 m | 0.5473 | 0.8610 m |

Suggested figure layout:

- GT for the IMU source.
- MDM Text-only.
- ITM Text + head IMU.
- ITM Text + wrists IMU.
- Target head/wrist acceleration curves and root trajectory inset.

Claim this figure should support:

- With the same vague text and same seed, changing IMU changes concrete motion
  details such as travel, rhythm and trajectory.

## Figure 2: Same Head IMU, Different Text

Use Stage-1 for the strongest text-completion diagnostic:

```text
outputs/mdm_control/experiments/same_head_imu_different_text/test_prompts/
```

Recommended success candidates:

| motion id | base arm swing | swing-text arm swing | delta | base head error | swing head error | swing jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `007695` | 0.1289 m | 0.3413 m | +0.2125 m | 0.1857 m | 0.1626 m | 2.4565 |
| `003255` | 0.0556 m | 0.2679 m | +0.2123 m | 0.1985 m | 0.2041 m | 1.4362 |
| `011385` | 0.0712 m | 0.2821 m | +0.2109 m | 0.1518 m | 0.0966 m | 4.1521 |
| `011978` | 0.0783 m | 0.2516 m | +0.1732 m | 0.0996 m | 0.1262 m | 2.6533 |

Recommended failure/limitation candidates:

| motion id | base arm swing | swing-text arm swing | delta | base head error | swing head error | swing jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `004344` | 0.3394 m | 0.2293 m | -0.1101 m | 0.5801 m | 0.5186 m | 5.5111 |
| `007322` | 0.3395 m | 0.2391 m | -0.1004 m | 0.1519 m | 0.0926 m | 1.6657 |
| `005697` | 0.3504 m | 0.2916 m | -0.0589 m | 0.1145 m | 0.0655 m | 4.2947 |
| `003005` | 0.3217 m | 0.2700 m | -0.0518 m | 0.0737 m | 0.0982 m | 1.9167 |

Suggested figure layout:

- Same target head IMU curve for all generated panels.
- Generated motions for:
  - no text / IMU-only diagnostic if available,
  - `a person walks`,
  - `a person walks while swinging arms`,
  - `a person walks quickly` or `a person turns while walking`.
- Head trajectory overlay and arm-swing proxy label.

Claim this figure should support:

- Text can affect unobserved upper-body motion under head-only IMU, but this
  phenomenon is not yet stable enough to make a strong solved claim.

Stage-4 text-anchor comparison:

```text
outputs/mdm_control/experiments_stage4_text_anchor/same_head_imu_different_text/test_prompts/
```

Use the same motion IDs above to compare Stage-1, Stage-2b and Stage-4. Stage-4
slightly recovers arm swing over Stage-2b but still does not match Stage-1, so it
is best presented as a text-anchor ablation rather than the main qualitative
result.

## Figure 3: A/B Matrix

Use Stage-2b matrix examples:

```text
outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_wrists/pair_001_004488_004222/
outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_wrists/pair_013_007089_006756/
outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_head/pair_001_004488_004222/
outputs/mdm_control/experiments_stage2b_balanced_b/matrix_test_head/pair_013_007089_006756/
```

Suggested layout:

- Put GT A/B on a separate top row.
- Use a 3x3 condition matrix:
  - Text: None / A / B.
  - IMU: None / A / B.
  - Leave None-None blank or use it for run metadata.

Claim this figure should support:

- The task is not only reconstruction. Text A + IMU B and Text B + IMU A are
  meaningful condition-swapping cases that separate semantic and motion-detail
  controls.

## What Still Needs Manual Inspection

- Check whether the listed low-jerk examples actually look semantically
  readable. A low metric can still hide an awkward body pose.
- For same-head text completion, include both one success and one failure in
  the paper. This is a limitation, not something to sweep under the rug.
- For Stage-4, inspect it as an ablation: it should answer whether Text-only
  anchoring helps, not replace the Stage-2b main model unless the visual quality
  is clearly better.
- Do not describe the acceleration proxy as real IMU orientation consistency.
  That requires decoding generated motion to SMPL/sensor frames.
