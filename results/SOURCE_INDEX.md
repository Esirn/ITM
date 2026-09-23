# 结果来源索引

| 结果组 | 主要来源 |
|---|---|
| 数据规模与Stage-1 | `docs/BASELINE_RESET_PROGRESS_2026-07-02.md` |
| 官方MDM full-set与MM | `outputs/mdm/checkpoints_extracted/.../eval_*_wo_mm.log`（除MM）及本地MDM论文源码`05_experiments.tex`、`tabels/humanml.tex`（MM） |
| MDM/ITM生成质量 | `docs/PAPER_EXPERIMENT_TABLES_2026-07-09.md` |
| Stage-1/2b/4控制 | `outputs/mdm_control/*/RESULTS_SUMMARY.json`，`docs/PAPER_EXPERIMENT_TABLES_2026-07-09.md` |
| 严格loss消融 | `outputs/mdm_control/loss_ablation/LOSS_ABLATION_SUMMARY.json`，`docs/PAPER_ABLATION_TABLES_2026-08-25.md` |
| 100-motion统计 | `outputs/mdm_control/control_test100/{summary,statistics}.json`，`docs/PAPER_STATISTICAL_EVIDENCE_2026-08-25.md` |
| 多seed控制 | `outputs/mdm_control/control_multiseed30/summary.json`，`docs/MDM_CONTROL_MULTISEED_2026-08-25.md` |
| 文本属性 | `outputs/mdm_control/text_attribute_formal/summary.json`，`docs/TEXT_ATTRIBUTE_CONTROL_2026-08-25.md` |
| Guidance/joint scale | `outputs/mdm_control/joint_scale_test100/summary.json`，`docs/PAPER_ABLATION_TABLES_2026-08-25.md` |
| Flexible IMUPoser | `docs/PAPER_EXPERIMENT_TABLES_2026-07-09.md`及对应evaluation JSON |
| MotionLab | `outputs/motionlab/control_test100/summary.json`，`docs/MOTIONLAB_MIGRATION.md` |
| MotionLab V4预筛 | `docs/MOTIONLAB_V4_CONTRASTIVE_PRESCREEN_2026-08-25.md` |
| Nymeria | `outputs/nymeria_pilot/stage2b_zero_shot/`，`docs/NYMERIA_ZERO_SHOT_PILOT_2026-07-17.md` |
| SMPL/virtual-IMU round trip | `outputs/mdm_control/virtual_imu_roundtrip/`，`docs/VIRTUAL_IMU_ROUNDTRIP_2026-08-25.md` |
| HY-Motion smoke | `outputs/hymotion/text_only_smoke/`，`outputs/hymotion/text_only_fixed_prompts/`，`docs/HYMOTION_MIGRATION.md` |

`outputs/`默认不进入Git。复制本目录中的数字用于论文前，应再次确认对应原始文件仍存在，
并保存最终实验环境、checkpoint hash与随机种子。
