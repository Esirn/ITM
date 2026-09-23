# ITM 主要实验结果

本目录汇总截至2026-09-23可在当前仓库和本地实验产物中核验的主要结果。正文使用中文，
尽量以表格呈现，适合论文写作、汇报和后续实验对照。

| 文件 | 内容 |
|---|---|
| [MAIN_RESULTS.md](MAIN_RESULTS.md) | 数据集、冻结基座、生成质量、控制质量、重建基线、消融和跨数据集结果 |
| [HYMOTION_EXPERIMENT_SUMMARY_2026-09-23.md](HYMOTION_EXPERIMENT_SUMMARY_2026-09-23.md) | A40上的HY-Motion Text+IMU控制实验、负结果和最新决策门槛 |
| [METRICS.md](METRICS.md) | Matching、R-precision、FID、MM、MPJPE、jerk等指标的通俗定义与方向 |
| [SOURCE_INDEX.md](SOURCE_INDEX.md) | 每组数字对应的原始JSON、日志和既有实验文档 |

## 硬件归属说明

交接前的MDM/ITM主要训练与量化实验是在双RTX 4090机器上完成的。HY-Motion Text+IMU
实验随后在A40机器完成；对应表格明确记录GPU阶段、checkpoint hash、数据subset、seed和
原始结果路径。两阶段的指标不强行横比，且`outputs/`中的大型产物仍不进入Git。

## 使用原则

- 不把不同任务的数字强行横比：IMUPoser的MPJPE是重建指标，MDM/ITM的FID是生成指标。
- `active trajectory/acceleration`目前主要是关节空间proxy，不称为真实设备IMU误差。
- 672样本结果是IMU-mapped子集5次重复，不冒充HumanML3D full-set主表。
- `±`按原结果含义保留；官方论文数字通常为95%置信区间，项目子集表为5次重复统计。
- 成功结果与负结果同时保留，尤其是head控制、快慢/大步文本属性和Nymeria wrists域偏移。
