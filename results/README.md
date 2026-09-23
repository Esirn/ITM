# ITM 主要实验结果

本目录汇总截至2026-09-10可在当前仓库和本地实验产物中核验的主要结果。正文使用中文，
尽量以表格呈现，适合论文写作、汇报和后续实验对照。

| 文件 | 内容 |
|---|---|
| [MAIN_RESULTS.md](MAIN_RESULTS.md) | 数据集、冻结基座、生成质量、控制质量、重建基线、消融和跨数据集结果 |
| [METRICS.md](METRICS.md) | Matching、R-precision、FID、MM、MPJPE、jerk等指标的通俗定义与方向 |
| [SOURCE_INDEX.md](SOURCE_INDEX.md) | 每组数字对应的原始JSON、日志和既有实验文档 |

## 硬件归属说明

交接前可核验的主要训练与量化实验是在双RTX 4090机器上完成的。当前A40环境中的完整
HY-Motion+IMU训练结果尚未同步到本工作区；当前可见的2026-09-09新产物主要是前端SMPL
mesh预览，不属于论文模型实验。当前环境执行`nvidia-smi`也无法连接驱动，因此本目录不
虚构A40性能数字。后续A40实验回传后，应新增独立表格并记录GPU、checkpoint、数据subset、
seed和原始结果路径。

## 使用原则

- 不把不同任务的数字强行横比：IMUPoser的MPJPE是重建指标，MDM/ITM的FID是生成指标。
- `active trajectory/acceleration`目前主要是关节空间proxy，不称为真实设备IMU误差。
- 672样本结果是IMU-mapped子集5次重复，不冒充HumanML3D full-set主表。
- `±`按原结果含义保留；官方论文数字通常为95%置信区间，项目子集表为5次重复统计。
- 成功结果与负结果同时保留，尤其是head控制、快慢/大步文本属性和Nymeria wrists域偏移。
