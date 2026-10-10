# 数字、实现和文献来源

基于仓库 `master` 合并后 `4da3b5b` 状态准备；更细输入路径、字节数和 SHA-256 见 [source_manifest.json](source_manifest.json)。日期为 2026-10-10。

## 当前实验

| 内容 | 可审阅来源 |
| --- | --- |
| 当前整体状态 | [CURRENT_STATE](../../itm_rework/CURRENT_STATE.md) |
| 四方法共同 32 head 验证 | [评测报告](../../itm_rework/protocol/HEAD_VALIDATION32_EVALUATION_20261010.md)；[聚合](../../../outputs/itm_rework/protocol/head_validation32_evaluation_20261010_v1/aggregates.json)；[逐来源行](../../../outputs/itm_rework/protocol/head_validation32_evaluation_20261010_v1/rows.json) |
| 局部监督候选，同 8 val | [候选取舍报告](../../itm_rework/protocol/HEAD_PRIMARY_CANDIDATE_EVALUATION_20261010.md)；[聚合](../../../outputs/itm_rework/protocol/head_primary_evaluation_20261010_v1/aggregates.json) |
| 控制实现 | [mdm_imu_control.py](../../../src/itm/models/mdm_imu_control.py) |
| 100epoch 训练设置 | [plan.json](../../itm_rework/protocol/expanded_canonical_training_20261009_v2/plan.json) |
| 数据配对与片段信息 | [dataset.json](../../../outputs/itm_rework/method/expanded_canonical_inputs_20261009_v1/dataset.json) |
| 官方 IMUPoser 当前接口缺口 | [接口核查](../../itm_rework/HEAD_IMUPOSER_INTERFACE_ADDENDUM_20261010.md) |
| 语义评测器资产现状 | [准备状态](../../itm_rework/HEAD_SEMANTIC_EVALUATOR_READINESS_20261010.md) |
| 早期 A40／HY-Motion 探索背景 | [历史汇总](../../../results/HYMOTION_EXPERIMENT_SUMMARY_2026-09-23.md) |

演示表格 CSV：[head32](../tables/head32.csv)、[candidate8](../tables/head_candidate8.csv)。机器可读完整快照：[metrics_snapshot.json](../tables/metrics_snapshot.json)。表格以完整片段等权汇总，未按输出挑选；32 val 与候选 8 val 严格分开。

可视化从固定六个历史锚点完整输出制作：三个 val、三个 train，全部为 head 配置。旧三来源 U/F 与扩展 U100 不构成单变量消融。视频不做速度改变、骨长缩放或选择性截段；封面为固定中点。各预测与 GT 的原文件哈希已绑定，原文件未修改。

## 外部工作的定位与原始链接

- **MDM**：当前文本动作扩散基座与方法定义来自 [官方代码](https://github.com/GuyTevet/motion-diffusion-model) 和 [原始论文](https://arxiv.org/abs/2209.14916)。本工作冻结骨干并加自己的 IMU 控制分支；当前数值是本地共同开发协议结果，不是原作者成绩。
- **HumanML3D**：文本动作标注、HumanML 表示及评测资产来源参考 [官方仓库](https://github.com/EricGuo5513/HumanML3D)。目前并未完成标准全量 benchmark 或官方语义评测器运行。
- **AMASS**：SMPL 动作来源与数据背景参考 [官方项目](https://amass.is.tue.mpg.de/)。本轮训练／验证集合为本地小规模来源隔离开发集。
- **IMUPoser（CHI 2023）**：计划作为官方头部重建对比，依据 [官方代码](https://github.com/FIGLAB/IMUPoser) 与 [论文 DOI](https://doi.org/10.1145/3544548.3581392)。目前仍需头部输入维度、坐标、SMPL FK 与共同评价适配；不能把已跑的自建 IMU-only 称为它。
- **Ego4o**：作为后续公开多模态方法对照候选，参考 [作者项目页](https://jianwang-mpi.github.io/ego4o/)。必要模型资产与共同任务接口尚未齐备，因此无本地可比成绩。

以上链接为外部原始资料，不据此自动判定该模型适合我们的共同协议；实际可比性需以任务、输入、输出、训练数据和评价适配核查为准。

## 用户提供背景与未完成主张

基金摘要与 A40 → 4090 迁移成本由用户提供，作为研究背景和复盘信息，不虚构患者群体、医院名称、迁移工时占比或临床结论。基金原目标包含双耳定标、物理约束、下肢精准重建；当前单 head 合成配对原型并未完成全部这些目标。

本材料未宣称：全面胜过强基线、统计显著性、已完成核心训练消融、正式测试集泛化、标准 FID／R-precision、真实耳机或患者验证、严格安装点加速度／朝向闭环。现有 LaTeX 正文未在本次修改，旧结果不直接搬入当前共同主表。
