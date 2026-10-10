# 基线接口核查进展（2026-10-09）

本轮已实际训练简单文本＋IMU、简单仅IMU并完成统一骨干MDM文本采样，见[20epoch开发比较](protocol/B0_EXPANDED_CANONICAL_20EPOCH_EVALUATION_20261009.md)。以下官方基线仍未填成绩。

## 官方IMUPoser

本机官方 `IMUPoser/src/imuposer/config.py` 的 `amass_combos` 明确包含 `h:[4]` 与 `lw_rw:[0,1]`，所需head/wrists配置在其设计范围内；无须给其他传感器补伪观测。`models/LSTMs/IMUPoser_Model.py` 使用每传感器12维输入，512维双向RNN，输出选定SMPL pose rotation（6D或9D），本类不输出全局translation。

后续适配重点：按官方预处理形成真实长度序列；确认24关节与所选输出集；使用一致坐标/root旋转与身体比例评价。扩展输入中的 `local_rotation` 是source20Hz保存的局部旋转，尚未将canonical Q作用到root；**不能直接用该字段冒充canonical姿态监督**。需由绑定source_pair的Q显式转换root，再用于官方姿态训练/解码。head/wrists全局平移误差不能给没有平移输出的模型填GT根轨迹后比较。

旧 `train_flexible_imu_poser.py` 是自建可变槽位变体，不作为官方IMUPoser名称下的结果。

## Ego4o

官方代码已在 `0relatedworks/ego4o-code-release`。`EgoOmniMocap/configs/imu_regressor/train_imu_regressor.py` 依赖两个外部预训练资产（transformer与VQ-VAE），配置仍为作者环境的 `/CT/EgoMocap/...` 路径；还需其HumanML mean/std和实际motion/tokenizer权重。当前本机仓库中未找到对应`.pt/.pth/.ckpt/.safetensors`文件。这是本机可用性核查，不声称作者没有发布任何权重。

其预处理有InitAlign、6D rotation、HumanML表示及196帧padding，并设置 `resize_input_sequence=True`。与本项目“不拉伸时间、保留真实帧长”设置的兼容需明确，不能仅换数据路径就宣称复现。在权重/无图像文本分支接口确认前不填虚构对比数值，也不让它阻塞已可运行最低矩阵。

官方来源：IMUPoser https://github.com/FIGLAB/IMUPoser；Ego4o https://github.com/jianwang-mpi/ego4o-code-release。精确本机文件作为代码接口事实，未向作者发消息或请求外部资产。
