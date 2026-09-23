# 实验指标说明

## 文本动作生成

| 指标 | 方向 | 含义 |
|---|---:|---|
| Matching Score / Multimodal Distance | ↓ | 生成动作与对应文本在预训练评价器特征空间中的距离；越小通常表示语义更匹配。 |
| R@1 / R@2 / R@3 | ↑ | 给定动作，从候选文本中检索正确描述进入前1/2/3名的比例。 |
| FID | ↓ | 生成动作和真实动作特征分布的Fréchet距离；越低表示整体分布越接近真实数据。 |
| Diversity | → | 不同文本生成动作之间的总体差异；不是越大越好，应接近真实动作分布。 |
| Multimodality (MM) | ↑/结合质量判断 | 同一文本多次采样所得动作的差异，衡量一条描述下能否产生多种合理动作；高MM不能替代语义和自然度。 |

## IMU重建与控制

| 指标 | 方向 | 含义 |
|---|---:|---|
| MPJPE | ↓ | 预测与GT关节位置的平均欧氏距离，常以cm报告，适合确定性姿态/动作重建。 |
| Rotation error | ↓ | 预测与GT关节旋转的平均测地距离，单位rad。 |
| Active-sensor trajectory error | ↓ | 将动作转为root-relative轨迹后，head或wrists对应关节与目标轨迹的平均距离。当前是joint proxy。 |
| Active acceleration error | ↓ | 活跃关节位置二阶差分与目标虚拟IMU加速度的差异，单位m/s²。当前不含完整传感器安装外参和方向误差。 |
| Paired vs shuffled/zero | paired更低 | paired使用本动作IMU；shuffled换成其他动作IMU；zero关闭具体IMU数值。用于区分实例控制与“控制通道开启”偏置。 |
| Improvement rate | ↑ | 有多少测试动作在paired条件下优于参照条件，同时应报告置信区间。 |

## 动作质量与行为属性

| 指标 | 方向 | 含义 |
|---|---:|---|
| Jerk ratio | 接近或低于1 | 生成动作三阶时间差分相对GT的比值；过高通常表示抖动，过低也可能表示过度平滑。 |
| Foot skating | ↓ | 脚应接触地面时仍发生水平移动的速度proxy，单位m/s。 |
| Arm swing | 按文本变化 | 腕部相对肩部的运动幅度，用于检查“摆臂”文本能否补全未观测上肢。 |
| Root travel | 按条件变化 | 根节点累计移动距离，用于描述速度和路径差异。 |
| Step frequency | 按条件变化 | 从下肢周期运动估计的步频proxy，单位Hz。 |
| Root yaw | 按文本变化 | 根朝向累计或净转角，用于检查“转身/转弯”语义。 |

## 解释边界

MPJPE低不表示文本生成质量高，FID低也不表示模型服从某一条IMU。论文主结论需要同时看
生成质量、paired-vs-shuffled控制、动作平滑性和定性可视化。当前MDM输出的HumanML 22
joints不能唯一恢复骨轴twist，因此不能从关节位置可靠得到真实sensor orientation。
