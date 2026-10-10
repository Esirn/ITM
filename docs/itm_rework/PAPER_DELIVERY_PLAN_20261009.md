# 面向投稿的推进调整（2026-10-09）

依据：用户本轮明确要求先完成“文本＋IMU→动作”的整体任务、对比与消融，再抠细节；模块允许有限但可重复的正收益，严重负作用应修复、替换或删除。当前进入整体复盘，不自动执行原控制均值探针。

## 当前可查看交付

现有 uvicorn Condition Lab 已接入全部六个开发动作 × head/wrists，12组、68面板，原20Hz全部帧。入口 `/static/rework-review.html`，直达弓步双腕 `/?run=itm-rework-20261009-v1-000506-wrists`。缓存 `outputs/demo_runs/itm-rework-20261009-v1-*`；目录和核验 `outputs/itm_rework/protocol/user_motion_review_20261009_v1/`。

GT、原MDM text-only路径、F paired、U paired、F physical-static全覆盖；已有IMU-only仅覆盖验证来源。弓步另显示main-knockout，其前100步是text-only，明确其起点协议不同，不能冒充标准从头训练消融。页面骨架在显示时各自root XZ居中，不改变原数据、高度或评价坐标，Root trails展示原全程位移。

全为已完成真实采样，模型/GPU查询/采样新增0，不重新挑种子或删除失败例；12组不是正式测试集。

## 推荐的最小论文实验矩阵

| 优先级 | 比较对象 | 回答的问题 | 已有情况与下一准备 |
| --- | --- | --- | --- |
| 必做 | 官方MDM（text-only） | 引入IMU的收益及文本生成质量保留 | 官方实现和权重已在工程中使用；正式测试需同一数据子集、长度和评测器重新比较。当前关闭IMU的模型路径只作内部对照，不能以它替代所有官方默认配置结果。 |
| 必做 | IMUPoser（IMU-only） | 融合方法是否胜过惯性重建 | 本机有官方源码与已有适配脚本。需核对有效传感器组合、朝向/加速度坐标、预处理、是否预测平移及可用权重。适配重训版本明确标识。 |
| 必做 | 简单文本＋IMU temporal fusion | 复杂设计是否胜过简单融合 | 已有temporal baseline代码；在新版一致数据桥上重新训练，统一监督/输入/训练数据，旧表不直接沿用。 |
| 优先核查 | Ego4o（IMU＋给定文本，无图像） | 对最接近的公开多模态方法取得何种优势 | 官方源码已在本机 `0relatedworks/ego4o-code-release`。代码可得不等于权重与无图像分支可直接复现；先核对训练/推理入口、权重和输入，避免无限期等资产。 |
| 文献及复现候选 | Spatial-Related Sensors Matters（AAAI2024） | 文本辅助惯性重建的已有方案 | 原文讨论传感器空间关系、文本融合及HTT。代码/权重及当前head/wrists设置适配尚未确认；若按原文自建，标为reimplementation。 |
| 可选 | DiffusionPoser | 与惯性条件扩散重建比较 | 官方工作使用自回归扩散和可变惯性/压力传感器；可用实现/资产及同一传感器设置需核对，不立即扩大最低实验矩阵。 |
| 扩展 | OmniControl类两阶段方案 | 简单“IMU先估计关节，再文本＋空间控制”是否足够 | 关节约束必须来自IMU估计器，不能提供GT关节轨迹；属于两阶段适配基线，计算两阶段总耗时。 |

原始6传感器TransPose/PIP属于额外传感器设置，不能直接当作head/wrists同输入主表。缺失传感器不以数值0替代原有效观测后宣称官方公平结果。

一手资料：
- MDM：https://github.com/GuyTevet/motion-diffusion-model
- IMUPoser：https://github.com/FIGLAB/IMUPoser
- Ego4o：https://jianwang-mpi.github.io/ego4o/；官方代码 https://github.com/jianwang-mpi/ego4o-code-release
- Spatial-Related Sensors Matters：https://arxiv.org/abs/2401.05412
- DiffusionPoser：https://diffusionposer.github.io/
- OmniControl：https://proceedings.iclr.cc/paper_files/paper/2024/hash/7827290f07f63485b849b66cfa3e5dd0-Abstract-Conference.html

## 公平比较与评价

先从同一新版数据桥扩展代表性train/val/test，按原AMASS来源分组排除重叠；旧7009/434/1333缓存不能未经新版映射核对直接宣布可用。开发六来源不充作正式test；预训练模型接触过的HumanML来源与新增IMU条件训练泄漏分别说明，不宣称未见过动作。

同一head-only或wrists、同一加速度/方向定义、输入帧率与序列时长；相同目标关节集合与坐标、相同文本可用性。主表标明各方法输入、外部预训练和训练数据、离线/在线条件。本文是离线任务，在线方法可比较，但注明未来帧访问差异。

重建/控制表：root-relative MPJPE、root/path或global误差（仅具备可比较平移输出的方法）、位置与速度代理误差。文本/动作质量表：同一HumanML评测器的文本匹配/R-precision与FID，附jerk、foot skating及可视化。运动幅度只作辅助诊断，不能以增幅单独判好。关节二阶差分保持proxy标注；严格IMU一致性仍需生成动作到传感器安装位置/方向的可靠闭环。

所有可比较方法按序列等权、head/wrists分开汇总，给出覆盖数和失败数。主要重建结果先采用固定K=1，不以生成方法挑选best-of-many击败确定性基线；多样性/多样本作为单列。开发先做一个预算明确的完整训练；模型和协议稳定后再做多个种子与不确定性评估，测试结果不用于选结构或权重。

## 消融的决策原则

候选项是时间编码/时序encoder、静态参照差分、条件融合/引导；损失配重F作为已失败开发候选保留，尚不是最终模块。最终主张只覆盖实际采用且验证过的设计。

关闭text或IMU可说明条件增量；结构模块删改需按相同训练设置重训，推理时去掉一项只作额外敏感性检查。一个模块可改善其对应维度而在其他维度轻微代价，不要求每样本每指标全部改善；但不能依赖一个偶然正例或一条方向有利的指标。稳定大退化则修复/替换/删除，整体无净收益的模块不保留成核心创新。

## 按完整交付组织下一阶段

1. 先和用户查看本次12组可视化，选定主任务边界与最低方法结构；不要求先修好单个弓步才允许其他工作。
2. 完成新版数据扩展可用性和上述基线接口核查，明确首轮完整训练的来源、epoch/updates、验证策略和预算；不把3来源200updates直接当方法上限。
3. 在代表性开发集跑完整方法及最低基线，依据动作、控制、语义三方面结果做一次结构取舍。诊断只为解决阻挡完整训练/比较的具体问题，避免无边界细分单个保存状态。
4. 稳定模型后做核心消融及正式测试，统一同一最终模型，不拼接多个候选的最优结果。此时冻结Method公式和Experiments主表；可先另写论文提纲/表格骨架，但本轮未改LaTeX。

原 `b0_control_mean_probe_20261009_v1`仍为未执行冻结设计，保留原文件。如果整体方案仍需要这一问题，只做有预算的辅助检查；它不再是所有后续工作的唯一前置任务。
