# HY-Motion 实验结果总览

更新：2026-09-23。本文只汇总可复核且对下一步决策有意义的实验；原始 checkpoint、NPZ、
JSON 和 GIF 均保留在 `outputs/`。除特别说明外，A40 的控制验证固定使用相同文本、帧数、
随机种子/初始噪声、50 个 Euler 步、Text CFG=5，并比较 TextOnly、paired IMU、zero IMU
和 shuffled IMU。

## 结论摘要

| 项目 | 当前结论 | 对全量训练的含义 |
|---|---|---|
| 4090 的 HY-Motion Lite TextOnly 复现 | 官方骨干可正常生成，虚拟 IMU 导出可用 | 仅证明基座和导出链路，不证明 IMU 控制有效 |
| 早期 hidden adapter 正式预筛 | paired 相对 shuffled 的辨识率通过，但 jerk 严重恶化 | 该架构禁止启动 7,009 条全量训练 |
| 低控制强度、jerk 辅助项和简单后处理 | 都未同时满足辨识与 jerk 门槛 | 不以推理缩放或后处理替代训练改进 |
| 直接 output-velocity controller | 三组后续验证均支持真实 IMU 控制；锁定 48 条和全新 64 条的均值 jerk 基本不变或降低 | 最新预注册非劣门槛仍因 head CI 上界差 0.0476 m/s^3 而失败，不授权全量训练 |
| MDM Stage-2b | checkpoint 已迁移并完成完整性核验 | 仍是当前论文主模型；本交接资料未提供可复核生成指标 |

## 4090 交接前实验

### HY-Motion Lite TextOnly

固定提示词：`a person walks forward while swinging both arms`；60 帧；seed 1234。

| 设备 | 核心采样时间 | PyTorch 峰值分配 | 输出核验 | 结论 | 原始产物 |
|---|---:|---:|---|---|---|
| RTX 4090（cuda:1） | 1.93 s | 约 19.6 GB | 201D motion、22x6D rotation、平移、52 joints 均有限 | 官方 TextOnly smoke 成功 | `outputs/hymotion/text_only_smoke/walk_swing_60.npz` |

交接前还保存了 walk/run/sit/turn/jump 的固定提示词样例；坐下语义明确，快跑与转向可见
变化，jump 的该 seed 较弱。这些是定性样例，不是控制效果或模型间性能比较。

### MDM 控制 checkpoint

| checkpoint | epoch | 文件大小 | SHA-256 | A40 核验状态 |
|---|---:|---:|---|---|
| `stage1_full_pilot_v2.pt` | 5 | 101,045,130 B | `62f789fe3fb405390f4ee71ccb544465debfb625ec162167564de4c8e9506b71` | `torch.load`、严格加载 IMU encoder/8 个 residual adapter、参数有限检查均通过 |
| `stage2b_balanced_b.pt` | 8 | 101,045,290 B | `5d0800e61994c79f862378871e2c149fde052c7419553939a06a6a6632788a98` | 同上 |

交接文档没有保留这两个 checkpoint 的 FID、R-Precision、IMU 误差或生成对照表，因此不能把
它们与下文 HY-Motion 数字直接比较。

## A40 基础复现与数据核验

| 项目 | 结果 | 备注 |
|---|---|---|
| HY-Motion Lite CUDA TextOnly smoke | 6.3259 s；18.25 GiB 峰值分配 | 共享 GPU 上测得，不等同于独占吞吐；60 帧全部有限 |
| 输出旋转正交性 | 最大误差 `1.16e-5` | TextOnly smoke |
| 标准 IMU 缓存 | 7,009 train / 434 val / 1,333 test，共 8,776 条 | 全部重建为 30 FPS；未使用 GPU |
| 640 条 HY-Motion 控制缓存审计 | IMU round-trip 最大差 `5.149841e-4`；根相对关节最大差 `9.536743e-7` | 仅验证 ITM 重建链路 |
| 世界 FK 与 WoodenMesh | 最大差 `7.1525574e-7` m | 修复了 6D rotation 列顺序及 root translation 坐标错误后得到 |

## A40 早期 hidden adapter 正式预筛

训练：512 条 train、2 epoch、batch 1、AdamW lr `1e-4`、360 帧前缀 padding、有效帧
flow loss；冻结 460,841,673 参数 backbone 与文本编码器，训练 21,530,112 参数 adapter。

| 项目 | 数值 |
|---|---:|
| epoch 1 / 2 平均 flow loss | 0.313992 / 0.205256 |
| 完成步数 | 1,024（均有限） |
| 训练峰值 PyTorch 分配 | 2,638,223,360 B（约 2.46 GiB） |
| adapter SHA-256 | `572682c993e3b8234ff743e76b3de04497a16114cee210d68656dff3bb3a2bdb` |
| 正式验证 | 128 条 held-out；1,024 个生成结果；完整性、split、seed、partner、有限性、SO(3) 均审计通过 |

| 传感器 | paired 优于 shuffled | paired jerk | TextOnly jerk | paired/TextOnly | 门槛结论 |
|---|---:|---:|---:|---:|---|
| 头部 | 95/128 = 74.22% | 327.2454 m/s^3 | 48.0099 m/s^3 | 6.82x | 辨识通过；jerk 失败 |
| 双手腕 | 105/128 = 82.03% | 274.1077 m/s^3 | 48.0099 m/s^3 | 5.71x | 辨识通过；jerk 失败 |

因此 `full_training_allowed=false`。这是早期 hidden controller 的正式结论，已被后续
output-velocity controller 的结果取代为当前决策依据。
原始汇总：`outputs/hymotion/prescreen_pad360_eval128_20260909/summary.json`。

## A40 失败诊断实验

以下均使用原始正式 adapter 的固定 16 条验证子集，除特别说明外为 paired 结果。它们用于
缩小问题范围，不能替代 128 条正式验证。

### 控制强度缩放

| control strength | 头部 paired 优于 shuffled | 手腕 paired 优于 shuffled | 头 jerk/TextOnly | 手腕 jerk/TextOnly | 结论 |
|---:|---:|---:|---:|---:|---|
| 1.00 | 以正式 128 条结果为准：74.22% | 82.03% | 6.82x | 5.71x | jerk 失败 |
| 0.25 | 68.75% | 75.00% | 1.547x | 1.508x | 头部辨识、jerk 均未过关 |
| 0.10 | 68.75% | 75.00% | 1.105x | 1.075x | 头部辨识、jerk 均未过关 |
| 0.05 | 56.25% | 81.25% | 1.039x | 1.024x | 接近 TextOnly，但头部控制失效 |

### 新训练目标与采样后处理

| 方法 | 设置 | 主要结果 | 结论 |
|---|---|---|---|
| endpoint position jerk 辅助项 | 512 条、2 epoch、权重 `1e-7` | 头 81.25%、手腕 87.50%；paired/TextOnly jerk 6.74x / 5.70x | 训练内 jerk 降低未转化为采样平滑，失败 |
| 每步 temporal velocity filter | cuda:0、16 条、强度 0.10 | 头 75.00%、手腕 93.75%；paired jerk 446.11 / 365.99，TextOnly 80.68 | ODE 步间滤波累积使轨迹更差，失败 |
| 根平移后处理 | 原正式输出 64 条，重复 32 次 | paired/TextOnly jerk 2.53x / 1.95x | 根抖动只是部分来源，失败 |
| 根平移 + 局部 6D rotation 后处理 | 固定骨长 FK，原正式输出 64 条，重复 4 次 | paired/TextOnly jerk 4.34x / 3.19x | 仍不通过，且未接入正式输出 |

## Direct output-velocity controller

该方案不再把 IMU 残差注入冻结 DiT 的 hidden state，而是在冻结骨干输出之后学习平滑的
201 维 flow-velocity 修正。最终候选使用 512 条训练记录、4 epochs、stride 4、timestep
gate power 1 和 4 次结构内残差平滑；adapter SHA-256 为
`802518836107aae12e4c9822351a62890ffead9fe44e4e35be26a192f9f72d12`。

| 验证集 | head paired>shuffled | wrists paired>shuffled | head jerk/TextOnly | wrists jerk/TextOnly | 判定 |
|---|---:|---:|---:|---:|---|
| 固定开发 16 条 | 75.00% | 87.50% | 0.9979x | 0.9984x | 首次同时通过开发门槛 |
| 锁定新 48 条（80--127） | 81.25% | 95.83% | 0.9979x | 1.0013x | 控制通过；wrists 严格均值门槛差 0.066 m/s^3 |
| 探索复现 64 条（16--79） | 78.13% | 82.81% | 0.9903x | 0.9954x | 机制复现通过，但不是全新测试集 |
| 预注册全新 64 条（128--191） | 78.13% | 92.19% | 0.9893x | 0.9901x | 控制和严格均值门槛通过；联合非劣门槛失败 |

全新 64 条验证生成 512 个结果，覆盖、split、checkpoint hash、zero initialization、RoPE、
指标重算、旋转有效性和跨传感器 TextOnly 一致性均通过审计。paired-minus-shuffled
IMU-score 的 95% bootstrap CI 为 head `[-0.01465, -0.00496]`、wrists
`[-0.10166, -0.02700]`，证明 paired IMU 的作用不是随机扰动。

预注册的 2% jerk 非劣绝对边界为 `0.99453 m/s^3`。全新验证中 head 的
paired-minus-TextOnly jerk 95% CI 为 `[-2.2094, 1.0421]`，上界超出边界
`0.0476 m/s^3`；wrists 为 `[-2.0949, 0.9464]`，通过。因此不能事后放宽规则，
`full_training_allowed=false`。这不否定控制机制：两种传感器的 paired jerk 点估计均低于
TextOnly，但证据不足以同时满足预先设定的保守不确定性边界。

## 解释与后续边界

1. 当前 IMU 是 joint-mounted 虚拟 IMU proxy，不是硬件设备标定误差；不得将本表 IMU
   指标外推为真实设备精度。
2. 早期 hidden controller 的 IMU 控制会显著恶化 jerk；直接 output controller 已消除
   该现象，并在三个后续集合上复现 paired 相对 shuffled 的优势。
3. 最新全新 64 条结果没有观察到 mean jerk 回归，但未同时通过预注册的保守 2% 非劣
   置信上界，且 semantic gate 尚未正式量化，因此不运行 7,009 条全量训练。
4. 后续实验不得复用 128--191 来选择新阈值；应预先固定新模型或新判定规则，并使用真正
   未触碰的测试数据。

## 主要原始结果索引

| 内容 | 位置 |
|---|---|
| A40 全部交接/实验流水账 | `docs/A40_PREPARATION_2026-09-08.md` |
| 4090 交接资料 | `docs/HOST_HYMOTION_HANDOFF_2026-08-26.md`、`docs/HYMOTION_MIGRATION.md` |
| 正式 128 条 HY-Motion 验证 | `outputs/hymotion/prescreen_pad360_eval128_20260909/summary.json` |
| 三个强度扫描 | `outputs/hymotion/control_strength_{025,010,005}_eval16_20260909/summary.json` |
| jerk 辅助项预筛 | `outputs/hymotion/prescreen_512_pad360_jerk1e7_20260909/`、`outputs/hymotion/prescreen_512_pad360_jerk1e7_eval16_20260909/summary.json` |
| velocity filter 扫描 | `outputs/hymotion/velocity_smoothing_010_eval16_20260910/summary.json` |
| 根/局部旋转 CPU 诊断 | `outputs/hymotion/root_smoothing_gpu1_64_20260910.json`、`outputs/hymotion/root_smoothing_gpu1_64_strong_20260910.json`、`outputs/hymotion/local_rotation_smoothing_gpu1_64_20260910.json` |
| output controller 锁定 48 条 | `outputs/hymotion/output_velocity_x4_locked48_merged_20260921/summary.json` |
| output controller 探索 64 条 | `outputs/hymotion/output_velocity_x4_exploratory64_16_80_audited_20260922/summary.json` |
| output controller 全新 64 条 | `outputs/hymotion/output_velocity_x4_fresh64_128_192_audited_20260923/summary.json` |
