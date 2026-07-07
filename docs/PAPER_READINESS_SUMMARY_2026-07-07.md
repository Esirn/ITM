# ITM Paper Readiness Summary - 2026-07-07

## 1. 项目任务与定位

ITM 的核心任务是 **IMU-guided Text-to-Motion Generation**：

> 在文本生成动作的基础上，引入少量 IMU 读数作为时序控制信号，使生成动作既符合文本语义，又能被稀疏身体传感器约束。

当前最重要的两个互补现象是：

1. **文本模糊时，IMU 提供具体运动约束。**
   例如同一句 `a person walks`，换不同人的 head/wrist IMU，生成动作应体现不同步速、步频、轨迹、身体摆动。
2. **IMU 部位不足时，文本补全全身语义。**
   例如只有头部 IMU 时，文本 `walking with swinging arms` 应让手臂合理摆动，而不是只做头部轨迹跟随。

这个任务的主心骨是 **controllable motion generation**，不是传统姿态估计。MPJPE、关节重建 MSE 等指标只能作为诊断；论文主指标应围绕文本生成质量、多样性和 IMU 控制一致性展开。

## 2. 意义与新颖性

### 2.1 为什么有意义

纯 Text-to-Motion 模型擅长生成语义合理的动作，但文本天然模糊：

- `a person walks` 不指定具体步频、速度、路径、身体晃动。
- 同一文本可以对应大量不同动作实例。
- 用户想控制“具体怎么走”时，文本不够直接。

纯 IMU-to-Motion 或 sparse-IMU pose estimation 可以利用真实身体传感器，但在传感器稀疏时：

- 头部 IMU 难以决定手臂、腿部、全身语义。
- IMU-only 更像重建或估计，不天然具备文本语义编辑能力。
- 稀疏传感器输入容易产生多解，单纯回归会过平滑或不自然。

ITM 的意义在于把这两类信息拆成不同职责：

- Text：给出动作语义、全身先验和高层意图。
- IMU：给出个体化、时序化、物理轨迹约束。
- Generator：在预训练动作生成先验中融合二者，而不是简单拼接两个输出动作。

### 2.2 可主张的新颖性

当前版本可以形成以下论文主张：

1. **任务定义新颖。**
   将稀疏 IMU 作为 Text-to-Motion 生成模型的控制条件，而不是把文本当作 IMU pose estimation 的辅助输入。
2. **反事实评估协议清晰。**
   明确设计同文本换 IMU、同 IMU 换文本、Text/IMU 交叉矩阵，用来观察两个模态是否真正互补。
3. **轻量注入方式实用。**
   冻结官方 MDM 和 CLIP，只训练 IMU encoder 与零初始化 residual adapters，降低训练成本并保留 MDM 的文本生成能力。
4. **灵活传感器配置。**
   当前支持 head-only 和 wrists，数据桥保留 6 个标准 IMU 槽位，为后续 1/2/3/6 传感器扩展打基础。

这些主张目前适合写成 **work-in-progress / pilot evidence**。要成为强论文结论，还需要更稳定的生成质量、更多正式指标和更完整 baseline。

## 3. 与常见任务的边界

| 任务 | 输入 | 输出 | 目标 | ITM 的区别 |
| --- | --- | --- | --- | --- |
| Text-to-Motion | text | motion | 文本语义匹配、多样性、自然性 | ITM 增加 IMU，使同一文本可被具体身体信号控制 |
| IMU pose estimation | sparse IMU | pose/motion | 尽量重建真实姿态，常用 MPJPE | ITM 不是只追求单一 GT 重建，而是在生成空间中融合语义与控制 |
| Text-assisted IMU reconstruction | text + IMU | pose | 文本辅助姿态估计 | ITM 的 backbone 是生成模型 MDM，目标是可控生成，不是回归姿态 |
| Motion inbetweening / trajectory control | text/trajectory/keyframes | motion | 按轨迹或关键帧补全动作 | ITM 的控制信号来自 wearable IMU 的 acceleration/orientation |
| Ego4D/Ego4o-style egocentric motion | egocentric video/sensors/text | motion/action | 第一视角理解或重建 | ITM 只参考其多模态思想，不绑定视频或完整 ego pipeline |

论文中应避免把 ITM 写成“更好的 TransPose/IMUPoser”。更准确的说法是：

> ITM studies sparse-IMU controllability for pretrained text-to-motion diffusion models.

## 4. 数据与预处理

### 4.1 数据来源

当前实验使用 HumanML3D/AMASS 对齐数据：

- HumanML3D caption 和 263D motion representation。
- 通过 HumanML3D `index.csv` 映射回 AMASS/SMPL 序列。
- 从 SMPL/AMASS 生成标准虚拟 IMU。

当前标准 IMU cache：

| split | aligned records |
| --- | ---: |
| train | 7009 |
| val | 434 |
| test | 1333 |

早期 sanity split：

| split | records |
| --- | ---: |
| train | 999 |
| val | 200 |
| test | 200 |

### 4.2 Motion 表示

融合模型沿用 MDM/HumanML3D：

- HumanML3D 263D motion representation。
- 20 FPS。
- 最大长度 196 frames。
- 输出后使用 `recover_from_ric` 恢复 22 个 HumanML3D joints。

### 4.3 IMU 表示

标准虚拟 IMU：

- 6 个固定槽位。
- 每槽每帧 12D：
  - 3D acceleration。
  - 展平的 3x3 global orientation。
- 缓存频率 30 FPS。
- 训练/采样时重采样到 20 FPS：
  - acceleration：线性插值。
  - orientation：Slerp。
- acceleration 使用训练集均值/方差标准化。
- orientation 不标准化。

固定槽位：

| slot | part |
| ---: | --- |
| 0 | Left Wrist |
| 1 | Right Wrist |
| 2 | Left Thigh |
| 3 | Right Thigh |
| 4 | Head |
| 5 | Pelvis |

当前融合训练覆盖：

- `head`: slot 4。
- `wrists`: slots 0, 1。

## 5. 当前模型实现

当前主模型是 **Frozen MDM + Sparse IMU Control Adapters**。

### 5.1 Frozen MDM backbone

使用官方 MDM HumanML3D checkpoint：

```text
humanml_trans_enc_512/model000475000.pt
```

关键配置：

| component | value |
| --- | --- |
| Text encoder | frozen CLIP ViT-B/32 |
| Motion representation | HumanML3D 263D |
| Latent width | 512 |
| Backbone | 8-layer Transformer Encoder |
| Diffusion steps | 1000 |
| Max length | 196 frames |
| Prediction target | predicted x_start / denoised motion |

### 5.2 IMU encoder

输入：

```text
B x T x 6 x 12
```

计算流程：

1. 每个传感器 12D feature 线性投影到 512D。
2. 加入可学习 sensor/part embedding。
3. 按 sensor mask 做 masked mean pooling。
4. 得到每帧一个 512D IMU token。
5. 送入 2 层、8 heads 的 temporal Transformer Encoder。

当前设计是在时序编码前对多传感器做 masked mean。这是一个简洁实现，但会损失不同传感器之间的显式交互结构，是后续可以改进的点。

### 5.3 Zero-initialized residual adapters

在 MDM 的每个 Transformer layer 后加入一个独立线性 adapter：

```text
H_l = MDMLayer_l(H_{l-1})
H_l = H_l + ZeroLinear_l(C_imu)
```

特点：

- 8 个 adapter，各层不共享。
- weight/bias 零初始化。
- 初始化时模型行为等同原始 MDM。
- 训练时只更新 IMU encoder 和 adapters。
- 不是完整 ControlNet：没有复制 MDM backbone，没有 cross-attention。

### 5.4 Guidance

推理时使用三分支 guidance：

```text
output =
    unconditional
    + text_scale * (text_only - unconditional)
    + imu_scale  * (text_imu - text_only)
```

这允许独立控制：

- Text-only：`imu_scale = 0`。
- IMU-only diagnostic：文本为空或 `text_scale = 0`。
- Text+IMU：文本和 IMU 同时启用。

## 6. 训练方式与超参数

当前 stage-1 fusion training：

| item | value |
| --- | --- |
| checkpoint | `outputs/mdm_control/stage1_full_pilot_v2.pt` |
| training records | 7009 |
| sensor configs | head, wrists |
| epochs | 5 |
| batch size | 16 |
| optimizer | AdamW |
| learning rate | `1e-4` |
| gradient clipping | 1.0 |
| MDM backbone | frozen |
| CLIP | frozen |
| trainable modules | IMU encoder + 8 adapters |
| text dropout | 0.1 |
| IMU dropout | 0.1 |
| loss | official MDM diffusion loss |

五个 epoch 的 loss：

```text
0.07791, 0.08080, 0.07893, 0.07619, 0.07642
```

当前没有加入：

- 显式 IMU consistency loss。
- joint trajectory loss。
- velocity consistency loss。
- foot contact / foot skating loss。
- physics loss。
- SMPL rotation decoder。

这解释了当前模型能产生互补信号，但 jerk 偏高、稳定性不够。

## 7. 已完成实验

### 7.1 早期线性重建 baseline

目的：验证数据流、manifest、synthetic IMU cache 和简单 ablation。

Train 999，Val 200：

| variant | eval MSE | eval MAE |
| --- | ---: | ---: |
| IMU only | 0.024062 | 0.080926 |
| orientation only | 0.024486 | 0.081705 |
| full | 0.025112 | 0.085994 |
| text only | 0.057667 | 0.132711 |
| acceleration only | 0.057778 | 0.127983 |
| time only | 0.058600 | 0.129147 |

Train 999，Test 200：

| variant | eval MSE | eval MAE |
| --- | ---: | ---: |
| IMU only | 0.024277 | 0.082120 |
| orientation only | 0.024697 | 0.082876 |
| full | 0.025556 | 0.087224 |
| acceleration only | 0.060430 | 0.131076 |
| time only | 0.061277 | 0.132269 |
| text only | 0.062518 | 0.137824 |

结论：synthetic IMU 直接来自 GT joints，因此 IMU-heavy 输入在重建 MSE 上很强。线性 text feature 基本无帮助。

### 7.2 Frame-level MLP baseline

目的：验证 PyTorch/GPU 训练、非线性 frame regression。

999 train / 200 val，30 epochs，2 层 256 hidden，batch size 1024，GPU 0：

| variant | train MSE | eval MSE | eval MAE |
| --- | ---: | ---: | ---: |
| IMU only | 0.009864 | 0.014230 | 0.061955 |
| full | 0.006732 | 0.020078 | 0.078787 |
| text only | 0.014815 | 0.081496 | 0.154478 |

结论：hashed text 容易记忆训练集，泛化差；文本没有在 deterministic reconstruction 目标下体现价值。

### 7.3 Temporal Transformer baseline with DistilBERT

目的：用 frozen pretrained text embedding 替换 hash，并加入时序建模。

999 train / 200 val / 200 test，4 层 Transformer，d=256，8 heads，30 epochs：

| variant | best epoch | val MSE | test MSE | test MAE |
| --- | ---: | ---: | ---: | ---: |
| temporal IMU-only | 29 | 0.017614 | 0.018271 | 0.075857 |
| temporal DistilBERT + IMU | 29 | 0.017996 | 0.018774 | 0.078116 |

结论：预训练文本减少了 hash 的灾难性泛化问题，但在全 6 传感器确定性重建任务里仍不如 IMU-only。

### 7.4 Sparse sensor ablation

目的：找到文本可能发挥作用的欠定传感器配置。

| sensors | text | val MSE | test MSE | test MAE |
| --- | --- | ---: | ---: | ---: |
| pelvis (1) | no | 0.030690 | 0.031836 | 0.099072 |
| pelvis (1) | yes | 0.030185 | 0.032504 | 0.099689 |
| head (1) | no | 0.028923 | 0.030411 | 0.097918 |
| head (1) | yes | 0.028442 | 0.030593 | 0.098537 |
| wrists (2) | no | 0.029189 | 0.030797 | 0.098079 |
| wrists (2) | yes | 0.028406 | 0.029610 | 0.096633 |
| pelvis + wrists (3) | no | 0.023170 | 0.023793 | 0.087235 |
| pelvis + wrists (3) | yes | 0.022950 | 0.023650 | 0.088568 |
| all (6) | no | 0.017614 | 0.018271 | 0.075857 |
| all (6) | yes | 0.017996 | 0.018774 | 0.078116 |

结论：

- wrists 是早期确定性重建中唯一 text+IMU 同时改善 test MSE/MAE 的配置。
- head-only 在 val 有小改善，但 test 不复现。
- sensor location 比 sensor count 更关键。
- 这些结果是 sanity baseline，不是最终生成任务证据。

### 7.5 CLIP temporal baseline and visualization

目的：加入领域更常见的 CLIP text encoder，并生成四路可视化。

| condition | val MSE | test MSE | test MAE |
| --- | ---: | ---: | ---: |
| wrists IMU only | 0.029189 | 0.030797 | 0.098079 |
| DistilBERT + wrists IMU | 0.028406 | 0.029610 | 0.096633 |
| CLIP ViT-L/14 + wrists IMU | 0.031065 | 0.033875 | 0.104977 |
| DistilBERT text only | 0.052178 | 0.056288 | 0.130745 |
| CLIP ViT-L/14 text only | 0.054305 | 0.057097 | 0.136571 |

可视化目录：

- `outputs/visualizations/wrists_distilbert/`
- `outputs/visualizations/wrists_clip_vitl14/`
- `outputs/visualizations/wrists_clip_vitl14_four_way/`
- `outputs/visualizations/head_distilbert_four_way/`
- `outputs/visualizations/head_clip_vitl14_four_way/`

结论：CLIP 是合理的论文 baseline，但旧 broadcast-add regression 架构不能发挥 CLIP 优势。用户定性反馈指出关节像果冻，Text-only 明显不合格，因此项目路线重置到官方 MDM。

### 7.6 Official MDM Text-only baseline

目的：建立合格 Text-to-Motion 底座。

使用官方 MDM HumanML3D checkpoint：

- CLIP ViT-B/32。
- `humanml_trans_enc_512/model000475000.pt`。
- 1000-step diffusion sampling。

已完成：

- wrapper：`scripts/run_mdm.py`。
- renderer：`scripts/render_mdm_results.py`。
- smoke artifact：`outputs/mdm/text_only_smoke/comparison.gif`。

官方 checkpoint 附带 20-rep reference log：

| metric | value |
| --- | ---: |
| FID | 0.5443 |
| R-precision top-1 | 0.3195 |
| R-precision top-2 | 0.4978 |
| R-precision top-3 | 0.6110 |
| Diversity | 9.5595 |

注意：本项目实现了 5-rep official debug evaluation，但尚未启动，因为上游脚本预估约 3 GPU-hours。

### 7.7 Flexible IMUPoser baseline

目的：建立合格 IMU-only 底座，避免把 poor IMU-only 与 fusion 对比。

实现：

- 固定五槽输入。
- sensor masking。
- 2-layer 512-wide bidirectional LSTM。
- 输出 24 SMPL joint rotations in 6D。
- pose velocity 与 acceleration losses。

Medium-data run：604 train / 115 validation。

| configuration | MPJPE | rotation error | jerk ratio | foot skating |
| --- | ---: | ---: | ---: | ---: |
| head | 14.75 cm | 0.275 rad | 0.314 | 0.142 m/s |
| wrists | 10.59 cm | 0.233 rad | 0.408 | 0.324 m/s |

结论：

- wrists 通过 12 cm gate。
- head-only 尚未通过。
- jerk 很低可能是过平滑，不一定代表真实动作更好。

### 7.8 MDM IMU Control qualitative pilot

目的：验证冻结 MDM + IMU adapter 是否有基本互补信号。

训练：

- 7009 train records。
- head + wrists。
- 5 epochs。
- final checkpoint：`outputs/mdm_control/stage1_full_pilot_v2.pt`。

可视化目录：

- `outputs/mdm_control/qualitative/four_way/`
- `outputs/mdm_control/qualitative/same_text_different_imu/`
- `outputs/mdm_control/qualitative/same_imu_different_text/`

训练样例结果：

- 同文本换 head IMU 产生 15.9-20.1 cm mean root-relative joint differences。
- 固定 head IMU 时，arm-swing caption 将 wrist-relative-to-shoulder motion proxy 从 0.155 m 提升到 0.298 m，约 92%。
- text scale 4.0 反而降到 0.110 m，说明 guidance 非单调。

初始 unseen-test pilot：

- 同文本换 IMU 仍产生 9.0-20.1 cm mean root-relative differences。
- 固定 head IMU + arm-swing prompt 未泛化，proxy 从 0.199 m 降到 0.100 m。

结论：第一方向较稳，第二方向需要更系统测试。

### 7.9 Next-stage counterfactual suite

目的：在 test split 上系统验证两个互补现象，并做 guidance sweep。

输出：

- `outputs/mdm_control/experiments/`
- 60 completed runs。
- 770 generated cases。
- `RESULTS_SUMMARY.md/json`。

#### Same vague text, different IMU

固定文本：

```text
a person walks
```

30 个 test samples，每个包含：

- MDM Text-only。
- ITM Text + head IMU。
- ITM Text + wrist IMUs。

| group | active sensor error | jerk ratio | arm swing | root travel |
| --- | ---: | ---: | ---: | ---: |
| all | 0.2871 m | 4.2072 | 0.1909 m | 2.1561 m |
| head | 0.1733 m | 4.5210 | 0.2021 m | 2.1826 m |
| wrists | 0.5147 m | 3.5796 | 0.1686 m | 2.1030 m |

结论：同文本换 IMU 会改变 root travel、step frequency 和 motion proxies，支持 IMU 提供具体运动约束。但 jerk 偏高。

#### Same head IMU, different text

固定 head-only IMU，换 6 条 walking prompts：

| prompt | head error | arm swing | jerk | root travel |
| --- | ---: | ---: | ---: | ---: |
| walking with swinging arms | 0.1662 m | 0.2615 m | 6.2430 | 3.3377 m |
| walking quickly | 0.1824 m | 0.2304 m | 5.6510 | 2.4707 m |
| turning while walking | 0.2088 m | 0.2153 m | 4.1204 | 1.4661 m |
| walking | 0.1731 m | 0.2088 m | 5.1645 | 2.4170 m |
| walking slowly | 0.1776 m | 0.1742 m | 4.1510 | 2.1081 m |
| walking with large steps | 0.1948 m | 0.1582 m | 3.7824 | 1.8873 m |

结论：相比早期单样本 pilot，文本补全现象更清楚。`walking with swinging arms` 的 arm swing proxy 最高，同时 head error 保持较低。但 jerk 仍高，不能作为最终质量结论。

#### Guidance sweep

扫描：

```text
text_scale in {1.0, 2.5, 4.0}
imu_scale  in {0.5, 1.0, 2.0}
sensor     in {head, wrists}
```

当前推荐：

| sensor | text scale | IMU scale | active sensor error | jerk |
| --- | ---: | ---: | ---: | ---: |
| head | 2.5 | 0.5 | 0.2003 m | 1.6134 |
| wrists | 2.5 | 0.5 | 0.4794 m | 1.5036 |
| wrists | 2.5 | 1.0 | 0.4847 m | 1.5066 |

#### Matrix experiment

20 组 A/B test samples，分别跑 head 和 wrists。

条件矩阵：

| Text | IMU |
| --- | --- |
| None | A |
| None | B |
| A | None |
| A | A |
| A | B |
| B | None |
| B | A |
| B | B |

同时展示 Ground Truth A/B。

候选可视化：

- low jerk：
  - `matrix_test_wrists/pair_001_004488_004222`
  - `matrix_test_wrists/pair_007_008238_006378`
  - `matrix_test_head/pair_001_004488_004222`
- high contrast：
  - `matrix_test_wrists/pair_013_007089_006756`
  - `matrix_test_head/pair_013_007089_006756`
  - `matrix_test_wrists/pair_004_002572_010157`
- good head tracking：
  - `matrix_test_head/pair_001_004488_004222`
  - `matrix_test_head/pair_002_008803_009184`
  - `matrix_test_head/pair_008_002888_002382`

## 8. Metrics 与含义

### 8.1 Text-to-Motion metrics

这些是论文最终应重点补齐的官方 MDM/HumanML3D 指标：

| metric | meaning |
| --- | --- |
| Matching Score | 生成动作与文本 embedding 的距离，越低越好 |
| R-precision | 文本-动作检索 top-k 准确率，越高越好 |
| FID | 生成动作分布与真实动作分布距离，越低越好 |
| Diversity | 生成动作整体多样性 |
| Multimodality | 同一文本下多样本差异 |

当前只引用了官方 checkpoint reference log，尚未完成本方法的 full official evaluation。

### 8.2 IMU/control metrics

当前使用的是 proxy 指标：

| metric | meaning |
| --- | --- |
| root_trajectory_error_m | 生成 root 轨迹与目标动作 root 轨迹距离 |
| head_trajectory_error_m | root-relative head joint 与目标距离 |
| wrist_trajectory_error_m | root-relative wrists 与目标距离 |
| active_sensor_trajectory_error_m | 当前传感器对应 joint 的 root-relative trajectory error |
| root_relative_motion_error_m | 全身 root-relative joint error |
| jerk_ratio | 生成 jerk / GT jerk，越接近或低于合理范围越好 |
| arm_swing_proxy_m | wrist-relative-to-shoulder motion amplitude |
| root_travel_m | root horizontal travel distance |
| step_frequency_hz | foot-speed frequency proxy |
| pairwise_root_relative_distance_m | 同批生成动作之间的 root-relative 差异 |

注意：这些指标不是严格真实 IMU orientation error。当前还没有从生成动作重新合成标准 IMU orientation 并与输入 IMU 做完整 comparison。

### 8.3 IMU-only metrics

Flexible IMUPoser 使用：

| metric | meaning |
| --- | --- |
| root-relative MPJPE | 去 root 后的平均关节位置误差 |
| rotation geodesic error | rotation matrix geodesic distance |
| jerk ratio | temporal smoothness diagnostic |
| foot skating | 接触脚水平速度 proxy |

这些适合 IMU-only pose estimation baseline，但不应成为 ITM 生成任务的主指标。

## 9. 当前主要挑战

1. **生成稳定性不足。**
   Stage-1 model 已出现互补信号，但 jerk 偏高，动作还不够自然。
2. **IMU 控制指标仍是 proxy。**
   需要正式实现 generated motion -> virtual IMU -> acceleration/orientation consistency。
3. **Text-to-Motion 官方指标未对 ITM 进行完整评估。**
   需要跑 MDM evaluator，确认 Text+IMU 不破坏 MDM 的文本匹配、FID、多样性。
4. **IMU-only baseline 还不够强。**
   Wrists 达到初步门槛，head-only 未达 12 cm MPJPE gate。
5. **传感器配置仍少。**
   当前 fusion 只覆盖 head 和 wrists；论文需要至少覆盖 1/2/3/6 sensors 或明确限定任务范围。
6. **当前训练 objective 没有显式控制损失。**
   这很可能是 jerk 高和 IMU adherence 不稳定的原因。
7. **多 seed 和置信区间缺失。**
   目前多数结果是 single seed。正式论文需要多 seed 或至少 bootstrap/confidence interval。

## 10. 是否足以启动论文撰写

### 结论

**足以启动论文撰写，但不足以进入最终投稿版。**

更具体地说：

- 可以开始写：
  - Introduction。
  - Related Work。
  - Problem Formulation。
  - Method。
  - Dataset/Protocol。
  - Preliminary Experiments。
  - Qualitative Results。
- 暂时不应定稿：
  - Abstract 中的强性能 claim。
  - Main Results table。
  - SOTA comparison。
  - 最终 conclusion。

### 为什么足以启动

当前已经具备论文初稿所需的核心骨架：

1. 明确任务：IMU-guided Text-to-Motion generation。
2. 明确差异：不是 IMU pose estimation，也不是纯 Text-to-Motion。
3. 可运行方法：Frozen MDM + IMU encoder + zero-initialized adapters。
4. 可重复数据桥：HumanML3D/AMASS -> standard virtual IMU。
5. 初步互补证据：
   - 同文本换 IMU 有稳定变化。
   - 同 head IMU 换文本能影响 arm swing。
6. 可视化/demo 系统已经能浏览正式实验结果。

### 为什么还不足以投稿

目前还缺论文强证据：

1. 没有完成 ITM 的官方 HumanML3D text generation metrics。
2. 没有正式 IMU consistency metric。
3. Stage-1 model jerk 高，定性质量还可能被质疑。
4. IMU-only baseline 不完整，head-only 未达门槛。
5. 缺多 seed、多样性、失败样例分析和统计显著性。
6. 没有第二阶段加入控制/平滑 loss 的改进模型。

### 建议论文撰写策略

现在可以立刻开一个论文 draft，采用以下结构：

1. **先写成清晰的问题定义论文。**
   强调任务和反事实评估，而不是声称全面 SOTA。
2. **方法部分按当前实现写。**
   明确 frozen MDM、IMU encoder、zero adapters、guidance。
3. **实验部分先放 pilot/diagnostic results。**
   把当前 60-run test suite 作为 preliminary evidence。
4. **同时推进第二阶段模型。**
   加入 trajectory/velocity consistency 和 jerk regularization。
5. **等第二阶段结果出来后再决定投稿目标。**
   如果质量明显改善，可冲正式 workshop/conference paper；如果提升有限，更适合作为 workshop/short paper 或技术报告。

## 11. 下一步优先级

### P0: 第二阶段训练

保持 MDM/CLIP 冻结，继续训练 IMU encoder/adapters，加入：

- active sensor trajectory consistency loss。
- velocity consistency loss。
- jerk regularization。
- 更均衡的 text-only / IMU-only / text+IMU condition dropout。

目标：降低 jerk，同时保持当前互补信号。

### P1: 正式评价

补齐：

- official MDM evaluator for Text-only and ITM。
- generated motion -> virtual IMU consistency。
- guidance scale trade-off curves。
- 多 seed 或 bootstrap。

### P2: Baseline 补强

补齐：

- stronger IMU-only baseline, especially head-only。
- official MDM text-only comparison。
- old regression Transformer 只作为 lower bound。

### P3: 论文材料

整理：

- method figure。
- counterfactual figure。
- qualitative matrix examples。
- failure cases。
- task boundary figure/table。

## 12. 当前可引用文件

| purpose | path |
| --- | --- |
| model architecture | `docs/MODEL_ARCHITECTURE.md` |
| next experiment commands | `docs/NEXT_EXPERIMENTS.md` |
| latest experiment progress | `docs/EXPERIMENT_PROGRESS_2026-07-07.md` |
| baseline reset progress | `docs/BASELINE_RESET_PROGRESS_2026-07-02.md` |
| early experiment progress | `docs/EXPERIMENT_PROGRESS_2026-06-26.md` |
| experiment results summary | `outputs/mdm_control/experiments/RESULTS_SUMMARY.md` |
| interactive demo | `src/itm/demo/app.py` |
