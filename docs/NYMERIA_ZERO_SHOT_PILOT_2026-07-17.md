# Nymeria 零样本初步对比实验（2026-07-17）

## 实验目的

检查训练于 HumanML3D/AMASS 虚拟 IMU 的 ITM，能否在不微调的情况下使用 Nymeria 的真实 Xsens IMU。该实验是跨数据集、跨传感器域的诊断，不是论文正式性能表。

## 协议

- 数据：Nymeria `body/xdata.npz` 与 `narration/motion_narration.csv`。
- 时间对齐：Head `DEVICE_TIME` 经逐序列 VRS 审计 offset 转为公共 `TIME_CODE`，再索引 Xsens `timestamps_us`。
- 有效样本：3 条来自不同录制序列的约 5 秒片段；5 条候选因目标传感器四元数无效而排除。
- 输入配置：MDM Text-only、ITM Text + head IMU、ITM Text + wrists IMU。
- 模型：`outputs/mdm_control/stage2b_balanced_b.pt`，无 Nymeria 微调。
- 公平性：三路使用相同 caption、动作长度、seed 和初始扩散噪声。

## 初步结果

| 输入 | active sensor trajectory error (m) | root-relative error (m) | jerk ratio | arm swing (m) | root travel (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MDM Text-only | 0.365 | 0.294 | 9.065 | 0.240 | 0.402 |
| ITM + head | **0.198** | **0.241** | **7.789** | 0.147 | 0.290 |
| ITM + wrists | 0.462 | 0.344 | 14.247 | 0.320 | 0.599 |

这些数值只基于 3 条样本。Text-only 行的 active sensor 指标沿用 head joint，便于与 head 配置比较；不能据此和 wrists 做严格同列排名。

## 可得结论

1. Nymeria head IMU 在零样本设置下并未被模型完全忽略：active-head 轨迹误差比 Text-only 低约 45.8%，root-relative error 也降低约 18.3%。
2. wrists 结果明显失败：轨迹误差和 jerk 均高于 Text-only，说明真实 Xsens wrists 与训练时 SMPL 虚拟 wrists 的坐标、安装朝向、噪声和加速度分布存在较强域偏移。
3. jerk ratio 整体偏高，部分原因是生成动作与 Nymeria GT 的骨架、动作分布和测量噪声不同；目前不能把它解释为模型已具备真实 IMU 泛化能力。
4. 该 pilot 支持继续做 Nymeria adapter calibration/fine-tuning，但不足以形成 Ego4o 或 Spatial-Related Sensors Matters 的正式对比结论。

## 输出

- 缓存与 manifest：`outputs/nymeria_pilot/cache/`
- 原始生成与指标：`outputs/nymeria_pilot/stage2b_zero_shot/`
- 每条四路可视化：`outputs/nymeria_pilot/stage2b_zero_shot/per_clip/`

## 下一步

1. 扩展到至少 30-50 条有效片段，并按官方 subject/session 划分训练和测试。
2. 在 Nymeria train split 上只微调 IMU encoder/adapters，保留 MDM/CLIP 冻结，比较 zero-shot 与 calibrated ITM。
3. 增加每个传感器的静态安装朝向校准与真实/虚拟 IMU 统计匹配，优先解决 wrists 失败。
4. 若要与 Ego4o 公平比较，需要按其代码训练 Nymeria 模型；因无官方 checkpoint，不能把当前 ITM 与论文数字直接横比。
