# ITM 论文统计证据补充

## MDM Stage-2b：统一 100-motion 控制实验

MDM Stage-2b 使用与 MotionLab 完全相同的 100 条 test motions。Factorized guidance 下，严格 Text-only 同时设置 `imu_scale=0` 和 `joint_scale=0`；仅将 `imu_scale` 置零仍会保留 Text-IMU interaction，不属于严格 Text-only。head 与 wrists 两次 Text-only 输出的最大绝对差为 `0.0`。

| Sensor | Reference | Error change (m), 95% CI | Improved, 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| head | Text-only | -0.0110 [-0.0204, -0.0010] | 60/100 (60%) [50%, 69%] | 0.0261 |
| head | zero control | -0.0045 [-0.0132, +0.0041] | 54/100 (54%) [44%, 64%] | 0.3052 |
| head | shuffled IMU | -0.0017 [-0.0086, +0.0055] | 51/100 (51%) [41%, 61%] | 0.6446 |
| wrists | Text-only | -0.0196 [-0.0284, -0.0103] | 70/100 (70%) [61%, 79%] | <0.0001 |
| wrists | zero control | -0.0101 [-0.0182, -0.0019] | 65/100 (65%) [55%, 74%] | 0.0155 |
| wrists | shuffled IMU | -0.0203 [-0.0269, -0.0136] | 74/100 (74%) [65%, 82%] | <0.0001 |

MDM wrists 在 paired-vs-shuffled 上首次超过预设 70% 稳定门槛；head paired-vs-shuffled 与 paired-vs-zero 均不显著，因此 head-only 的实例控制主张仍应保持收窄。100-motion 通用动作子集上的 head/wrists jerk 差值 95% CI 均跨零，不能把早期 30 条 walking suite 的 jerk 改善泛化为所有动作。

原始结果：

```text
outputs/mdm_control/control_test100/summary.json
outputs/mdm_control/control_test100/statistics.json
```

## 协议

MotionLab V2 控制实验使用固定的 100 条 test motions。每条动作在共享文本、长度、随机种子和初始生成噪声的前提下，比较 Text-only、paired IMU、zero control 与 shuffled IMU。以下区间由 20,000 次确定性配对 bootstrap 得到；p 值来自 20,000 次双侧 paired sign-flip randomization test。误差差值定义为 `paired - reference`，负值表示 paired IMU 更好。

## Active-joint trajectory proxy

| Sensor | Reference | Error change (m), 95% CI | Improved, 95% CI | p |
| --- | --- | ---: | ---: | ---: |
| head | Text-only | -0.0185 [-0.0319, -0.0073] | 60/100 (60%) [50%, 69%] | 0.0010 |
| head | zero control | +0.0014 [-0.0043, +0.0069] | 42/100 (42%) [32%, 52%] | 0.6269 |
| head | shuffled IMU | -0.0070 [-0.0116, -0.0029] | 63/100 (63%) [54%, 72%] | 0.0008 |
| wrists | Text-only | -0.0309 [-0.0512, -0.0122] | 63/100 (63%) [54%, 72%] | 0.0024 |
| wrists | zero control | -0.0064 [-0.0192, +0.0049] | 54/100 (54%) [44%, 64%] | 0.3163 |
| wrists | shuffled IMU | -0.0159 [-0.0271, -0.0074] | 67/100 (67%) [58%, 76%] | <0.0001 |

## Jerk ratio

| Sensor | Paired minus Text-only, 95% CI | p |
| --- | ---: | ---: |
| head | -0.3143 [-0.5199, -0.1556] | <0.0001 |
| wrists | -0.3485 [-0.5543, -0.1905] | <0.0001 |

## 可支持的结论

- paired IMU 相对 Text-only 和 shuffled IMU 的平均 active-joint error 更低，且 jerk 更低。
- paired 相对 shuffled 的改善说明模型不只响应“控制通道已开启”，其中包含实例级 IMU 信息。
- paired 相对 zero control 的差异在 head 和 wrists 上均不显著，说明控制通道仍存在较强的固定条件偏置。
- 单样本稳定性没有达到预设的 70% 门槛：head 与 wrists 相对 Text-only 的改善率 95% 区间分别为 50%-69% 和 54%-72%。因此不能声称模型在绝大多数动作上都能可靠跟随 IMU。

## 边界

这些结果只适用于固定的 100-motion 配对子集。`active-joint trajectory error` 是 HumanML joint-space proxy，不是真实 generated-IMU orientation error。统计显著的平均改善不等于每条序列都改善，论文必须同时报告改善率及其区间。

可重复生成命令：

```bash
conda run --no-capture-output -n itm python \
  scripts/summarize_motionlab_control_statistics.py \
  --input outputs/motionlab/control_test100/summary.json \
  --output outputs/motionlab/control_test100/statistics.json
```
