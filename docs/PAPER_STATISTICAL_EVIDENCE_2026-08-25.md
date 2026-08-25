# ITM 论文统计证据补充

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
