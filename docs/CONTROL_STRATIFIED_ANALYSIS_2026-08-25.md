# ITM 控制效果分层分析

## 协议

本分析不重新采样，复用 MDM Stage-2b 和 MotionLab Direct-V2 在同一固定 100-motion test subset 上的严格 Text-only、paired 和 shuffled IMU 输出。

- 语义分层采用预先固定的关键词多标签规则，一个 caption 可以同时属于多个类别。
- 长度分层固定为 1--80、81--140、141--196 帧。
- 所有结果均为探索性描述；类别存在重叠且部分样本量较小，不进行类别级显著性主张。
- `paired - reference` 为负表示 paired IMU 的 active-joint trajectory proxy 更低。

完整逐样本结果位于：

```text
outputs/comparisons/control_test100_strata/summary.json
outputs/comparisons/control_test100_strata/summary.md
```

## MDM Stage-2b 语义分层

| Sensor | Category | N | Paired-Text error | Better than Text | Paired-Shuffled error | Better than Shuffled | Jerk change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | locomotion | 34 | -0.0186 m | 22/34 | -0.0010 m | 17/34 | -0.124 |
| head | turning | 14 | -0.0292 m | 10/14 | -0.0206 m | 7/14 | -0.181 |
| head | upper body | 38 | -0.0000 m | 21/38 | -0.0069 m | 21/38 | +0.240 |
| head | sit/stand | 18 | -0.0288 m | 11/18 | -0.0005 m | 11/18 | -0.088 |
| wrists | locomotion | 34 | -0.0294 m | 26/34 | -0.0204 m | 28/34 | -0.202 |
| wrists | turning | 14 | -0.0453 m | 13/14 | -0.0235 m | 11/14 | -0.166 |
| wrists | upper body | 38 | -0.0190 m | 27/38 | -0.0189 m | 26/38 | +0.203 |
| wrists | sit/stand | 18 | +0.0048 m | 10/18 | -0.0123 m | 14/18 | -0.090 |

Wrists 对 locomotion 和 turning 最稳定。Head 在 turning 类别的平均误差下降较大，但 paired 仅在 7/14 条上优于 shuffled，说明一部分收益可能来自开启控制通道的总体偏置，而不是可靠的实例匹配。Head 在 upper-body 类别几乎没有平均 tracking 增益，且 jerk 增加。

## 长度分层

| Model | Sensor | Length | N | Paired-Text error | Better than Text | Paired-Shuffled error | Jerk change |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| MDM Stage-2b | head | 1--80 | 15 | +0.0288 m | 6/15 | +0.0036 m | +0.648 |
| MDM Stage-2b | head | 81--140 | 17 | -0.0042 m | 8/17 | +0.0025 m | +0.004 |
| MDM Stage-2b | head | 141--196 | 68 | -0.0214 m | 46/68 | -0.0039 m | -0.077 |
| MDM Stage-2b | wrists | 1--80 | 15 | -0.0253 m | 10/15 | -0.0269 m | +0.619 |
| MDM Stage-2b | wrists | 81--140 | 17 | +0.0022 m | 8/17 | -0.0211 m | -0.112 |
| MDM Stage-2b | wrists | 141--196 | 68 | -0.0238 m | 52/68 | -0.0186 m | -0.174 |
| MotionLab Direct-V2 | head | 1--80 | 15 | -0.0263 m | 12/15 | -0.0098 m | -0.450 |
| MotionLab Direct-V2 | head | 81--140 | 17 | -0.0318 m | 10/17 | -0.0122 m | -0.568 |
| MotionLab Direct-V2 | head | 141--196 | 68 | -0.0134 m | 38/68 | -0.0050 m | -0.221 |
| MotionLab Direct-V2 | wrists | 1--80 | 15 | -0.0731 m | 9/15 | -0.0609 m | -0.460 |
| MotionLab Direct-V2 | wrists | 81--140 | 17 | -0.0324 m | 11/17 | -0.0107 m | -0.625 |
| MotionLab Direct-V2 | wrists | 141--196 | 68 | -0.0212 m | 43/68 | -0.0072 m | -0.255 |

MDM Stage-2b 的 head 控制在短序列上失败，而长序列的平均 tracking 与 jerk 均改善。Wrists 在短序列仍改善 tracking，但 jerk 明显增加。MotionLab 的平均 jerk 在三个长度组均下降，不过样本级改善率仍不稳定。

## 结论边界

- 语义类别与长度并非独立变量，短序列可能集中包含某些动作类型。
- 多标签关键词规则用于可复现诊断，不等价于人工动作分类或训练标签。
- 这些分层解释了整体平均数的来源，但不能替代多随机种子实验。
- 下一项优先实验应在分层抽取的 30 条动作上使用 5 个 diffusion seeds，验证 wrists 的 locomotion/turning收益和 MDM head 的短序列失败是否稳定。
