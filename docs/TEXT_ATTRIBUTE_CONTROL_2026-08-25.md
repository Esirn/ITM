# ITM 文本属性控制实验

## 问题

固定同一条 head IMU、动作长度、seed 和初始扩散噪声，只改变文本时，模型能否按文本改变未观测的动作属性，同时尽量保持头部约束？

属性与指标定义如下：

| Text contrast | Metric | Desired direction |
| --- | --- | --- |
| quickly vs slowly | root path speed | quick - slow > 0 |
| turns while walking vs walks | accumulated root yaw | turn - base > 0 |
| swinging arms vs walks | wrist-relative-to-shoulder swing | swing - base > 0 |
| large steps vs walks | body-forward foot excursion span | large - base > 0 |

## 协议

- 模型：MDM Stage-2b。
- 数据：30条test motions，每条使用6条walking prompts。
- 传感器：固定head-only IMU。
- Guidance：`text_scale=2.5, imu_scale=1.0`。
- 每条动作保留自身有效长度，范围39--196帧；不再采用旧批处理的最短序列截断。
- 每个属性按motion做配对差值，报告均值、20,000次bootstrap 95% CI和期望方向样本数。

## 结果

| Attribute contrast | Mean difference | 95% CI | Desired direction |
| --- | ---: | ---: | ---: |
| Quick - Slow speed | -0.0199 m/s | [-0.1197, 0.1169] | 5/30 |
| Turn - Base yaw | +2.3689 rad | [1.5691, 3.1909] | 25/30 |
| Swing - Base arm swing | +0.0979 m | [0.0812, 0.1144] | 29/30 |
| Large - Base stride span | -0.0237 m | [-0.0782, 0.0308] | 12/30 |

五种编辑prompt相对普通`a person walks`使active head error平均增加`0.0074 m`，95% CI为`[0.0031, 0.0117] m`。

## 结论

- **摆臂最稳定**：29/30样本按期望增加，上肢补全现象在完整时长下成立。
- **转向较稳定**：25/30样本增加累计yaw，但该指标还可能包含不自然的身体旋转，需结合可视化。
- **快慢不成立**：完整时长下quick并不比slow更快。旧截断短片段得到的正差值属于协议敏感结果，不进入正式结论。
- **大步不成立**：stride span没有按文本稳定增加，与人工观察一致。
- 文本编辑对head tracking的平均影响约0.74 cm，数值较小但区间为正，说明语义变化并非完全不扰动传感器约束。

二级核验进一步显示：turn prompt相对base的净yaw增加`3.2573 rad`，26/30方向正确；净转角占累计转角的效率增加`0.3196`，23/30方向正确。因此主要转向结果不是只由yaw高频抖动构成。Quick相对slow的cadence仅19/30增加，large-step相对base的脚间距仅14/30增加，仍不足以挽救这两个失败属性。

原始结果：

```text
outputs/mdm_control/text_attribute_formal/summary.json
outputs/mdm_control/text_attribute_formal/summary.md
```

历史最短序列截断诊断保存在：

```text
outputs/mdm_control/text_attribute_diagnostic/
```

该历史结果只用于说明为什么必须采用变长采样协议，不用于论文主结论。
