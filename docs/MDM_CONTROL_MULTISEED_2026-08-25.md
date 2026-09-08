# MDM Stage-2b 多随机种子控制实验

## 实验协议

- 数据：固定 100-motion test subset 中按长度分层抽取 30 条。
- 分层：1--80、81--140、141--196 帧各 10 条。
- Diffusion seeds：1234、2345、3456、4567、5678。
- 输入条件：严格 Text-only、paired IMU、固定 shuffled IMU。
- 传感器：head、wrists。
- Guidance：factorized，Text-only 同时设置 `imu_scale=0, joint_scale=0`。
- Bootstrap：先对每条 motion 的五个 seed 求平均，再以 30 条 motion 为独立单位重采样。

共完成 90 个 sampling chunks，无失败或跳过。完整输出位于：

```text
outputs/mdm_control/control_multiseed30/summary.json
outputs/mdm_control/control_multiseed30/summary.md
```

## 总体结果

| Sensor | Paired-Text error, 95% CI | Paired-Shuffled error, 95% CI | Jerk change, 95% CI | >=3/5 seeds better than Text | >=3/5 better than shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | -0.0038 m [-0.0108, +0.0031] | -0.0013 m [-0.0088, +0.0056] | +0.001 [-0.086, +0.087] | 13/30 | 15/30 |
| wrists | **-0.0281 m [-0.0414, -0.0160]** | **-0.0179 m [-0.0296, -0.0078]** | -0.052 [-0.147, +0.037] | **24/30** | **22/30** |

Head 的三个区间均跨零，实例级控制不能跨 seed 稳定复现。Wrists 相对 Text-only 和 shuffled IMU 的误差区间均严格小于零，且多数 motion 在至少三个 seed 上方向一致。这是当前最强的 sparse IMU control 证据。Wrists 的 jerk 均值略降，但区间跨零，不能声称通用平滑收益。

## 每个 seed

| Sensor | Seed | Paired-Text | Better than Text | Paired-Shuffled | Better than shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 1234 | +0.0013 m | 15/30 | +0.0038 m | 14/30 |
| head | 2345 | -0.0089 m | 18/30 | -0.0075 m | 19/30 |
| head | 3456 | -0.0110 m | 15/30 | +0.0009 m | 15/30 |
| head | 4567 | -0.0017 m | 15/30 | +0.0036 m | 12/30 |
| head | 5678 | +0.0016 m | 15/30 | -0.0073 m | 17/30 |
| wrists | 1234 | -0.0144 m | 21/30 | -0.0096 m | 23/30 |
| wrists | 2345 | -0.0337 m | 25/30 | -0.0174 m | 20/30 |
| wrists | 3456 | -0.0272 m | 22/30 | -0.0177 m | 19/30 |
| wrists | 4567 | -0.0322 m | 22/30 | -0.0172 m | 18/30 |
| wrists | 5678 | -0.0329 m | 20/30 | -0.0277 m | 24/30 |

Wrists 在所有五个 seed 上的平均 paired error 均低于 Text-only 和 shuffled；head 的方向随 seed 改变。

## 长度分层

| Sensor | Length | Paired-Text error, 95% CI | Paired-Shuffled error, 95% CI | Jerk change, 95% CI |
| --- | --- | ---: | ---: | ---: |
| head | 1--80 | +0.0062 m [-0.0067, +0.0163] | +0.0014 m [-0.0080, +0.0089] | +0.078 [-0.092, +0.243] |
| head | 81--140 | -0.0097 m [-0.0214, +0.0014] | +0.0012 m [-0.0134, +0.0152] | -0.149 [-0.282, -0.015] |
| head | 141--196 | -0.0078 m [-0.0178, +0.0033] | -0.0065 m [-0.0213, +0.0053] | +0.073 [-0.004, +0.166] |
| wrists | 1--80 | **-0.0349 m [-0.0687, -0.0066]** | **-0.0278 m [-0.0570, -0.0024]** | -0.039 [-0.268, +0.148] |
| wrists | 81--140 | **-0.0225 m [-0.0359, -0.0111]** | **-0.0191 m [-0.0313, -0.0082]** | -0.157 [-0.285, -0.023] |
| wrists | 141--196 | **-0.0269 m [-0.0456, -0.0098]** | -0.0069 m [-0.0169, +0.0031] | +0.041 [-0.058, +0.132] |

Wrists 相对 Text-only 在三个长度段均稳定改善；相对 shuffled 的实例匹配在短、中序列稳定，长序列仍不确定。先前单 seed 分析中 MDM head 短序列的明显恶化没有跨 seed 稳定复现，因此论文应以本表的多 seed 结果为准。

## 论文结论

- 可以主张双腕 IMU 在多个 diffusion seeds 下提供稳定的实例级控制。
- 不能主张单头 IMU 已提供稳定实例级控制。
- 不能把 walking subset 上的 jerk 降低外推为通用平滑收益。
- 多 seed 结果增强了 wrists 主结论，同时进一步收窄了 head-only 主张。
