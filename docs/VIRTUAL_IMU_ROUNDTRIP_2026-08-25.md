# HumanML Joints to SMPL Virtual-IMU Round-trip

## 目的

验证是否可以把MDM生成的HumanML 22 joints拟合到neutral SMPL，再从SMPL顶点和全局旋转导出虚拟IMU，用于严格评价输入与生成动作的acceleration和orientation consistency。

按照评价门槛，必须先在GT joints上通过round-trip，才允许将该流程用于生成结果的论文指标。

## 协议

- 数据：统一100-motion control subset中的前5条不同GT动作。
- SMPL：neutral v1.1.0。
- 优化：每条完整序列200 iterations。
- Shape：neutral fixed shape，同时优化global orientation、body pose、translation和global scale。
- Virtual IMU：与标准缓存相同的6槽传感器顶点、关节方向、20 FPS和5帧平滑。
- 分别使用head和wrists active sensor mask。

门槛：

| Metric | Gate |
| --- | ---: |
| Active trajectory | <=0.05 m |
| Acceleration | <=0.5 m/s2 |
| Orientation geodesic | <=0.2 rad |

## 结果

| Sensor | Successful | Trajectory | Acceleration | Orientation | Eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| head | 5/5 | 0.00765 m | 0.8851 m/s2 | 0.3201 rad | no |
| wrists | 5/5 | 0.00781 m | 2.5635 m/s2 | 1.0325 rad | no |

五条拟合均成功，joint trajectory误差低于1 cm，但head和wrists的acceleration与orientation均未通过门槛。Wrists显著更差，符合22个关节位置无法确定前臂绕骨轴twist的几何欠定性。Acceleration还会放大逐帧位置拟合的微小误差，因此即使trajectory很低也不能直接得到可靠二阶信号。

## 决策

- 不对MDM/Stage-2b生成结果报告“真实virtual-IMU orientation consistency”。
- 论文继续把现有active-joint acceleration明确称为joint-derived proxy。
- 不通过降低门槛或只挑成功序列制造可用结果。
- 严格orientation评价应迁移到直接输出局部关节旋转的生成表示，或使用带方向监督的SMPL fitting；仅从HumanML joints后处理无法可靠完成。

原始输出：

```text
outputs/mdm_control/virtual_imu_roundtrip/gt_5x200.json
outputs/mdm_control/virtual_imu_roundtrip/gt_wrists_5x200.json
```
