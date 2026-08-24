# ITM迁移前兼容优化与证据闭环

## 完成状态

迁移MotionLab/HY-Motion前计划中的代码工作已经完成：统一backbone接口、条件缓存、变长batch、legacy/factorized guidance、CFG合批、严格loss消融runner和SMPL/virtual-IMU离线评价均已实现并通过测试。正式四变体loss消融已经完成训练和60-run test suite。

## 兼容与性能

- 历史实验继续默认`legacy + sequential`，Stage-2b checkpoint和历史结果定义不变。
- MDM的CLIP text embedding和IMU control各缓存一次，不再在1000个扩散step中重复编码。
- 单样本相同seed下，缓存与未缓存motion逐元素一致，最大绝对误差为0。
- 单样本诊断时间：未缓存legacy约6.67秒，缓存legacy约5.05秒。
- factorized四分支支持`batched`执行；单样本诊断约4.12秒，但实际收益取决于batch大小和显存带宽。
- 变长batch改为最大长度padding，metrics和浏览器JSON按每个case的`effective_lengths`截取。

## 四分支Guidance

新增：

```text
f00 = unconditional
f10 = text-only
f01 = IMU-only
f11 = text+IMU

output = f00
       + text_scale  * (f10 - f00)
       + imu_scale   * (f01 - f00)
       + joint_scale * (f11 - f10 - f01 + f00)
```

严格单模态条件必须关闭交互项：

- Text-only：`imu_scale=0, joint_scale=0`
- IMU-only：`text_scale=0, joint_scale=0`
- Text+IMU：`joint_scale=1`

修正后的8-case factorized matrix已写入：

```text
outputs/mdm_control/guidance_ablation/factorized_matrix/
```

10样本same-text诊断中，`imu_scale=1, joint_scale=1`的联合条件与legacy按公式基本等价，因此两者主要指标几乎一致。这不是新方法提升证据；四分支的主要价值是提供严格IMU-only分支和独立interaction消融。

## 严格Loss消融

所有模型均从`stage1_full_pilot_v2.pt`继续训练，使用相同seed、7009条训练记录、head/wrists配置、epoch 8、学习率和legacy guidance。每个变体完成60 runs、770 cases。

输出：

```text
outputs/mdm_control/loss_ablation/
  diffusion.pt
  trajectory.pt
  trajectory_velocity.pt
  full_regularized.pt
  experiments_*/RESULTS_SUMMARY.md
  LOSS_ABLATION_SUMMARY.md
  LOSS_ABLATION_SUMMARY.json
```

### Same text, different IMU

| Variant | Active trajectory (m) | Acc. proxy (m/s2) | Jerk ratio | Arm swing (m) |
| --- | ---: | ---: | ---: | ---: |
| diffusion | 0.2725 | 3.3448 | 1.1782 | 0.1017 |
| +trajectory | 0.2764 | 3.4035 | 0.9989 | 0.0946 |
| +velocity | 0.2782 | 3.4283 | 1.0240 | 0.0976 |
| +jerk | 0.2766 | 3.4837 | 1.0237 | 0.0990 |

### Same head IMU, different text

| Variant | Active trajectory (m) | Acc. proxy (m/s2) | Jerk ratio | Arm swing (m) |
| --- | ---: | ---: | ---: | ---: |
| diffusion | 0.1975 | 2.7279 | 2.3268 | 0.1108 |
| +trajectory | 0.1910 | 2.5928 | 1.9784 | 0.0940 |
| +velocity | 0.1932 | 2.6793 | 2.0913 | 0.0992 |
| +jerk | 0.1922 | 2.6822 | 2.0219 | 0.1026 |

结论：辅助loss主要降低jerk，而不是稳定改善active sensor tracking。same-text中trajectory error反而增加约1%到2%，acceleration proxy增加约2%到4%；同时arm swing下降。此前Stage-1/Stage-2b的跨阶段比较不能替代这一严格消融。论文应将其表述为平滑性与动作幅度/文本补全之间的trade-off，不应声称这些loss普遍提升IMU一致性。

## SMPL与Virtual IMU有效性

HumanML 22 joints已可离线拟合到neutral SMPL，并从SMPL顶点和全局旋转重新生成6槽virtual IMU。评价脚本要求先运行GT round-trip；未通过时自动写入：

```text
eligible_for_paper_claims = false
```

一条GT序列、50次优化的初测：

| Metric | Result | Gate | Pass |
| --- | ---: | ---: | --- |
| active trajectory | 0.0377 m | 0.05 m | yes |
| acceleration | 1.4093 m/s2 | 0.5 m/s2 | no |
| orientation | 1.0564 rad | 0.2 rad | no |

22个关节位置不能唯一确定肢体绕骨轴的twist，因此MDM/HumanML输出无法可靠恢复完整传感器方向。当前论文继续使用明确标注的joint-derived acceleration proxy。正式orientation consistency应等迁移到直接输出关节旋转的基座后再做，HY-Motion在这一点上比MDM更适合。

## 验证

- 全部测试：`59 passed`
- 128-record full-regularized smoke：loss finite，checkpoint可采样。
- 正式loss ablation：4 checkpoints、240 runs全部完成。
- legacy缓存等价：最大绝对误差0。
- factorized matrix：Text-only、IMU-only、Text+IMU开关已核对。

## 下一步迁移门槛

1. 先实现MotionLab Text-only adapter并复现固定prompts和HumanML3D评价，不立即接IMU。
2. 使用同一backbone-neutral输出、672子集和condition-swapping suite比较MDM与MotionLab。
3. MotionLab Text-only达标后，再把IMU encoder接入其`text_hint`路径。
4. 在显存充足宿主机复现HY-Motion Full；优先验证旋转输出和virtual-IMU round-trip。
5. 新基座若未同时改善文本生成和IMU控制，不替换当前Stage-2b论文主模型。
