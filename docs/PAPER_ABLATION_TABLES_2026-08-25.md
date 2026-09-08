# ITM 论文消融实验汇总

## 1. 严格训练损失消融

四个变体均从同一个 `stage1_full_pilot_v2.pt` 恢复，使用相同数据顺序、随机种子、训练轮数、head/wrists 配置和 legacy guidance。每个变体均完成 60 个 run、770 个 generated cases。因此，这组结果比 Stage-1/Stage-2b/Stage-4 的阶段性比较更适合回答“某项辅助损失本身带来了什么”。

### 同文本换 IMU

| Loss | Active trajectory (m) | Acc. proxy (m/s^2) | Jerk ratio | Arm swing (m) |
| --- | ---: | ---: | ---: | ---: |
| Diffusion only | 0.2725 | 3.3448 | 1.1782 | 0.1017 |
| + trajectory | 0.2764 | 3.4035 | **0.9989** | 0.0946 |
| + trajectory + velocity | 0.2782 | 3.4283 | 1.0240 | 0.0976 |
| + trajectory + velocity + jerk | 0.2766 | 3.4837 | 1.0237 | 0.0990 |

### 同 head IMU 换文本

| Loss | Active trajectory (m) | Acc. proxy (m/s^2) | Jerk ratio | Arm swing (m) |
| --- | ---: | ---: | ---: | ---: |
| Diffusion only | 0.1975 | 2.7279 | 2.3268 | **0.1108** |
| + trajectory | **0.1910** | **2.5928** | **1.9784** | 0.0940 |
| + trajectory + velocity | 0.1932 | 2.6793 | 2.0913 | 0.0992 |
| + trajectory + velocity + jerk | 0.1922 | 2.6822 | 2.0219 | 0.1026 |

### Text/IMU A-B matrix

| Loss | Active trajectory (m) | Acc. proxy (m/s^2) | Jerk ratio | Arm swing (m) |
| --- | ---: | ---: | ---: | ---: |
| Diffusion only | **0.3597** | **4.0718** | 1.5601 | 0.1490 |
| + trajectory | 0.3657 | 4.2162 | **1.5598** | 0.1476 |
| + trajectory + velocity | 0.3734 | 4.2400 | 1.5858 | 0.1529 |
| + trajectory + velocity + jerk | 0.3693 | 4.2810 | 1.5717 | **0.1534** |

### 结论

- 辅助损失没有普遍降低 active trajectory 或 acceleration proxy；在同文本与 matrix 协议中，这两项反而略有恶化。
- `trajectory` 单项在同文本和同 head 协议中取得最低 jerk，并改善同 head tracking，但 arm swing 从 0.1108 m 降至 0.0940 m。
- 显式加入 velocity 和 jerk loss 没有稳定超过 trajectory-only。
- 因而数据支持的是“平滑性与未观测部位文本补全存在 trade-off”，而不是“加入更多辅助损失必然改善控制”。

原始汇总位于 `outputs/mdm_control/loss_ablation/LOSS_ABLATION_SUMMARY.md`。

## 2. Guidance 消融

Legacy guidance 为：

```text
f00 + text_scale * (f10 - f00) + imu_scale * (f11 - f10)
```

Factorized guidance 为：

```text
f00
+ text_scale  * (f10 - f00)
+ imu_scale   * (f01 - f00)
+ joint_scale * (f11 - f10 - f01 + f00)
```

在 `imu_scale=1, joint_scale=1` 时，factorized 的后两项相加正好等于 `f11-f10`，因此与 `imu_scale=1` 的 legacy guidance 代数等价。现有 10-motion smoke 也显示 paired head/wrists 指标仅有浮点级差异。这不是性能改进实验，而是条件可辨识性实验：factorized 模式能够严格构造 `f00`、Text-only `f10`、IMU-only `f01` 和联合条件 `f11`。

严格关闭 IMU 的方式为：

- Legacy：`imu_scale=0`。
- Factorized：`imu_scale=0, joint_scale=0`。

若 factorized 模式只设置 `imu_scale=0` 而保留 `joint_scale=1`，Text-IMU 交互项仍然存在，不能称为 Text-only。论文中的统一 100-motion control evaluation 已使用修正后的严格条件。

## 3. 论文使用建议

- 主文放严格 loss ablation 的精简表，并报告上述负结果。
- Guidance 在方法或附录中说明公式和严格开关；不声称四分支本身提升性能。
- Stage-2b 继续作为当前 MDM 主 checkpoint，因为其选择依据是完整阶段实验的综合平滑性，而不是严格消融中每一项均单调改善。
- 所有 acceleration 数值继续标注为 joint acceleration proxy，不称为真实 generated-IMU acceleration error。

## 4. 100-motion joint-scale sweep

已使用固定的 MotionLab/MDM 共享 100-motion test subset 生成以下协议：

```text
sensor_config in {head, wrists}
joint_scale   in {0, 0.25, 0.5, 1, 1.5}
text_scale    = 2.5
imu_scale     = 1.0
seed          = 1234
```

只改变 factorized guidance 的 Text-IMU interaction scale；Text-only reference 复用已完成的严格 100-motion 输出。spec 位于：

```text
outputs/mdm_control/joint_scale_test100/specs/manifest.json
```

运行命令：

```bash
conda run --no-capture-output -n itm python scripts/run_mdm_control_conditions.py \
  --manifest outputs/mdm_control/joint_scale_test100/specs/manifest.json \
  --control-checkpoint outputs/mdm_control/stage2b_balanced_b.pt \
  --mdm-args outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json \
  --standard-imu-manifest outputs/manifests_full/test_standard_imu.jsonl \
  --output-dir outputs/mdm_control/joint_scale_test100 \
  --device cuda:1

conda run --no-capture-output -n itm python scripts/evaluate_mdm_joint_scale_sweep.py \
  --manifest outputs/mdm_control/joint_scale_test100/specs/manifest.json \
  --results-dir outputs/mdm_control/joint_scale_test100 \
  --text-reference-dir outputs/mdm_control/control_test100 \
  --text-reference-manifest outputs/mdm_control/control_test100/specs/manifest.json \
  --output outputs/mdm_control/joint_scale_test100/summary.json
```

runner 会跳过已经完成的 `.npz` chunk，可以在中断后继续。本次 10 groups、50 chunks 均已在 GPU 1 完成，无失败或跳过。

| Sensor | Joint scale | Active error (m) | Improvement vs Text (m), 95% CI | Improved | Jerk ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 0 | 0.2144 | +0.0101 [+0.0007, +0.0191] | 61/100 | 1.032 |
| head | 0.25 | 0.2143 | +0.0102 [+0.0009, +0.0192] | 61/100 | 1.045 |
| head | 0.5 | 0.2150 | +0.0095 [-0.0004, +0.0187] | 61/100 | 1.047 |
| head | 1 | 0.2136 | +0.0110 [+0.0010, +0.0205] | 60/100 | 1.041 |
| head | 1.5 | **0.2131** | +0.0114 [+0.0017, +0.0208] | 58/100 | **1.006** |
| wrists | 0 | **0.4833** | **+0.0217 [+0.0129, +0.0303]** | **71/100** | 0.961 |
| wrists | 0.25 | 0.4846 | +0.0204 [+0.0114, +0.0293] | 69/100 | 0.957 |
| wrists | 0.5 | 0.4851 | +0.0199 [+0.0106, +0.0288] | **71/100** | 0.967 |
| wrists | 1 | 0.4854 | +0.0196 [+0.0101, +0.0285] | 70/100 | 0.951 |
| wrists | 1.5 | 0.4868 | +0.0182 [+0.0087, +0.0275] | 70/100 | **0.937** |

与默认 `joint_scale=1` 做逐样本配对 bootstrap 后，其他四档的 active error 和 jerk 差异的 95% CI 均跨零，sign-flip p-value 均大于 0.2。虽然 wrists 在 `joint_scale=0` 时有最低平均 tracking error，差值仅为 -0.0022 m，95% CI 为 [-0.0058, +0.0015]，不足以支持更换默认设置。

该实验表明：

- head 和 wrists 在多数档位相对严格 Text-only 都有正的平均 tracking 改善；wrists 的证据更稳定。
- `joint_scale` 从 0 扫到 1.5 时，输出动作差异和指标变化都很小。
- 当前控制主要来自 factorized guidance 的 IMU main effect；额外 Text-IMU interaction 没有形成强且稳定的增量。
- 默认 `joint_scale=1` 保持不变，避免依据不显著的均值差异调参。

完整逐样本结果、置信区间和随机化检验位于 `outputs/mdm_control/joint_scale_test100/summary.json` 与 `summary.md`。
