# ITM 主要实验数据汇总

## 1. 数据与实验协议

| 数据/协议 | 规模 | 用途 | 备注 |
|---|---:|---|---|
| HumanML3D/AMASS对齐训练集 | 7,009条 | MDM-ITM与IMU adapter训练 | HumanML文本/263D动作映射回AMASS并合成标准虚拟IMU |
| 对齐validation / test | 434 / 1,333条 | 模型选择、IMUPoser测试 | 标准虚拟IMU为30 FPS，MDM加载时重采样至20 FPS |
| IMU-mapped generation subset | 682条可用，实际672条 | MDM/MotionLab生成质量 | batch 32，5次重复；不是full-set 20-rep协议 |
| Counterfactual suite | 60 runs，770个生成case | 同文本换IMU、同IMU换文本、guidance与A-B matrix | 固定seed和初始噪声 |
| 通用控制子集 | 100条test动作 | Text-only/paired/zero/shuffled | head、wrists分开报告，单seed |
| 多seed控制子集 | 30条 × 5 seeds | 检查控制是否依赖扩散噪声 | 按短/中/长各10条分层 |
| 文本属性子集 | 30条test动作 × 6 prompts | 快慢、转向、摆臂、大步 | 固定head IMU、长度、seed和噪声 |
| Flexible IMUPoser test | 1,333条 | IMU-only重建边界 | head-only和双腕共享一个灵活mask模型 |
| Nymeria真实IMU pilot | 3条约5秒片段 | 零样本跨域诊断 | 另有5条因目标四元数无效排除，不进入主表 |
| SMPL round trip | 5条GT动作 × 200 iterations | 验证严格virtual-IMU评价能否成立 | 未通过acceleration/orientation门槛 |
| MotionLab V4预筛 | 512 train / 128 val | 对比条件辨识 | 2 epochs，未达70%门槛后停止 |
| HY-Motion Lite smoke | 5个固定prompt | 强基座和旋转输出工程验证 | 尚无Text+IMU训练结果 |

## 2. 冻结基座与可训练模块

| 系统 | 冻结基座 | 文本编码 | 可训练部分 | 输出 |
|---|---|---|---|---|
| MDM Text-only | 官方MDM `model000475000.pt` | CLIP ViT-B/32 | 无 | HumanML3D 263D，20 FPS，最长196帧 |
| ITM Stage-1/2b/4 | 同一官方MDM，8层、latent 512、1000 diffusion steps | 冻结CLIP ViT-B/32 | 2层IMU temporal Transformer及8个零初始化residual adapter | HumanML3D 263D/22 joints |
| Flexible IMUPoser | 无生成基座 | 无文本 | 2层、宽度512的双向LSTM | 24个SMPL关节6D旋转 |
| MotionLab ITM | 冻结MotionLab MotionFlow | 冻结CLIP | IMU encoder及原生512D control-token adapter | HumanML兼容动作 |
| HY-Motion Lite | 冻结0.46B、18层MMDiT（当前仅Text-only） | Qwen3 + CLIP资产 | 尚未训练IMU adapter | 201D、22关节6D旋转、根平移和WoodenMesh joints |

MDM Stage-1训练7,009条、5 epochs、batch 16、AdamW、学习率`1e-4`、text/IMU dropout
均为0.1。Stage-2b从Stage-1继续到epoch 8，学习率`5e-5`，加入trajectory、velocity、jerk
损失。训练过程中MDM和CLIP始终冻结。

## 3. 官方MDM参考结果

HumanML3D full test官方论文/随checkpoint日志协议：除MM运行5次外，其余运行20次。MM来自
本地保存的MDM论文表，当前项目的`wo_mm`日志没有重新计算它。

| Matching↓ | R@1↑ | R@2↑ | R@3↑ | FID↓ | Diversity→ | MM↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 5.5659±0.0270 | 0.3195 | 0.4978 | 0.6110 | 0.5443±0.0442 | 9.5595±0.0857 | 2.799±0.072 |

这里的Matching数值尺度与672子集表不同，不能跨协议直接比较绝对值。

## 4. 生成质量：MDM与ITM

相同672条IMU-mapped HumanML3D子集、5次重复：

| 模型 | 传感器 | Matching↓ | R@1↑ | R@2↑ | R@3↑ | FID↓ | Diversity→ | MM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MDM Text-only | 无 | 3.4152±0.0196 | 0.4530±0.0178 | 0.6506±0.0157 | 0.7586±0.0116 | **0.5089±0.0441** | 10.0517±0.1657 | 未评估 |
| ITM Stage-2b | head | **3.2680±0.0519** | **0.4580±0.0131** | **0.6682±0.0187** | **0.7780±0.0139** | 0.6526±0.0521 | 10.1686±0.1396 | 未评估 |
| ITM Stage-2b | wrists | 3.3546±0.0378 | 0.4551±0.0150 | 0.6598±0.0218 | 0.7673±0.0103 | 0.8167±0.0357 | 10.0319±0.1746 | 未评估 |

结论：head/wrists没有造成文本检索崩溃，但控制使FID变差，双腕的分布代价更明显。

## 5. IMU-only重建边界：Flexible IMUPoser

| 配置 | 测试条数 | MPJPE↓ | Rotation error↓ | Jerk ratio↓ | Foot skating↓ |
|---|---:|---:|---:|---:|---:|
| head-only | 1,333 | 14.91 cm | 0.278 rad | 0.294 | **0.123 m/s** |
| wrists | 1,333 | **10.74 cm** | **0.233 rad** | 0.455 | 0.308 m/s |

双腕通过预设12 cm门槛，单头没有通过。该表评价确定性IMU重建，不能与ITM的FID或文本
检索指标直接排名；较低jerk也可能意味着过度平滑。

## 6. MDM控制和平滑性

### 6.1 同一模糊文本，更换IMU

| 模型 | Active trajectory↓ | Acceleration proxy↓ | Acc. ratio↓ | Jerk ratio↓ | Arm swing | Root travel | Step frequency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-1 | 0.2871 m | 4.7340 m/s² | 3.5541 | 4.2072 | 0.1909 m | 2.1561 m | 1.5505 Hz |
| Stage-2b | **0.2833 m** | **4.5360 m/s²** | **2.0936** | **1.7599** | 0.1615 m | 1.4666 m | 1.4815 Hz |

Stage-2b的主要收益是显著降低抖动；不能仅凭该walking suite声称所有动作jerk都会改善。

### 6.2 按传感器拆分

| 模型 | 传感器 | Active trajectory↓ | Acceleration proxy↓ | Jerk ratio↓ | Root travel | Step frequency |
|---|---|---:|---:|---:|---:|---:|
| Stage-1 | head | 0.1733 m | 3.2895 m/s² | 4.5210 | 2.1826 m | 1.5833 Hz |
| Stage-2b | head | **0.1663 m** | **3.1365 m/s²** | **1.7407** | 1.5127 m | 1.4786 Hz |
| Stage-1 | wrists | **0.5147 m** | 7.6230 m/s² | 3.5796 | 2.1030 m | 1.4848 Hz |
| Stage-2b | wrists | 0.5172 m | **7.3352 m/s²** | **1.7981** | 1.3744 m | 1.4872 Hz |

### 6.3 固定head IMU，更换文本

| 模型 | Head trajectory↓ | Acceleration proxy↓ | Jerk ratio↓ | Arm swing↑ | Root travel | Step frequency |
|---|---:|---:|---:|---:|---:|---:|
| Stage-1 | **0.1838 m** | 3.7829 m/s² | **4.8521** | **0.2080 m** | 2.2811 m | 1.4899 Hz |
| Stage-2b | 0.2152 m | **3.2269 m/s²** | 5.6981 | 0.1581 m | 1.4263 m | 1.4094 Hz |
| Stage-4 text-anchor | 0.2103 m | 3.3283 m/s² | 5.9120 | 0.1666 m | 1.4565 m | 1.3961 Hz |

Stage-1文本补全更强但抖动高；Stage-2b更平滑但削弱摆臂；Stage-4只恢复一部分摆臂。

## 7. 严格损失消融

四个模型从同一Stage-1 checkpoint出发，数据顺序、seed、训练轮数和guidance一致；每个
变体覆盖60 runs、770 generated cases。

| Loss | Same-text trajectory↓ | Same-text jerk↓ | Same-head trajectory↓ | Same-head jerk↓ | Same-head arm swing↑ | Matrix trajectory↓ |
|---|---:|---:|---:|---:|---:|---:|
| Diffusion only | **0.2725 m** | 1.1782 | 0.1975 m | 2.3268 | **0.1108 m** | **0.3597 m** |
| + trajectory | 0.2764 m | **0.9989** | **0.1910 m** | **1.9784** | 0.0940 m | 0.3657 m |
| + trajectory + velocity | 0.2782 m | 1.0240 | 0.1932 m | 2.0913 | 0.0992 m | 0.3734 m |
| + trajectory + velocity + jerk | 0.2766 m | 1.0237 | 0.1922 m | 2.0219 | 0.1026 m | 0.3693 m |

Trajectory loss改善平滑性和same-head tracking，却降低arm swing；velocity和jerk项没有稳定
超过trajectory-only。辅助损失并未单调改善控制。

## 8. 统计稳健性

### 8.1 通用100-motion，Stage-2b paired相对参照

误差变化定义为`paired-reference`，负值更好；CI为20,000次配对bootstrap。

| 传感器 | 参照 | Error change，95% CI | 改善数，95% CI | p |
|---|---|---:|---:|---:|
| head | Text-only | -0.0110 m [-0.0204,-0.0010] | 60/100 [50%,69%] | 0.0261 |
| head | zero | -0.0045 m [-0.0132,+0.0041] | 54/100 [44%,64%] | 0.3052 |
| head | shuffled | -0.0017 m [-0.0086,+0.0055] | 51/100 [41%,61%] | 0.6446 |
| wrists | Text-only | -0.0196 m [-0.0284,-0.0103] | 70/100 [61%,79%] | <0.0001 |
| wrists | zero | -0.0101 m [-0.0182,-0.0019] | 65/100 [55%,74%] | 0.0155 |
| wrists | shuffled | **-0.0203 m [-0.0269,-0.0136]** | **74/100 [65%,82%]** | <0.0001 |

双腕达到预设70%实例改善门槛；head相对shuffled不显著。

### 8.2 30 motions × 5 diffusion seeds

| 传感器 | Paired-Text，95% CI | Paired-Shuffled，95% CI | Jerk change，95% CI | 多数seed优于Text | 多数seed优于shuffled |
|---|---:|---:|---:|---:|---:|
| head | -0.0038 m [-0.0108,+0.0031] | -0.0013 m [-0.0088,+0.0056] | +0.001 [-0.086,+0.087] | 13/30 | 15/30 |
| wrists | **-0.0281 m [-0.0414,-0.0160]** | **-0.0179 m [-0.0296,-0.0078]** | -0.052 [-0.147,+0.037] | **24/30** | **22/30** |

这是当前最强的实例级控制证据：双腕控制跨seed稳定，单头控制尚不稳定；通用jerk变化的
区间均跨零。

## 9. 文本属性控制

固定同一head IMU、长度、seed与初始噪声，只改变文本；30条test motions。

| 文本对比 | 指标差值 | 95% CI | 期望方向样本数 | 判断 |
|---|---:|---:|---:|---|
| Quickly - Slowly | -0.0199 m/s | [-0.1197,+0.1169] | 5/30 | 不成立 |
| Turn - Base | +2.3689 rad | [1.5691,3.1909] | 25/30 | 较稳定 |
| Swing arms - Base | **+0.0979 m** | **[0.0812,0.1144]** | **29/30** | 稳定 |
| Large steps - Base | -0.0237 m | [-0.0782,+0.0308] | 12/30 | 不成立 |

五种编辑prompt相对普通walk使head error平均增加0.0074 m，95% CI为[0.0031,0.0117]。

## 10. MotionLab跨基座迁移

### 10.1 672子集生成质量，5次重复

| 模型 | Matching↓ | R@1↑ | R@2↑ | R@3↑ | FID↓ | Diversity→ | MM |
|---|---:|---:|---:|---:|---:|---:|---:|
| MotionLab Text-only | **2.7223±0.0230** | **0.5292** | **0.7438** | **0.8321** | **0.2683±0.0217** | 9.8051 | 未评估 |
| MotionLab V2 + head | 2.8010±0.0348 | 0.5170 | 0.7280 | 0.8193 | 0.3173±0.0239 | **10.4175** | 未评估 |
| MotionLab V2 + wrists | 2.8025±0.0364 | 0.5196 | 0.7185 | 0.8205 | 0.3879±0.0472 | 10.1779 | 未评估 |

### 10.2 100-motion控制

| 配置 | Text error | Paired error↓ | Zero error | Shuffled error | Paired<Text | Paired<Zero | Paired<Shuffled | Text jerk | Paired jerk↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 0.1733 | 0.1548 | **0.1534** | 0.1618 | 60/100 | 42/100 | 63/100 | 2.319 | 2.005 |
| wrists | 0.3924 | **0.3615** | 0.3679 | 0.3774 | 63/100 | 54/100 | 67/100 | 2.319 | **1.971** |

MotionLab Text-only生成质量强于当前MDM，但V2控制没有达到70%实例改善门槛，且paired相对
zero不稳定，因此没有替换MDM Stage-2b。V4 condition matcher预筛的validation top-1由
56.4%升至61.9%，paired-beats-zero为96.9%，仍低于70%门槛，停止全量训练。

## 11. Nymeria真实IMU零样本预备实验

| 输入 | Active trajectory↓ | Root-relative error↓ | Jerk ratio↓ | Arm swing | Root travel |
|---|---:|---:|---:|---:|---:|
| MDM Text-only | 0.365 m | 0.294 m | 9.065 | 0.240 m | 0.402 m |
| ITM + head | **0.198 m** | **0.241 m** | **7.789** | 0.147 m | 0.290 m |
| ITM + wrists | 0.462 m | 0.344 m | 14.247 | **0.320 m** | **0.599 m** |

仅3条真实片段。Head显示初步零样本信号；wrists因真实Xsens与SMPL虚拟IMU的安装朝向、
噪声和统计域偏移而失败，不能作为正式跨数据集结论。

## 12. Virtual-IMU闭环验证

| 传感器 | 拟合成功 | Trajectory↓ | Acceleration↓ | Orientation geodesic↓ | 是否过门槛 |
|---|---:|---:|---:|---:|---|
| head | 5/5 | 0.00765 m | 0.8851 m/s² | 0.3201 rad | 否 |
| wrists | 5/5 | 0.00781 m | 2.5635 m/s² | 1.0325 rad | 否 |

虽然关节轨迹拟合低于1 cm，但加速度和方向都失败，说明HumanML 22 joints无法确定骨轴
twist。项目没有用降低门槛或挑样本的方式把该结果包装成正式generated-IMU指标。

## 13. HY-Motion与A40阶段状态

| 项目 | 数据/设置 | 结果 | 状态 |
|---|---|---|---|
| HY-Motion Lite Text-only smoke | 60帧，seed 1234，关闭prompt rewrite | 核心采样1.93 s，PyTorch峰值约19.6 GB；输出`[1,60,201]`、22关节6D旋转和52 joints | 在4090阶段完成 |
| 固定prompt定性 | walk/run/sit/turn/jump，共5条 | 坐下最清楚；跑和转向可见；当前jump seed较弱 | 工程/定性验证 |
| Rotation→virtual IMU | head 15，wrists 20/21 | 全局旋转正交误差约`1.5e-5`，行列式接近1 | joint-mounted proxy已打通 |
| 早期hidden adapter | 512 train、2 epochs、128条held-out | paired>shuffled为74.22%/82.03%（head/wrists），但jerk为Text-only的6.82x/5.71x | 控制可辨识，但因jerk失败 |
| Output-velocity controller | 512 train、4 epochs；全新indices 128--191 | paired>shuffled为78.13%/92.19%；jerk为Text-only的0.9893x/0.9901x | 控制和严格均值门槛通过 |
| 预注册2% jerk非劣检验 | 同一全新64条；97.5% bootstrap上界 | head上界1.0421 m/s^3，高于0.9945边界0.0476；wrists通过 | 联合门槛失败，不授权7,009条训练 |

这些结果证明paired IMU能稳定改变HY-Motion生成，而不再出现早期hidden adapter的平均
jerk恶化。但最新预注册联合门槛仍失败，semantic gate也尚未正式量化，因此
`full_training_allowed=false`。详见
[HY-Motion实验总结](HYMOTION_EXPERIMENT_SUMMARY_2026-09-23.md)。

## 14. 早期回归基线（仅作lower bound）

| 模型/输入 | 数据 | MSE↓ | MAE↓ | 结论 |
|---|---|---:|---:|---|
| Linear IMU-only | 999 train / 200 test | **0.024277** | **0.082120** | synthetic IMU在确定性回归中很强 |
| Linear Text-only | 同上 | 0.062518 | 0.137824 | hash文本泛化差 |
| Frame MLP IMU-only | 999 train / 200 val | **0.014230** | **0.061955** | 优于线性IMU-only |
| Frame MLP Text+IMU | 同上 | 0.020078 | 0.078787 | hash文本降低train loss但恶化validation |
| Temporal Transformer IMU-only，6 sensors | test | 0.018271 | 未统一报告 | 文本未提供增益 |
| Temporal Transformer DistilBERT+IMU，6 sensors | test | 0.018774 | 未统一报告 | 促使项目转向预训练生成基座 |

这些MSE基线的任务是逐帧/序列重建，不进入最终生成模型主表。
