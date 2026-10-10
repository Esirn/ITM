# 头部局部监督候选：完整训练与生成评价

一次完整比较已经关闭。候选改善头部位置误差约5.2%，但头部速度没有改善，全身全局/根轨迹代价较大。**当前不以该候选替换原U100，不将这组辅助损失加入最终Method；作为开发对照完整保留。** 头部论文主线继续推进，不继续扫描此损失的系数。

## 实际完成与公平对照

[训练计划](head_primary_training_20261010_v1/plan.json)保持原U100初始权重、seed、optimizer、100epoch/25600曝光及真实长度16图累积。head12800/wrists12800曝光顺序、扩散时间步和噪声地址不变，仅在head且IMU存在的11528曝光启用0.1归一化局部位置＋0.1局部速度损失；wrists和IMU缺失曝光的梯度仍对应原263维目标。GT仅作训练监督，尺度仅来自128train/14876唯一帧。worker新增5项CPU测试通过，原损失/尺度CPU验收复用。

[完成训练](../../../outputs/itm_rework/method/head_primary_training_authorized_20261010_v2/queue_status.json)：1600updates/25600MDMcalls/51200encoder/160caption完成，401.902434秒，peak691351552bytes，冻结骨干逐参数校验不变。最终checkpoint SHA `77e97455827a4914a811d0f14bac4f015c580a2934d471ecb7e07881b6207fbb`，仅最终epoch，无val挑checkpoint。

[采样计划](head_primary_sampling_20261010_v1/plan.json)、[完成采样](../../../outputs/itm_rework/method/head_primary_sampling_authorized_20261010_v1/queue_status.json)：新增11条head完整1000步DDPM（8val＋3train），11000MDMcalls/22encoder/11caption/11decode，83.736695秒，peak531765760bytes；textCFG2.5/IMU1/joint1、原11noise groups保持。采样选择新增2项CPU测试通过。复用原U100、MDM和两简单基线的83个头部输出，不重采参考、不新增双腕生成。

[CPU比较](../../../outputs/itm_rework/protocol/head_primary_evaluation_20261010_v1/summary.json)：11独立解码、242项主要/局部指标第二公式与代数核验、11噪声签名一致；全部1600receipt/25600曝光与旧U的序号/真实帧数/噪声SHA一致，11528辅助启用范围检查通过。无缺失。原8val、32val简单基线与3train参考分别保存。

首次训练尝试v1在0.0294秒的预检停止，`nvidia-smi`返回9，模型调用0；沙箱没有`/dev/nvidia*`。沙箱外只读查询同UUID成功后，保留v1失败，创建v2以同冻结计划运行，无恢复/拼接或科学参数变更。资源访问修正见[记录](head_primary_training_20261010_v1/sandbox_failure_resolution.json)。物理cuda:1三个预检均free8660MiB/util97%，已有进程不拒绝；UUID锁、3600秒整队上限、24GiB allocated cap和首故障停止保持，不修改无关进程。

## 同8val来源的头部比较

每完整片段等权。位置、全身/根误差单位m，速度m/s，jerk m/s³。

| 方法 | 头部局部位置 | 头部局部速度 | 全身相对姿态 | 全身global | 根轨迹 | 头部局部jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MDM textCFG2.5 | 0.120393 | 0.323916 | 0.161450 | 0.671622 | 0.691515 | 26.011718 |
| **原U100（当前主线）** | **0.107872** | **0.293157** | **0.150016** | **0.520770** | **0.514922** | **38.422223** |
| head局部监督候选 | 0.102256 | 0.298389 | 0.154492 | 0.686713 | 0.718928 | 37.767572 |
| 简单文本＋IMU100 | 0.095365 | 0.354529 | 0.127223 | 1.003387 | 1.120423 | 109.693511 |
| 简单IMU-only100 | 0.090704 | 0.347229 | 0.116678 | 0.817521 | 0.919782 | 119.180863 |

相对原U：头部位置−5.21%（6/8来源改善）、速度+1.78%（仅2/8改善）、全身相对姿态+2.98%、global+31.87%、root-path+39.62%；局部jerk−1.70%、全身jerk−3.24%。位置改善更多体现于时间平均偏差（0.067791→0.060705m），动态误差RMS变化小（0.085136→0.084558m）。不能把它称为动态控制修复。

相对MDM，候选头部位置−15.07%、速度−7.88%、全身相对姿态−4.31%，但global+2.25%、root-path+3.96%、局部jerk+45.19%。原U对MDM的global改善更强，局部速度也更好。两简单基线仍在头部位置与全身相对姿态上较强，不能仅用MDM列宣布整体优越。

局部幅度没有统一收缩：部分val头部速度比GT过大，也有个别参考收缩；004728训练参考候选head空间变化/速度比0.315/0.320，原U约0.364/0.351。原弓步失败和旧U/F用户正面观感不修改，本轮不赋予新行为成功标签。

[静态比较图PNG](../../../outputs/itm_rework/protocol/head_primary_review_20261010_v1/head_comparison.png)及[PDF](../../../outputs/itm_rework/protocol/head_primary_review_20261010_v1/head_comparison.pdf)已生成并目视核对，不依赖前端。图只是开发误差对照，不是动作语义/自然性认证。

## 研究决策

保留原U100作为阶段头部工作模型；候选不替换、不扫系数，辅助监督有效性的最终论文主张暂不成立。下一交付转为补齐原U100/MDM在全部32val来源的head结果，已有8来源复用，仅新增其余24来源×两方法48完整采样。两简单基线已有各32个head输出。扩大来源覆盖，避免一直围绕同8个开发例子判断。

官方IMUPoser head接口在[补充核查](../HEAD_IMUPOSER_INTERFACE_ADDENDUM_20261010.md)发现12维模型/60维数据加载不一致，需明确适配、canonical旋转及公平SMPL解码；不伪称仅改路径即可复现。后续按头部最低基线、条件贡献/核心结构消融、文本语义评价、正式test/多种子推进。双腕不阻挡主线；论文LaTeX和前端均未修改。

所有结果仍是单seed/K1、8val开发比较；新caption未人工全审，严格传感器朝向/顶点加速度闭环、文本语义、FID与行为成功尚未认证。root/global虽为辅助列，本轮仍公开其较大代价。
