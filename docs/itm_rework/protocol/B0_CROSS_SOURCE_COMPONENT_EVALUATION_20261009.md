# B0 跨来源 IMU 主项/交互项与误差 CPU 评价（2026-10-09）

**原480条F/U保存轨迹、240项配对全部valid，七条件和全部120完整输出参考保留。IMU主项的动态/误差效应随来源、配置及状态改变；不能依据主弓步片段的收缩直接删除整个主项。** F在训练来源paired状态24/24改善root＋RIC MSE，在验证来源则17/24变差；验证来源20/24增加平均姿态误差。它没有统一降低时间能量。下一项最小实验选择检查原latent control的时间常量分量，先做48个同状态预测探针，不训练、采样或解码完整动作。原F及已完成main-knockout的not_supported结论不变。

## 身份与完整性

[冻结计划](b0_cross_source_component_CPU_20261009_v1/plan.json) SHA `49cf5ad9f5738aa653ef07ababa6100e5027e7e3892aad6f4e450a34efb7f9de`，17906绑定来源未改。人口：6原来源/12配置、F/U、原t900/500/250/50，paired96、text_only96、physical_static96、IMU-only48、backbone_bypass48、numeric_zero48、tail_reversed48。paired F/U配对48项，总240项。不同配置/时间不是新的独立来源。

[计算implementation v1](../../../outputs/itm_rework/method/cross_source_component_implementation_20261009_v1/implementation.json) SHA `70ccba55f0b18def0f378420ea0c3d4452c275e24ccd2c6f96fde60f3929613f`，17911来源；9新增合成身份/权重/时间/返回值/输入/形状/掩码/有限性/配对及缺失分母tests通过。全部原raw、metadata、真实长度、float32六预测及完整mask核对，GT/MeanStd和原11视图/balanced不变。

[全部计算结果](../../../outputs/itm_rework/protocol/cross_source_component_evaluation_20261009_v1/artifact_index.json)480行、240个F−U差值全部valid，无缺失；每一行以原状态上的a=f10、u=f01−f00、j=f11−f10−f01+f00构造float64 a+u/a+j。使用原temporal_decomposition及same_state_delta，计算平均姿态/时间误差、能量及有符号交叉项。假设数组是CPU代数，不是新native returns或rollouts。原F/U的状态不同，不把F−U解读为同状态权重因果。

首个第二公式入口在Python3.10解析`table[*key,arm]`时失败，未执行数学核验；[v1失败记录](../../../outputs/itm_rework/method/cross_source_component_implementation_20261009_v1/first_verification_failure.json)及原代码/manifest保持。[兼容v2](../../../outputs/itm_rework/method/cross_source_component_implementation_20261009_v2/implementation.json) SHA `86bdfd465e97717f6ea6c4ea84a9562e8782da7472f59bd9228d75acd598fc82`，17922来源，只用显式tuple修正索引语法和绑定原结果，未重算/替换旧结果或放宽浮点界。

[最终第二公式证明](../../../outputs/itm_rework/method/cross_source_component_implementation_20261009_v2/result_verification.json)独立从原数组用dot公式核验全480行、240差值及等来源聚合：**602400标量通过**，含1008项null一致性检查，最大绝对差7.275958e−12（含较大的GT归一化比值），全部满足原 `64eps×context scale`。原始单位与GT比值分别用对应运算量纲的context scale，没有扩大科学误差界。112个row/view组合的GT能量比值为null，未变成0或删除；每个聚合字段保留planned_n/valid_n。

全阶段GPU查询/模型构造或前向/噪声/采样/解码/训练为0，未改论文或行为标签，主会话自检不是独立02/人类/盲审。

## 来源差异：能量、平均姿态和时间误差需要同时看

[24项source/config/arm表](../../../outputs/itm_rework/protocol/cross_source_component_panel_20261009_v1/paired_source_tables.json)、[全部七条件符号统计](../../../outputs/itm_rework/protocol/cross_source_component_panel_20261009_v1/condition_sign_counts.json)、[能量−误差图](../../../outputs/itm_rework/protocol/cross_source_component_panel_20261009_v1/paired_component_energy_error.png)。以下均是严格零符号描述，没有新成功阈值、p值或把四个t当独立样本。

| paired来源/臂 | 状态数 | 主项降低时间能量 | 主项降低MSE | 主项降低平均姿态误差 | 主项降低时间误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F/train | 24 | 14 | 24 | 21 | 24 |
| F/val | 24 | 8 | 7 | 4 | 12 |
| U/train | 24 | 16 | 22 | 16 | 24 |
| U/val | 24 | 8 | 5 | 7 | 12 |

整体F paired的主项22/48降低时间能量、31/48降低MSE；U为24/48与27/48。两臂时间误差均36/48下降。raw mean/temporal分解保留有符号交叉项，不按能量大小分配“机制百分比”。训练/验证同时包含不同动作与轨迹状态，不能由这些计数证明唯一过拟合原因或泛化显著性。

F paired关键例（归一化root＋RIC特征的有符号变化，主项单独添加至f10）：

| 来源/配置 | t | 时间能量变化 | MSE变化 | 平均姿态误差变化 | 时间误差变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| val000382/wrists 拍手 | 50 | 0.006159 | 0.265703 | 0.258816 | 0.006887 |
| val000139/head 顺序 | 50 | -0.010355 | 0.034706 | 0.053393 | -0.018687 |
| val000433/wrists 踢腿返回 | 250 | 0.004328 | -0.006418 | 0.009589 | -0.016007 |
| train000506/wrists 弓步 | 50 | 0.000662 | -0.042515 | -0.020207 | -0.022308 |
| train004728/wrists 静止持物 | 900 | 0.297122 | -0.219074 | -0.065069 | -0.154004 |

拍手片段的主项增加误差且主要体现为平均姿态误差；顺序片段head降低时间误差但增加平均姿态误差，总MSE仍变差。主弓步wrists晚段增加能量且降低误差，静止持物来源也可增加预测能量并改善时间误差。因此“主项统一压低动态”“删主项统一改善”都不成立；静止标签不代表全部HumanML特征GT能量严格为零。GT mean/variance不是conditional expectation或动作正确性证明。

交互项的MSE方向也随来源/t改变：F26/48、U25/48为负；其MSE变化绝对值在这96个paired状态中均小于同状态主项，但两项相关，不能视为独立贡献或证明交互无效。只有条件代数与保存预测的证据，没有新增动作成功结果。

## 对照与负例

text-only96、physical-static96、bypass48的root＋RIC主项及交互MSE变化均精确为0；IMU-only48的root＋RIC交互变化也精确为0，保留其非零主项。numeric_zero48保持实际encoder下的非零效应（不是严格不存在条件），最大|主项MSE变化|0.274146；tail_reversed48最大0.281199，不合并到paired或忽略。原clap/kick-return/顺序/持物负例、120完整输出及既有七条件审阅都保持；数据准入、floor/contact N/A与判据未改。等来源汇总保留split/config/condition/time/arm，原长度不按帧数加权。

## 最小下一实验：48个同状态控制均值探针

[新冻结plan](b0_control_mean_probe_20261009_v1/plan.json) SHA `b3fe884ebc53b71b049ff02e4e8dc498eae2ff9173ea58d289433b53b3fd39d9`，17967来源、12case/48state/240软件检查，3角色/时间/失败来源冻结tests通过。[冻结来源/人口证明](../../../outputs/itm_rework/method/cross_source_component_implementation_20261009_v2/next_probe_verification.json)。仅设计冻结，尚未实现或GPU执行。

保留F200、原encoder与static-delta observed−physical-static控制C、所有原输入、text/IMU presence true、factorized(1,1,1)、caption batch1/MDMbatch4。仅在原有效帧上计算原float32控制时间均值μ_C，替换为C−μ_C。保留时间变化分量，不改原raw IMU/GT归一化或给预测做拟合。原latent DC与解码姿态mean不必线性对应，它可含有用姿态信息，故保留全部6来源/12配置而非只挑有害验证片段。

在每个原F paired x_t/t上重新预测，范围原900/500/250/50，共48前向/192examples、encoder24/caption12，无noise/采样/解码/训练。原未改control/caption/cache输入核验；每状态x_t bitwise复制、f00/f10与旧原预测在原1e−6 abs/relative下相等、unitCFG核对；每case控制均值/重构/zero保留核对，总240checks。raw先保存，首次故障停止全部队列，失败/缺失分母保留。

实现并CPU验收后，只执行一次新**600秒绝对期限/24GiB cap**队列，不重试/恢复；优先物理cuda:1 UUID `GPU-43c2f925-db05-c655-bb46-25196d0627d6`，长期自主GPU授权有效，3样本间隔2秒/free≥4096MiB，利用率仅记录无上限，不干预他人。完成同状态特征/mean/temporal/error评价后再判断是否值得完整轨迹或方法训练；探针本身不能证明动作修复。

[交付证明](../../../outputs/itm_rework/method/cross_source_component_implementation_20261009_v2/delivery_verification.json)、[归档](../method/cross_source_component_run_20261009_v1/artifact_index.json)。当前无活动队列；原F/新main-knockout仍not_supported。
