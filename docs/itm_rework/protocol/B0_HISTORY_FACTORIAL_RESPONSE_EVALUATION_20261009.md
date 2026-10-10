# B0 全程历史×后段条件响应 CPU 评价（2026-10-09）

**新T-prefix 3600行与已验证F-prefix 3600行已完整分析，7200有效单元、3600历史内路径对照、1800逐步历史交互全部valid。** wrists在T历史下paired路径的实际预测时间能量900步均低于text路径，但同一paired状态上的局部IMU效应有616步为负、284步为正。head的局部IMU效应在两种历史下均900步为负，而T历史路径差异从t145起为正。不能把局部能量符号等同于长期轨迹效果或完整行为修复。原F及四个新T后段输出仍incomplete/not_supported，论文主张不变。

## 身份、分母与验证

- [冻结CPU计划](b0_history_factorial_response_CPU_20261009_v1/plan.json) SHA `2ac7d1f050970a53ab153ea6fe351b5a0d5095bb424429c983b8b5a23b5ef8fc`，17813来源，未改。
- [本阶段implementation](../../../outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1/implementation.json) SHA `e39985bd594fd0c3a976f85d1d39b3d868a0ec460b86ac6b2040c945d8f08e15`，17819依赖。显式验证并绑定前次软件proof中的verifier，补闭合嵌套来源。
- 新T真实保存数组3600行，F3600行直接复用原CPU结果及4246200项第二公式证明。没有重算旧F数学或重复旧GPU队列。6预测数组均保留原float32形状(1,263,1,136)及完整bool mask，归一化GT与Mean/Std沿用原绑定文件。
- 全部原11特征视图及balanced字段、6预测的temporal分解、四组件/六有符号协方差及五条件变化完整保留。contact等原GT时间能量为零的比值保持null，不能把不适用项改成0或剔除分母。
- [全部新行及复用索引](../../../outputs/itm_rework/protocol/history_factorial_response_evaluation_20261009_v1/artifact_index.json)：新3600行、3600 paired−text路径差、1800 T-effect−F-effect交互全部valid，无缺失/无效。
- [第二公式证明](../../../outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1/result_verification.json)：从新T raw独立使用Gram/dot公式，并核对7200角色、全部3600对照/1800交互；**4640400项标量通过，max绝对差5.1514348342607263e−14**。原float64 `64eps×context scale`界不变。旧F的off−keep对照明确反号，RMS维持无符号。
- 7项新增合成CPU验收通过，覆盖T角色/strict absence、错误历史/条件/存在标志、重复/缺失单元及失败分母保留。合成测试不作为模型结果。主会话自检，不是独立02、人工或盲审。
- 本阶段GPU查询/构造模型/模型前向/生成噪声/采样/解码/训练均为0。原120完整输出、480轨迹、七条件及val000382拍手/000433踢腿返回/000139顺序、train004728静止等负例继续绑定，不新增数据准入或修复判据。

## 全程响应

[900步图](../../../outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1/history_factorial_response.png)、[有符号协方差图](../../../outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1/history_factorial_covariance.png)、[完整7200行面板](../../../outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1/all_time_panel.json)、[所有符号区间/旧时点](../../../outputs/itm_rework/protocol/history_factorial_response_panel_20261009_v1/summary.json)。比值为centered normalized root＋RIC67能量/固定GT；不是米、动作正确率或机制占比。

每个history内路径差固定为paired−text；交互固定为T-history路径差−F-history路径差。只有同history的t899起点相同，之后状态随条件演化，两个history之间起点也不同。

| 配置 | 比較对象 | 正负区间（t降序） |
| --- | --- | --- |
| head | F路径paired−text | 899→0全负 |
| head | T路径paired−text | 899→146负；145→0正 |
| head | T效应−F效应 | 899→0全正 |
| wrists | F路径paired−text | 899→877正；876→0负 |
| wrists | T路径paired−text | 899→0全负 |
| wrists | T效应−F效应 | 899→0全负 |
| head | F/T paired状态局部E(f11)−E(f10) | 各900步全负 |
| wrists | F paired状态局部变化 | 899→871正；870→102负；101→0正 |
| wrists | T paired状态局部变化 | 616负/284正；899→571持续负，94→0持续正，全部24个区间保留于summary |

区间来自零点符号的描述，不是新成功阈值、候选切换边界或独立重复样本。不据此挑新t、seed或checkpoint。

T历史paired轨迹的关键预测（同一行的f00/f10/f11）如下；actual组合因原float32运算次序与f11有极小差异，完整结果分别保存，未替换。

| 配置 | t | f00/GT | f10/GT | f11/GT | 局部IMU变化/GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 899 | 1.358864 | 1.449900 | 1.397552 | -0.052347 |
| head | 500 | 1.678056 | 1.687724 | 1.634580 | -0.053144 |
| head | 250 | 1.613698 | 1.619471 | 1.571816 | -0.047655 |
| head | 50 | 1.428820 | 1.432435 | 1.386133 | -0.046301 |
| head | 0 | 1.126842 | 1.129607 | 1.086217 | -0.043390 |
| wrists | 899 | 0.303802 | 0.302889 | 0.285634 | -0.017255 |
| wrists | 500 | 0.066650 | 0.066216 | 0.065835 | -0.000382 |
| wrists | 250 | 0.043229 | 0.042723 | 0.042436 | -0.000288 |
| wrists | 50 | 0.029964 | 0.029684 | 0.030125 | 0.000442 |
| wrists | 0 | 0.022865 | 0.022713 | 0.026569 | 0.003857 |

wrists后段局部变化转正时，f00/f10本身也已低动态。它与F历史结果一致表明“末端IMU每一步都压低动态”不能概括全程。head局部变化始终为负但没有wrists式近静态结局。以上不证明backbone唯一原因或权重无关；同checkpoint不同路径状态与条件响应共同演化。

## IMU主项与交互项：保存状态上的代数

为选择最小后续干预，额外从已保存四组件作明确标记的posthoc代数，未调用模型。设a=f10，u=f01−f00，j=f11−f10−f01+f00，则：

`E(a+u+j)−E(a) = [2Cov(a,u)+E(u)] + [2Cov(a,j)+E(j)] + 2Cov(u,j)`。

[完整3600 paired-state分解](../../../outputs/itm_rework/protocol/imu_response_component_algebra_20261009_v1/artifact_index.json)，3600恒等式通过（max8.807e−16）。[直接raw第二公式](../../../outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1/component_second_formula_verification.json)重新构造a+u/a+j并中心化计算，28800标量通过（max8.084e−15），沿用原64eps尺度界。假设能量不是新rollout或解码动作。

T-history/wrists：

| t | 单独添加u的变化/GT | 单独添加j的变化/GT | 2Cov(u,j)/GT | 合计变化/GT |
| ---: | ---: | ---: | ---: | ---: |
| 899 | -0.020026 | 0.002078 | 0.000693 | -0.017255 |
| 500 | -0.000594 | 0.000192 | 0.000020 | -0.000382 |
| 250 | -0.000397 | 0.000117 | -0.000008 | -0.000288 |
| 50 | 0.000377 | 0.000076 | -0.000011 | 0.000442 |
| 0 | 0.003732 | 0.000109 | 0.000016 | 0.003857 |

主项在所示早/中段状态为负，交互单独添加为正；末端主项已转正。因此不能简单认定去掉主项必然恢复动作，也不能称主项/交互提供独立百分比。这里只提出可检验的计算项干预，保留所有符号和协方差。

## 唯一下一任务

[新冻结两路径package](b0_imu_main_knockout_20261009_v1/package.json) SHA `a0a4b0bee0f4d61729357dc7a4592d9c24a1b98cf5d1314211bce0a34de905a1`，17842来源；[重建证明](../../../outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1/next_freeze_verification.json)9 payload逐字节一致，两配置精确T-prefix/paired首步锚点、1812 checks身份通过，3项角色/时间/权重冻结tests通过。**仅设计已冻结，尚未实现或GPU执行。**

固定原train000506 head/wrists的T-prefix x899，保持text/IMU presence true和原paired static-delta控制；仅将factorized `(text,IMU,joint)` 系数从(1,1,1)改为**(1,0,1)**，即去掉IMU主项、保留文本及交互。沿用原native组合运算顺序，不能用代数简化式替换浮点计算。后段t899→0保持原F200、caption、schedule和逐步noise；无新训练、seed、来源、边界或系数搜索。旧T text/paired及F四单元作已完成参考，不重跑。

实现与CPU验收后，单次新有界队列：2路径/1800MDM calls/7200 examples、E4/caption2、旧noise1798/decode2；总1200秒绝对期限/24GiB allocated cap/首故障停止/不重试不恢复。优先物理cuda:1 UUID `GPU-43c2f925-db05-c655-bb46-25196d0627d6`，沿用用户长期自主GPU授权与高利用率政策，3次资源观测free≥4096MiB，利用率只记录，不干预其他任务。

后续完整评价仍须原弓步恢复行为与五项数值规则，动态增加本身不能升级为修复成功。若这个计算项干预改变轨迹，也不能升级为唯一encoder/adapter因果归属或泛化能力。

[本阶段交付证明](../../../outputs/itm_rework/method/history_factorial_response_implementation_20261009_v1/delivery_verification.json)、[归档索引](../method/history_factorial_response_run_20261009_v1/artifact_index.json)。新GPU队列为0，当前无活动计算队列。
