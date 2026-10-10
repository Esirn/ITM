# B0 IMU主项去除实验：完整评价（2026-10-09）

**已在物理cuda:1完整完成原T-prefix x899→(1,0,1)两路径。去掉IMU主项、保留交互后，wrists的预测时间能量接近text-only，但完整弓步仍不支持修复，且相对旧U五项数值门槛全部失败。head同样不支持修复。** 在两个新路径的全部保存状态上，去掉主项都增加root＋RIC归一化MSE与时间能量；“减少收缩”与“提高完整动作正确性”需要分开判断。原F及论文主张未变。

## 已完成事实与身份

[科学package](b0_imu_main_knockout_20261009_v1/package.json) SHA `a0a4b0bee0f4d61729357dc7a4592d9c24a1b98cf5d1314211bce0a34de905a1`，17842来源未变。[入口/CPU验收报告](../method/B0_IMU_MAIN_KNOCKOUT_RUN_20261009.md)。[最终implementation](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/implementation.json) SHA `7f6eb510b1c3fad38fa64af2a94b00dbf5db5566d8772b1c42acaa9a2903855d`，17860来源；7新增CPU tests通过（52.21秒），完整合成回放、真实fsync故障/journal、1800/1812/3626分母、父时限/无GPU prepare/UUID绑定均通过。未覆盖旧绑定代码或失败。

[执行记录](../method/B0_IMU_MAIN_KNOCKOUT_EXECUTION_20261009.json)沿用长期自主GPU授权，优先物理cuda:1 UUID `GPU-43c2f925-db05-c655-bb46-25196d0627d6`，单卡隔离后worker本地cuda:0对应它。预检GPU利用率98/97/97%、free12051MiB、已有1个计算进程；4次预检查询，没有额外mapping查询。未设置利用率上限、修改其他进程或全局GPU参数。

[完整runtime](../../../outputs/itm_rework/method/imu_main_knockout_authorized_20261009_v1/runtime/status.json)：2路径/1800steps/1812checks/6traces/2finals全部completed。MDM1800calls/7200examples，encoder4/caption2，旧noise1798/decode2，新训练/seed/initial noise0。实际 **311.911015秒**，peak allocated **526790144bytes / 0.490611GiB**；1200秒绝对期限/24GiB cap（非显存预留）、UUID项目锁/native allocator/首故障停止，不重试、不恢复。

- [全部软件/链/指标第二公式](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/result_verification.json)：17860来源、3626worker产物、1812软件等式、1800状态hash links、1798noise descriptors、5338指标标量通过（max3.553e−15）。1812含2原起点/10首步paired x_t/f00/f10/f01/f11/1800原float32组合顺序，原1e−6绝对＋相对软件界不变。状态链是保存描述符核对，不伪称新的完整后验数值重算。
- [10单元/8差值第二公式](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/contrast_verification.json)：1624标量通过（max2.220e−15），独立从关节重算五指标/jerk/extent以及signed comparisons。此前F/T八单元是完成参考，不是新并行重复。
- [新2final/6trace/8对照](../../../outputs/itm_rework/protocol/imu_main_knockout_evaluation_20261009_v1/artifact_index.json)全部valid，原120final/480trace及七条件、全部负例保留，缺失分母没有删除。

## 最终动作与原门槛

新MK001=head、MK002=wrists；原F200/静态差分控制、caption、136帧、1000DDPM原posterior/旧noise保留，唯一变化是原factorized顺序中的IMU主项系数1→0。真实text/IMU presence均true，交互系数仍1；不是把IMU关闭或退化为strict text-only。

[10单元完整表](../../../outputs/itm_rework/protocol/imu_main_knockout_evaluation_20261009_v1/factorial_finals.json)。以下T-history路径均从同一原x899起点开始；后续状态随guidance演化不同。能量为centered normalized root＋RIC67/固定GT，不是米、动作正确率或机制占比。

| 配置 | 后段 | 能量/GT | Global MPJPE m | Root-relative m | Root path RMSE m | Head/wrists proxy m | Velocity proxy m/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 原text-only | 0.995083 | 1.238703 | 0.277152 | 1.227222 | 1.286763 | 1.184371 |
| head | 原paired(1,1,1) | 1.086217 | 1.009086 | 0.250609 | 1.015312 | 1.045631 | 1.084080 |
| head | 新去主项(1,0,1) | 0.989244 | 1.232811 | 0.277552 | 1.221576 | 1.282182 | 1.180783 |
| wrists | 原text-only | 0.704812 | 1.637532 | 0.328397 | 1.555607 | 1.639487 | 0.872362 |
| wrists | 原paired(1,1,1) | 0.026569 | 0.740410 | 0.249886 | 0.718885 | 0.686602 | 0.626586 |
| wrists | 新去主项(1,0,1) | 0.701386 | 1.629136 | 0.328221 | 1.545734 | 1.632296 | 0.871364 |

wrists相对原T paired的能量/GT增加0.674817，与text-only仍差−0.003426；head相对原T paired减少0.096972，与text-only差−0.005839。在这个固定source/seed与起点下，删除此计算项使轨迹动态更接近text路径；不能称主项是唯一encoder/adapter原因、所有历史都如此或交互项完全无作用。新轨迹与原text不是bitwise等同，本报告只描述量值接近。

[全部136帧/曲线/交互滑块](../../../outputs/itm_rework/protocol/imu_main_knockout_review_20261009_v1/review.html)、[粗审notes及原判据decision](../../../outputs/itm_rework/protocol/imu_main_knockout_behavior_20261009_v1/artifact_index.json)：

- **head：incomplete/not_supported。** 初段走步样换腿，未形成GT已开始的左侧前跨屈膝恢复；后半持续前倾屈膝，中段没有GT已观察到的回收至更直立状态。骨盆高度80帧0.796732→105帧0.802881，最后0.789784m；范围0.198849m。相对原U五项门槛全部失败；相对自身text/static，root-relative变差0.000401m，原界下也失败。
- **wrists：incomplete/not_supported。** 全片低位深屈膝，早段一侧腿伸出、后续两侧腿位变化，但初始及中段回收缺失，末段仍低位。骨盆高度0.386177m起、105帧0.271787、最后0.208914；范围0.326751m。相对原U五项门槛全部失败；相对自身text/static五项数值通过，不能覆盖行为失败。

骨盆高度范围由原骨盆曲线计算，区别于motion_extent的所有关节Y范围。全部136帧已查看，标签暴露、单AI粗审；不是独立02/人工/盲审。显示按每帧root XZ居中只用于观察，指标仍原坐标，不拟合。GT起点已屈膝，末尾仍下沉，不要求片外起步/末端站直；train floor/contact保持N/A，没有加入新成功阈值。

## 全程保存响应：同时增加动态和误差

[全部1800保存状态CPU结果](../../../outputs/itm_rework/protocol/imu_main_knockout_responses_20261009_v1/artifact_index.json)保留6原预测的11视图/balanced temporal与四组件/有符号协方差分解。额外在每个新knockout状态上用保存的f00/f10/f01/f11按原float32顺序重构native(1,1,1)，计算同状态新(1,0,1)−native111的误差分解。这是无额外模型调用的同状态代数，不是再跑一条paired轨迹。

[第二公式](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/response_verification.json)：**2413800标量通过，max6.040e−14**，原float64 64eps尺度界未改；contact/零GT能量的比值保持null。CPU分析/核验/渲染均GPU查询/模型/噪声/解码0。

[全900步图](../../../outputs/itm_rework/protocol/imu_main_knockout_response_panel_20261009_v1/knockout_responses.png)、[全部值/旧时点](../../../outputs/itm_rework/protocol/imu_main_knockout_response_panel_20261009_v1/summary.json)。head和wrists各900步同状态删除主项均增加root＋RIC时间能量及MSE。wrists示例：

| t | 新实际能量/GT | 同状态native111能量/GT | 同状态删除主项的MSE变化 |
| ---: | ---: | ---: | ---: |
| 899 | 0.304967 | 0.285634 | 0.216332 |
| 500 | 0.279327 | 0.257588 | 0.131451 |
| 250 | 0.326506 | 0.299703 | 0.159298 |
| 50 | 0.469047 | 0.433380 | 0.242894 |
| 0 | 0.701386 | 0.655812 | 0.349312 |

t0同状态native111仍0.655812，而真实paired路径t0为0.026569；两者checkpoint相同、状态不同。这与先前同paired状态末段主项增能量的现象一起说明响应符号和长期变化依赖演化状态，不能把末端局部差值解释成全程机制百分比。删除主项提升动态同时牺牲GT误差/关节代理指标，当前不作为方法修复或部署候选。

## 唯一下一任务

[已冻结跨来源组件/误差CPU计划](b0_cross_source_component_CPU_20261009_v1/plan.json) SHA `49cf5ad9f5738aa653ef07ababa6100e5027e7e3892aad6f4e450a34efb7f9de`，17906绑定来源；480原F/U trace/240配对，其中96 paired rows/48 paired F-U对照，6既有来源/12配置、原t900/500/250/50，全部七条件与120final保留。3项合成人口/配对/负例冻结检查通过。仅CPU设计冻结，尚未执行。

实现后复用原raw四分支，在每个原状态上计算IMU主项/交互项的有符号时间能量和GT误差分解，并检查原静止、strict absence、bypass及全部负例。所有11视图与原balanced字段、GT能量为零的null及完整分母保留；F/U状态不同，不称同状态checkpoint因果。先从全部来源判断抑制动态与误差收益在哪些条件共同出现，再选择最小方法干预。本计划GPU/模型/噪声/解码0，没有选定下一GPU队列或guidance搜索。

[交付证明](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/delivery_verification.json)、[归档](../method/imu_main_knockout_run_20261009_v1/artifact_index.json)。当前无活动队列；后续GPU仍自主判断并优先物理cuda:1。
