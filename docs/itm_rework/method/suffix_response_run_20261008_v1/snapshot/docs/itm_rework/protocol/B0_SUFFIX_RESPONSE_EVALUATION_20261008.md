# B0 完整900步保存响应CPU评价（2026-10-08）

**全部3600保存状态/分支点与1800路径对照valid，4246200个第二公式标量核验通过。** 主wrists的局部IMU能量变化在t870→102为负，t101→0已转为正；但这段状态上的backbone无条件和text预测本身都趋于静态。因此“最后一步残差即时抵消运动”仍不是充分解释。此前同F起点关闭IMU的真实suffix队列只增加部分动态，head/wrists完整行为仍incomplete/not_supported，没有更新论文主张。

## 完成身份与算术

[冻结CPU计划](b0_suffix_response_CPU_20261008_v1/plan.json) SHA `e6bfa66b25e656f574144149fc8f69d3e452927aa7a6999a2c33674ae58434ed`，10468来源；1个source×head/wrists×keep/off×t899:0，共3600行，1800同t/config路径对照。不是6个source的干预，不把900步或136帧当独立样本。

[CPU implementation](../../../outputs/itm_rework/method/suffix_response_implementation_20261008_v1/implementation.json) SHA `179534723c15e07b4567b963ae4aefbd8fcee934e97842c8044ed409bc5a84bc`，10471依赖、3600原始raw/metadata身份、F200/float32/fullmask/原严格缺失分支软件界核验通过。8项身份拒绝测试通过，继承原temporal/response数学验收；没有重建模型或生成噪声。

[全部rows](../../../outputs/itm_rework/protocol/suffix_response_evaluation_20261008_v1/rows.json)覆盖x_t/f00/f10/f01/f11/actual_combined_prediction的11视图与原balanced temporal字段；响应按base、text、IMU、interaction的能量及6个有符号twice-covariance分解，并保留原5种局部条件差。[1800对照](../../../outputs/itm_rework/protocol/suffix_response_evaluation_20261008_v1/path_contrasts.json)为off-minus-keep、全视图MSE/均值误差/时间误差/能量/协方差和原state RMS描述，source_n始终1。

[第二公式proof](../../../outputs/itm_rework/method/suffix_response_implementation_20261008_v1/result_verification.json)通过4246200标量，最大差4.086e−14，在继承64eps×上下文尺度界内。Gram/dot公式独立于指标入口，仍是主会话自检，不是独立02/人工复核。源GT为固定时间参照，不是条件期望。

本CPU阶段新GPU查询、模型前向、采样、噪声生成、训练和解码全部0。真实3600前向和4个解码属于[上阶段完整suffix实验](B0_SUFFIX_SWITCH_EVALUATION_20261008.md)，未重新执行或拼接任何失败队列。原120完整/480轨迹参照及全部七条件负例仍绑定。

## 全程曲线及主wrists

![全程保存响应](../../../outputs/itm_rework/protocol/suffix_response_panel_20261008_v1/full_suffix_response.png)

时间能量按相同固定GT能量缩放；下表conditional列为 `(E(f11)−E(f10))/E(GT)`，不是可加因果百分比。不同路径除了x899起点外，其余state不同；同状态的局部响应只属于本行state。

| 配置 | t | keep f00/GT | keep f10/GT | keep f11/GT | 局部IMU能量差/GT | off实际预测/GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 899 | 1.402219 | 1.512977 | 1.443956 | -0.069020 | 1.512977 |
| head | 500 | 1.840386 | 1.851951 | 1.803362 | -0.048589 | 1.913080 |
| head | 250 | 1.903611 | 1.911745 | 1.875029 | -0.036716 | 1.963562 |
| head | 50 | 1.971727 | 1.980053 | 1.944974 | -0.035079 | 2.004857 |
| head | 0 | 2.058255 | 2.067820 | 2.019157 | -0.048663 | 2.042759 |
| wrists | 899 | 0.354940 | 0.358615 | 0.361189 | 0.002574 | 0.358615 |
| wrists | 500 | 0.068693 | 0.068249 | 0.058820 | -0.009429 | 0.098417 |
| wrists | 250 | 0.029743 | 0.030084 | 0.026918 | -0.003166 | 0.088262 |
| wrists | 50 | 0.003673 | 0.003827 | 0.004431 | 0.000604 | 0.084268 |
| wrists | 0 | 0.000358 | 0.000380 | 0.001069 | 0.000689 | 0.112553 |

- wrists：局部能量变化t899→871为正（29步），t870→102为负（769步），t101→0又为正（102步）。这些符号区间是输出后数学描述，不是寻找成功时间边界或统计检验。t0的f10仅0.000380/GT、f00仅0.000358/GT；加入IMU反而到0.001069/GT。把末端近静止全归结为即时残差抵消不成立。
- head：900步局部IMU变化均为负，但最终f11能量/GT仍2.019157，没有出现wrists同样的近静态。局部负协方差/能量减少本身不认证塌缩。
- 两路径x899 state RMS差为0；t0分别head0.450362、wrists1.431820。它们在共同起点/噪声下逐步形成不同状态，不能拿各自f10作为同state对照。旧text-only最终wrists0.704812有不同prefix，仍是历史参照。

完整图表与符号区间见[panel索引](../../../outputs/itm_rework/protocol/suffix_response_panel_20261008_v1/artifact_index.json)和[sign intervals](../../../outputs/itm_rework/method/suffix_response_implementation_20261008_v1/sign_intervals.json)。IMU局部项与backbone/state反馈相关，不把曲线差按比例切成互斥机制，也不排除其他边界/权重/源上的不同表现。

## 已选定并冻结的最小下一干预

当前已有F-prefix→paired、F-prefix→text两个完整新路径，及原text-prefix→text历史参照；尚缺text-prefix→paired。下一步补齐条件历史×suffix面板，每个配置从完整prefix v3中保存的 **text-only x899** 开始：keep text-only（必须复现旧F text-only）；switch paired on（原真实paired静态差分control）。两条新路径共享T起点/旧噪声；已有F起点keep/off结果固定保留。不是更换t边界、训练、seed、source、caption、checkpoint或guidance搜索。

[新factorial package](b0_suffix_history_factorial_20261008_v1/package.json) SHA `d79ac5d04db24867a3ee53b42b40e1272c46bd3540ca33a9883c3c9f70414baf`，10492来源，9payload逐字节重建、2个T来源起点配对和8个原text-only锚点组核验通过；2项增量CPU冻结检查通过。新4路径/3600 MDM/14400examples、E4、caption2、旧noise3596、decode4、7244checks；1200秒/24GiB cap、UUID/flock/空余4GiB/利用率只记录、首故障停止，与原共享政策一致。历史输出不当独立重复样本。

该设计在CPU曲线暴露后冻结，为前瞻性补齐缺失单元，不是盲预注册。**唯一下一任务：实现并CPU验收这个有界T-prefix keep/on入口，然后按长期GPU自主授权运行一次新队列与完整评价。** 新入口尚未实现/执行，不修改已接受的F-prefix worker/manifest。要求keep-text baseline的三组旧traces和最终features/joints/mask通过原界，完整行为/原5门槛与所有负例继续保留。若需要进一步干预，在本设计完整评价后再决定。

本阶段代码/测试/实现/数学proof/全部结果与panel索引/新设计归档到[archive](../method/suffix_response_run_20261008_v1/artifact_index.json)。当前没有活动GPU或CPU队列；GPU自主授权有效，原F行为not_supported及论文主张不变。
