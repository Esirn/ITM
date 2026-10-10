# B0 前缀历史×后段条件完整对照（评价日期2026-10-09）

**已按用户最新偏好在物理cuda:1完整执行T-prefix→text/paired四路径队列。text-only基线与旧F逐字节相等；两个新paired输出仍不支持完整弓步修复。** 合并前次F-prefix→paired/text后，head/wrists各四个单元完整。wrists在两种起始历史下，paired后段都比text后段大幅降低时间变化；换成text历史并未消除收缩，历史仍改变幅度。head的条件效应则随历史改变方向，不能把现象解释为与状态无关的全局缩放器。

## 设备、身份与软件完成事实

[设备偏好政策](../GPU_DEVICE_PREFERENCE_20261008.json) SHA `bd156929…`记录用户“未来一段时间可以尽量用cuda:1”。物理index1为RTX4090，UUID `GPU-43c2f925-db05-c655-bb46-25196d0627d6`；单卡CUDA_VISIBLE_DEVICES隔离后worker本地cuda:0对应它。该指令覆盖旧资源文字中的固定物理cuda:0，只变设备选择，不改科学package/precision/tolerance/预算。首次只读mapping查询1次；实际队列预检4次，3样本free12051MiB、GPU98/98/97%、controller72/72/52%，已有1个compute process。没有利用率准入上限，未测他人任务性能/健康、未干预其他进程。

- [冻结factorial package v2](b0_suffix_history_factorial_20261008_v2/package.json) SHA `abd7894aaf4fdc90f7fa35d595b51152fb8a9533c64accbed0c0e5928034d40c`，10504来源；被CPU来源检查拒绝的v1锚点草稿保持原位，未执行。
- [最终implementation](../../../outputs/itm_rework/method/suffix_history_factorial_implementation_20261008_v1/implementation.json) SHA `83de9380f428bf1075524a1f30da6abcab7581eda4306d5b7027e1542b2f75e0`，10522绑定依赖；[CPU验收](../../../outputs/itm_rework/method/suffix_history_factorial_implementation_20261008_v1/CPU_acceptance.json)7个distinct tests覆盖通过：完整3600合成worker与独立1000步text基线、strict absence/on改变输出、call/noise/解码/7244checks/7242artifact计数、raw-before-anchor-fault与完整失败分母、真实journal reservation/fsync/tail、batch/deadline、无GPU/model prepare、cuda:1 UUID绑定。
- 完整合成fixture为吞吐暂抑制fsync（75.64秒），独立故障/journal tests实际fsync；没有对真实worker禁用fsync。设备偏好追加后真实prepare与device绑定2定向tests再通过（8.37秒）；继承9项未变化资源政策测试，不重做无变化验收。
- [唯一执行记录](../method/B0_SUFFIX_HISTORY_FACTORIAL_EXECUTION_20261008.json)，[runtime](../../../outputs/itm_rework/method/suffix_history_factorial_authorized_20261008_v1/runtime/status.json)全completed。4路径/3600steps/12traces/4finals/7244checks全部完成，MDM3600calls/14400examples，encoder4、caption2、原noise3596、decode4；新训练/seed/initial noise0。
- 实际耗时 **648.010967秒**，peak allocated **527068672bytes / 0.490871GiB**；1200秒绝对期限/24GiB cap、按UUID项目锁、首故障停止，未重试/恢复/换卡。
- [全量软件/轨迹第二公式核验](../../../outputs/itm_rework/method/suffix_history_factorial_implementation_20261008_v1/result_verification.json)通过10522依赖/7242worker产物、7244等式＋4源起点检查、3600状态链、3596旧noise descriptors和8422指标标量（max3.553e−15）。[跨单元第二公式](../../../outputs/itm_rework/method/suffix_history_factorial_implementation_20261008_v1/factorial_verification.json)独立重算8单元的五运动指标/jerk/extent/时间能量及signed effect/interaction，1568标量通过（max2.220e−15）。均使用原64eps尺度界/旧科学软件界，主会话自检，不是独立02或人工验收。
- [baseline bitwise证明](../../../outputs/itm_rework/method/suffix_history_factorial_implementation_20261008_v1/bitwise_baseline_verification.json)：HX001/head、HX003/wrists的final normalized features/joints/mask与旧F text-only全bitwise相等，跨设备换用cuda:1没有改变这两个基线产物。仅此两基线的观察，不泛化所有算子/硬件。

## 四个history/suffix单元

F-prefix为原F200 paired前100步，T-prefix为原text-only前100步，均来自已完成prefix v3的x899。后段始于固定t899→0。保持caption、最终F200、unit factorized guidance、MDMbatch4/caption1/full1000 DDPM、旧逐步noise。F-prefix两单元来自前次完整队列，T-prefix两单元是本次新队列；旧单元不当独立/ contemporaneous replication，只有1source/2config/1seed，不宣称泛化。

[全部8单元](../../../outputs/itm_rework/protocol/suffix_history_factorial_evaluation_20261008_v1/factorial_finals.json)、[factorial signed对照](../../../outputs/itm_rework/protocol/suffix_history_factorial_evaluation_20261008_v1/factorial_contrasts.json)。时间能量为centered normalized root＋RIC67/固定GT，不表示运动正确、米或因果百分比。

| 配置 | 前缀 | 后段 | 能量/GT | Global MPJPE m | Root-relative m | Root path RMSE m | Head/wrists proxy m | Velocity proxy m/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| head | F_paired | paired | 2.019157 | 0.792135 | 0.245008 | 0.809089 | 0.821648 | 1.041056 |
| head | F_paired | text_only | 2.042759 | 1.022153 | 0.276204 | 0.992186 | 1.066456 | 1.124136 |
| wrists | F_paired | paired | 0.001069 | 0.639832 | 0.174087 | 0.651379 | 0.569547 | 0.638416 |
| wrists | F_paired | text_only | 0.112553 | 0.855864 | 0.267650 | 0.813620 | 0.776381 | 0.655412 |
| head | T_text_only | text_only | 0.995083 | 1.238703 | 0.277152 | 1.227222 | 1.286763 | 1.184371 |
| head | T_text_only | paired | 1.086217 | 1.009086 | 0.250609 | 1.015312 | 1.045631 | 1.084080 |
| wrists | T_text_only | text_only | 0.704812 | 1.637532 | 0.328397 | 1.555607 | 1.639487 | 0.872362 |
| wrists | T_text_only | paired | 0.026569 | 0.740410 | 0.249886 | 0.718885 | 0.686602 | 0.626586 |

每种历史的suffix effect固定为paired−text；interaction固定为T-history effect−F-history effect。取能量/GT：

| 配置 | F历史suffix effect | T历史suffix effect | signed interaction |
| --- | ---: | ---: | ---: |
| head | -0.023602 | 0.091134 | 0.114736 |
| wrists | -0.111483 | -0.678243 | -0.566759 |

wrists从T历史开启paired使能量/GT0.704812→0.026569，比原F历史always-paired0.001069高，但仍仅GT约2.66%，没有恢复完整动态。原F前100步不是本固定case中收缩的必要前提；同时history effect不为0，不能称早期历史无关、不可逆或唯一encoder/adapter原因。head的effect符号翻转更明确说明条件影响依赖演化历史。

## 行为及原门槛

[新4输出的notes](../../../outputs/itm_rework/protocol/suffix_history_factorial_behavior_20261009_v1/notes.json)与[原规则decision](../../../outputs/itm_rework/protocol/suffix_history_factorial_behavior_20261009_v1/decision.json)已保存。两个on输出全部136帧检查，两个text基线沿用bitwise原完整审阅并核对新图/曲线；标签已暴露、single-AI粗审，不是盲/独立/人类评价。

- **HX002/head on incomplete**：早段腿部移动，随后下沉并保留右侧领先弓步样姿态；中段骨盆有小幅0.647m→0.669m回升，却没有原GT所见完整站直回收/交替序列。相对原U五门槛4项失败，仅root-relative通过；相对自身/原text/static五项数值均通过。
- **HX004/wrists on incomplete**：全片低位屈膝/跪姿样，骨盆范围仅0.077744m、路径0.496544m；缺少初始恢复、观察到的换侧/中段回收。相对原U4项失败，仅velocity改善；相对text/static数值通过。
- 两text基线仍是原不完整动作，不能把接近GT的normalized能量值当作行为成功。当前四历史/条件单元的修复支持均not_supported，原F模型判定和论文主张保持不变。GT片段外的初始迈步/最后站直不要求；train floor/contact仍N/A，无时间/空间拟合或新阈值。

![wrists四单元](../../../outputs/itm_rework/protocol/suffix_history_factorial_review_20261008_v1/wrists_overview.png)
![head四单元](../../../outputs/itm_rework/protocol/suffix_history_factorial_review_20261008_v1/head_overview.png)

[全部帧交互页](../../../outputs/itm_rework/protocol/suffix_history_factorial_review_20261008_v1/review.html)有oblique/ZY；[28文件面板](../../../outputs/itm_rework/protocol/suffix_history_factorial_review_20261008_v1/artifact_index.json)保留全8单元帧表与旧U/text参照。原120最终输出、480历史trace及七条件/val kick/clap/walk/train hold等负例全部继续绑定；没有制造其余10配置的本次新干预或新行为标签。

## 下一研究决策

当前完整对照支持“suffix条件与历史共同影响采样轨迹”，不支持从更好text起点开启F200就能修复主动作；不追加训练/损失权重/guidance/boundary搜索，不把失败升级为模型认证或纸面主结果。

唯一下一任务已冻结为[全程history/suffix响应CPU计划](b0_history_factorial_response_CPU_20261009_v1/plan.json)：计算新T-prefix3600保存分支/state行，复用已验证F-prefix3600 CPU行，总有效7200行；3600 within-history路径对照/1800每t/config factorial interaction，原11视图/balanced temporal及四项响应有符号协方差。只复用已存数组，不查询GPU、不载模型、不生成noise、不解码或重新采样。先完成全程响应及第二公式核验，再选择是否确需下一项最小机制干预。计划在结果暴露后冻结，是描述性后续，不是盲预注册或独立t/frame样本。

[本阶段归档](../method/suffix_history_factorial_run_20261009_v1/artifact_index.json)保留代码/CPU验收/科学package/设备偏好/执行记录/收据/日志/评价/行为/面板与完整run索引。当前无活动队列；未来仍优先物理cuda:1，长期GPU自主授权与高利用率共享政策有效，无需反复手动批准。运行/源设计目录沿用20261008不可变身份，本报告和后续CPU计划按评价日20261009记录。
