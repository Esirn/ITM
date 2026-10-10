# B0 同历史后段IMU关闭干预：完整运行与评价（2026-10-08）

**已在99% GPU利用率、两个既有计算进程下完整完成一次冻结队列。后段关闭IMU改变了wrists轨迹并增加动态，但head/wrists均不支持完整弓步修复。** 两条keep-paired路径的旧F中间锚点和完整输出全部复现，最终features/joints/mask还逐字节相等。此实验将后段条件从共同F历史上隔离出来；结果不能把较活跃直接解释成正确动作，也没有推翻原F `not_supported`。

## 冻结身份、验收与真实运行

- [科学package v2](b0_suffix_switch_20261008_v2/package.json) `6f84cef7715b94f4ea3b8df374691b48104beee3424f24a931f13fcb616e0e8d`，3174来源。原train000506/head、wrists各2路径，从完整prefix v3保存的F paired x899加载同字节，保持F200和原1000 DDPM schedule，执行t899→0。
- [最终implementation v2](../../../outputs/itm_rework/method/suffix_switch_implementation_20261008_v2/implementation.json) `21b4252a8fdc3bfc0a516fe35b71eda3ce1cb97f96418bd308aedcf9c6843a01`，3192依赖；[CPU验收](../../../outputs/itm_rework/method/suffix_switch_implementation_20261008_v2/CPU_acceptance.json) `0ea3e636…`。6项新CPU验收覆盖完整3600步合成fixture、完整1000步baseline对照、native batch/严格缺失分支、原最终解码布局、首锚点故障/raw保存/完整失败分母、reservation/journal尾、绝对期限与无GPU/model prepare；沿用9项未变化的资源政策验收。
- 首轮CPU测试5通过/1失败（367.54秒），只因旧评价samples JSON是带rows字段的object。修复解析后定向2tests通过（7.52秒，含原失败项及真实prepare）。原v1 implementation/旧解析源码/失败JUnit保留，最终v2才是执行身份；worker/科学冻结未改，不重复已通过的耗时完整fixture。这不是GPU失败或放宽数值规则。
- [唯一执行记录](../method/B0_SUFFIX_SWITCH_EXECUTION_20261008.json)，沿用户长期GPU自主授权。原UUID/项目flock/native allocator、1200秒绝对期限、24GiB allocated cap、首故障停止，未重试、恢复、换卡、改优先级或干预他人。
- [runtime](../../../outputs/itm_rework/method/suffix_switch_authorized_20261008_v1/runtime/status.json) 全completed；3次资源样本GPU99/99/99%、controller30/30/30%、free7560MiB，2个既有进程，4次预检查询。高利用率无准入阈值；本次有实际计算进度，但没有测量其他任务性能或健康。
- 实际耗时 **563.816477秒**，peak allocated **527068672 bytes / 0.490871GiB**。4路径/3600步/12保存轨迹/4完整输出/7244checks全完成；MDM3600calls/14400examples、E4、caption2、旧noise3596、decode4，训练/新seed/新initial noise为0。
- [独立公式自检](../../../outputs/itm_rework/method/suffix_switch_implementation_20261008_v2/result_verification.json)核验3192依赖/7242worker产物、全部7244软件等式＋4load-to-source bitwise、3600后验链、3596旧noise descriptors；8422指标标量通过，最大绝对差3.553e−15（继承64eps尺度界）。这里“独立”指第二公式，不是独立02/人工审阅。

## 完整输出与原数值门槛

[最终4行](../../../outputs/itm_rework/protocol/suffix_switch_evaluation_20261008_v1/finals.json)、[12轨迹行](../../../outputs/itm_rework/protocol/suffix_switch_evaluation_20261008_v1/traces.json)均valid。centered root＋RIC能量/GT只表示归一化时间变化；MPJPE等沿原20Hz、不作空间/时间拟合，train contact/floor仍N/A。

| 配置/路径 | 能量/GT | Global MPJPE m | Root-relative m | Root path RMSE m | Head/wrists proxy m | Velocity proxy m/s | 骨盆高度范围 m | 骨盆路径 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head / keep_paired | 2.019157 | 0.792135 | 0.245008 | 0.809089 | 0.821648 | 1.041056 | 0.713995 | 3.251839 |
| head / switch_imu_off | 2.042759 | 1.022153 | 0.276204 | 0.992186 | 1.066456 | 1.124136 | 0.652125 | 3.257863 |
| wrists / keep_paired | 0.001069 | 0.639832 | 0.174087 | 0.651379 | 0.569547 | 0.638416 | 0.027213 | 0.151732 |
| wrists / switch_imu_off | 0.112553 | 0.855864 | 0.267650 | 0.813620 | 0.776381 | 0.655412 | 0.134828 | 0.311146 |

原keep-paired两配置的U/text/static五项门槛均pass，但行为仍失败。关闭IMU后：head相对原U五项全部nonworse失败；wrists四项失败，仅velocity改善。两个off对原F text/static仍过数值门槛，仍不足以支持行为。全部epsilon、严格root-relative改善与零退化规则保持原定义，未为本干预新设容差。[门槛细节与有符号差](../../../outputs/itm_rework/protocol/suffix_switch_behavior_20261008_v1/decision.json)。这是沿用原门槛的局部干预输出评价，不把IMU-off认证为paired控制模型。

wrists自身keep→off：能量/GT0.001069→0.112553（绝对+0.111483），骨盆高度范围0.027213→0.134828m；但骨盆均处于低姿态，右侧弓步/中段回收没有恢复。相对原F paired，off五项误差全部更大。不要用“约105倍能量”包装成功，基准本来接近0。原text-only完整输出能量/GT0.704812，off并未回到该值；text-only有不同prefix历史且自己也不是行为正确基线。

## 保存后段曲线

表中各行是各自路径同t的实际combined prediction；两路径只在x899起点状态相同，此后state不同，不是同状态条件预测对照。

| 配置/路径 | t500能量/GT | t250 | t50 | 完整输出 |
| --- | ---: | ---: | ---: | ---: |
| head / keep_paired | 1.803362 | 1.875029 | 1.944974 | 2.019157 |
| head / switch_imu_off | 1.913080 | 1.963562 | 2.004857 | 2.042759 |
| wrists / keep_paired | 0.058820 | 0.026918 | 0.004431 | 0.001069 |
| wrists / switch_imu_off | 0.098417 | 0.088262 | 0.084268 | 0.112553 |

## 完整行为与限制

[逐帧页面](../../../outputs/itm_rework/protocol/suffix_switch_review_20261008_v1/review.html)支持0:135全帧及oblique/ZY两视角；[审阅notes](../../../outputs/itm_rework/protocol/suffix_switch_behavior_20261008_v1/notes.json)固定保存观察。全部新off帧均检查；keep与旧F bitwise一致，沿用此前完整粗审阅并核对新概览/曲线。单AI、标签已暴露，不是盲评、独立02或人工验证。

- **head off incomplete**：早段有腿部移动，随后持续下沉/近坐姿；GT窗口67:106所见中段回收缺失，后半段仍低蹲。
- **wrists off incomplete**：起始低屈膝，少量早段变化后持续低位；没有完整初始恢复、右前跨/换侧顺序和中段回收。
- keep head/wrists分别复现原F incomplete；四输出按原行为＋数值门槛均not_supported。GT初始已屈膝、末端仍下沉，不要求片段外初始迈步或最终站直，没有因为缺少未观测动作而判失败。

![wrists完整对照](../../../outputs/itm_rework/protocol/suffix_switch_review_20261008_v1/wrists_overview.png)
![head完整对照](../../../outputs/itm_rework/protocol/suffix_switch_review_20261008_v1/head_overview.png)

[120旧完整参照](../../../outputs/itm_rework/protocol/suffix_switch_evaluation_20261008_v1/old_final_references.json)仍保留原六来源/十二配置/七条件和负例。本队列仅一个source、两个配置、一个seed，不对其余10配置制造新干预结果或声称广泛改善；train004728保持、val000382/000433/000139等原问题不变。

## 研究判断与唯一下一步

这一固定起点的条件干预支持：**在相同F prefix/权重/后验噪声下，后段IMU存在与否会改变继续采样的完整输出，wrists原近静止可被部分缓解。** 不支持关闭IMU是弓步修复，也不能把早期历史宣称不可逆、排除其他边界、把所有变化唯一归因于encoder/adapter，或据此追加训练/损失/guidance sweep。

唯一下一任务已冻结为[完整900步保存响应CPU计划](b0_suffix_response_CPU_20261008_v1/plan.json)：复用全部4×900＝3600组原始分支/state，按原11视图和balanced temporal字段做全程时间能量/四项响应/有符号协方差及1800组路径描述对照，再作第二公式重放。只有一个source；步数/帧数不当作独立样本。不生成新noise、不解码、不开新GPU队列，完成后再选择是否需要更小的机制干预。计划在结果暴露后冻结，明确是描述性后续，不伪称盲预注册。

代码、CPU验收/修复、冻结设计、执行记录、收据/日志、评价/行为与面板索引归档至[本阶段archive](../method/suffix_switch_run_20261008_v1/artifact_index.json)。当前无活动队列，GPU自主授权与高利用率准入有效；原F行为结论和论文主张均未变。
