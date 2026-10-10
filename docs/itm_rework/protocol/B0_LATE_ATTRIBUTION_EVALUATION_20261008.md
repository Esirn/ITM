# B0 晚段保存预测分解与最小后段干预决策（2026-10-08）

**冻结的96行晚段CPU分解已全部完成，两种顺序均显示主wrists的轨迹状态差项在后段扩大，同状态checkpoint差项保持较小。** 这是已有预测的代数事实，尚未隔离状态形成的因果原因。旧text-only自身轨迹后段动态回升，而F paired继续收缩；这支持下一步从同一个F历史状态开始的后段IMU关闭对照，不认证关闭IMU能修复动作。原F主head/wrists行为判定仍为 `not_supported`，未改论文主张。

## 身份与完成范围

- [冻结CPU计划](b0_late_attribution_CPU_20261008_v1/plan.json) SHA `26ccf5866ba3f21bcfa4b80cc4e75ec76ed8c42a40621c3784e0972cc485e2e9`，405来源；6来源×head/wrists×4时间步×2顺序＝96行。只复用原生诊断完整队列的F200/U200在F/U paired状态上的P分支，不增补未保存的text-only状态预测。
- [CPU实现](../../../outputs/itm_rework/method/late_attribution_implementation_20261008_v1/implementation.json) SHA `35b48b45c9000be223502f656300e25140741bcf86b480be802b79988f5ee10e`，513绑定依赖、192个已保存作业的身份/原生batch/float32 shape/target/frame mask均通过。原生package `95283f4c…`、MDM5184examples/1152calls等完整收据已核验。
- [全部96行](../../../outputs/itm_rework/protocol/late_attribution_evaluation_20261008_v1/rows.json)全valid。[分组表](../../../outputs/itm_rework/protocol/late_attribution_evaluation_20261008_v1/aggregates.json)保留split/config/t/ordering、等来源mean/min/max/planned/valid n，无帧数加权或跨时间池化。
- 复用并完整保留480个旧轨迹点与120个完整输出，七条件和所有旧负例均在参照表内；30个旧text/static/bypass跨F/U最终输出bitwise等式的既有证据未改。
- 新增8项身份/顺序适配测试和5项suffix冻结测试通过。最终[第二公式核验](../../../outputs/itm_rework/method/late_attribution_implementation_20261008_v1/result_verification_v2.json)通过195772个标量、192个原始作业、480轨迹/120最终输出及分组；回查1103个原始参照来源。最大绝对差1.819e−12，均在继承的float64 64eps×数值尺度界内，没有放宽科学阈值。这是主会话自检，不是独立02或人工验证。
- 本阶段新GPU查询、模型前向、噪声生成、训练、采样、解码均为0。

## 两种顺序及主案例

记 `p=F@F_state`、`r=U@U_state`。顺序A取 `i=U@F_state`，checkpoint项 `d=p−i`，state项 `s=i−r`；顺序B取 `i=F@U_state`，checkpoint项 `d=i−r`，state项 `s=p−i`。两者都满足 `p−r=d+s`。原始差能量包含 `2〈d,s〉`，MSE和中心化时间能量差还有相对reference的有符号交叉项，不能把各项当作可加百分比。

下表为root＋RIC67归一化特征的RMS；不表示米、不表示不同时间间的状态距离，也不作显著性检验。

| 配置 | t | A checkpoint | A state | B checkpoint | B state |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 900 | 0.030314 | 0.214052 | 0.030842 | 0.211718 |
| head | 500 | 0.023334 | 0.210558 | 0.024657 | 0.210907 |
| head | 250 | 0.023071 | 0.237582 | 0.024668 | 0.237078 |
| head | 50 | 0.022848 | 0.298483 | 0.024649 | 0.296228 |
| wrists | 900 | 0.033956 | 0.573888 | 0.033487 | 0.569601 |
| wrists | 500 | 0.033346 | 1.012987 | 0.025296 | 1.011099 |
| wrists | 250 | 0.026099 | 1.138417 | 0.026844 | 1.133874 |
| wrists | 50 | 0.019332 | 1.302999 | 0.027793 | 1.289857 |

全部12配置在全部4时间步、两种顺序下state RMS都大于checkpoint RMS（96/96）。幅度并不一致；例如t50顺序A的val000433/head为0.031056与0.003860，train000506/wrists为1.302999与0.019332。不能从大小关系推出权重无关：这些状态本来就是此前权重/条件经过采样累积形成的。head的state项也增长，但最终仍有较高动态能量，因此“state项较大”自身不能诊断塌缩。

## 原text-only与完整输出参照

这里只对照各自原轨迹上的实际combined prediction，不能把它们冒充同状态的条件反事实。表内是root＋RIC67中心化能量/固定GT能量；最后一列是原完整输出，不是t0的保存预测，也不认证动作正确。

| 配置/原路径 | t900 | t500 | t250 | t50 | 完整输出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| head / F paired | 1.436202 | 1.803362 | 1.875029 | 1.944974 | 2.019157 |
| head / U paired | 1.220609 | 1.631162 | 1.701522 | 1.787710 | 1.823020 |
| head / F text_only | 1.446644 | 1.686393 | 1.593444 | 1.349488 | 0.995083 |
| wrists / F paired | 0.363517 | 0.058820 | 0.026918 | 0.004431 | 0.001069 |
| wrists / U paired | 0.978249 | 1.233554 | 1.278558 | 1.243606 | 1.111474 |
| wrists / F text_only | 0.308388 | 0.283234 | 0.330238 | 0.472529 | 0.704812 |

![完整主案例晚段面板](../../../outputs/itm_rework/protocol/late_attribution_panel_20261008_v1/primary_late_panel.png)

wrists原text-only t900较低，随后升至最终0.704812；F paired最终仅0.001069。此前保存响应分解还显示F paired t50局部IMU响应略增时间能量。因此“最后一步IMU即时抵消运动”不足以解释这个收缩，也不能只凭原text-only最终较活跃宣称后段关闭IMU会成功。

所有参照与负例见[480轨迹](../../../outputs/itm_rework/protocol/late_attribution_evaluation_20261008_v1/reference_traces.json)、[120最终输出](../../../outputs/itm_rework/protocol/late_attribution_evaluation_20261008_v1/reference_finals.json)和[参照分组](../../../outputs/itm_rework/protocol/late_attribution_evaluation_20261008_v1/reference_aggregates.json)。train004728保持动作、val000382下肢/鼓掌、val000433踢腿回收及imu-only/numeric-zero等原负例保留，未用这个局部分析更新其行为结论。原F完整判据见[F评价](B0_FEATURE_BALANCE_EVALUATION_20261008.md)。

## 已冻结的唯一下一任务

[最小后段IMU关闭干预package](b0_suffix_switch_20261008_v2/package.json) SHA `6f84cef7715b94f4ea3b8df374691b48104beee3424f24a931f13fcb616e0e8d`，3174来源；9payload逐字节重建与全部来源哈希通过。CPU准备首个v1草稿因把prefix快照字典当列表读取而停止、未产出package；v1失败记录保留，最终v2才是冻结身份。不是GPU运行失败或科学对照变化。

固定主train000506/wrists和head伴随配置，每个配置都加载完整前缀v3中已保存的同一个F paired `x_899`，比较两个剩余900步suffix：继续原paired；从t899起严格设置IMU存在标志false且control零、保留原caption。保持F200、原1000步DDPM schedule、batch1 caption/batch4 MDM、原t899→1逐步噪声地址与字节，t0不抽噪声。不存在重新生成初始x999、提前重跑prefix、另选边界/权重/guidance或追加训练。

总4路径/3600次MDM调用/14400examples，encoder4、caption2、旧后验噪声重放3596（无新seed），12保存轨迹/4完整输出，7244软件检查；1200秒总wall/24GiB allocated上限，首故障停止，独占项目锁与原UUID，空余≥4GiB，高利用率只记录无门槛。仍使用用户长期GPU委托，无需再手动批准。

继续paired的2路径必须在t500/250/50的全部原数组及最终features/joints/mask复现旧F，沿用原1e−6绝对＋相对界；失败会阻断主要干预解释并保留全部缺失分母。必须先实现有界worker/supervisor并通过冻结CPU验收，然后才能创建唯一新GPU执行目录。本次只冻结设计，**未实现或执行该suffix队列**。

即使IMU-off恢复动态，也只先认定动态变化；完整交替弓步、回收/粗顺序及原数值/负例证据决定是否支持行为修复。若不恢复，不能宣称F早期历史不可逆或所有其他边界无效。干预为已暴露案例后的前瞻探索，不是盲预注册、独立审阅或泛化证据。

## CPU准备问题与证据归档

初次扩展参照分组自检遇到零GT contact能量导致的undefined ratio（None），已按原评价的planned_n/valid_n规则保留undefined，而不是转成0或剔除源；重放195772标量通过，96行和原参照指标未改。首个118492标量的基础核验和扩展失败记录均保留。suffix准备另有schema读取错误，已修正并用新v2目录冻结；均无模型/GPU调用。

代码、测试记录、冻结计划、实现、最终验证、面板与结果索引归档到[late_attribution_run_20261008_v1](../method/late_attribution_run_20261008_v1/artifact_index.json)。目前无活动GPU队列；唯一下一任务是实现并验收上述冻结的有界suffix入口，再按长期GPU委托执行一个新队列及完整评价。
