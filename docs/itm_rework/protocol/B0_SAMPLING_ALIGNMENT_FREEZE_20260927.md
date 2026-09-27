# 02 → 01：固定 R guidance 对照的设计与评价冻结

读取 **`d2d28fb`**，承接 `c6aef58` 的全部正负结果。**采纳固定R final200、只将文本guidance从2.5改为1的最小候选；设计与评价已冻结，01可进入实现和CPU验证。** 不训练、不选checkpoint、不搜guidance，不将代数对应训练输出当作有效性证明。本包不构成运行授权。

入口：[package.json](b0_sampling_alignment_20260927_v1/package.json)，SHA256：

`537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60`

唯一权重仍为旧队列R final200 `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`。公共数据契约保持 `itm-rework-v0.3-draft`，不扩大3 train＋3 development-val准入。旧D/L/R包、授权、输出、评价及失败记录不改写。包的authorization、run directory、GPU UUID为空；新入口/实现/环境另行绑定并验收后才申请单次新授权。

## 可直接消费的文件

| 冻结文件 | 用途 |
| --- | --- |
| [design.json](b0_sampling_alignment_20260927_v1/design.json) | R身份、Q执行形状、S1唯一干预、trace、软件容差、01验收项 |
| [probe_slots.json](b0_sampling_alignment_20260927_v1/probe_slots.json) / [probe_groups.json](b0_sampling_alignment_20260927_v1/probe_groups.json) | 480分支、96组及固定顺序 |
| [probe_noise.json](b0_sampling_alignment_20260927_v1/probe_noise.json) | 原D exact有效区间噪声；不消费padding噪声 |
| [sampling_slots.json](b0_sampling_alignment_20260927_v1/sampling_slots.json) | 新60槽位、真实条件flags、输入hash、模型/旧对照映射 |
| [historical_samples.json](b0_sampling_alignment_20260927_v1/historical_samples.json) | 旧R60条NPZ/JSON及固定guidance，不重跑旧臂 |
| [sampling_noise_groups.json](b0_sampling_alignment_20260927_v1/sampling_noise_groups.json) | 12组旧完整random plan的文件/JSON指针及hash |
| [trace_slots.json](b0_sampling_alignment_20260927_v1/trace_slots.json) | 240个实际采样前向的旁路记录，额外调用0 |
| [evaluation.json](b0_sampling_alignment_20260927_v1/evaluation.json) / [review_order.json](b0_sampling_alignment_20260927_v1/review_order.json) | 所有比较、失败分母、120条新旧混排审阅 |
| [budget.json](b0_sampling_alignment_20260927_v1/budget.json) | 计数/时间硬限额、停止和保留规则 |
| [sources.json](b0_sampling_alignment_20260927_v1/sources.json) / [validation.json](b0_sampling_alignment_20260927_v1/validation.json) | 来源身份和本次CPU构造核验 |

`comparison_case_id`是本试验的旧R对照；继承的`baseline_case_id`只是更早B0的历史字段，不能误作本试验主对照。来源中的方法代码hash记录冻结时版本；01实现后的新增/变更代码由新implementation绑定并明确列出差异，不能静默改本包来源hash。固定权重、数据、历史产物和随机计划身份不因此放松。

## Q：R最终权重的96组定点诊断

6来源×2配置×4个t（50/250/500/900）×2个旧noise seed＝96组。**每组恰好一次batch-size-5 MDM前向**，顺序为 `f00, f10, f11(paired), f11(static), f11(reversed)`；合计480 branch examples，不能按96计费。所有模块eval，真实长度、无padding、不混来源组。GT归一化、q_sample及有效噪声沿用原exact D，五分支共用同一x_t；记录原t、scaled t、diffusion参数及实际张量hash。

- f00/f10使用严格null控制和false IMU presence，不调用IMU encoder；f00文本embedding也关闭。审计caption仍保存，不等于文本参与。
- 三个开启IMU的条件分别计算观测/原始O[0]静止参考，最多每条件2个encoder examples，整批上限576。先缓存条件再组batch；R差分和无bias语义不变。
- 保存全部480个float32预测、target、x_t、mask、分支/模型/输入/参考/随机身份，float64 reductions只统计有效263×T元素。输入和重复数组可按hash共享引用，不能增加前向获取缺少字段。

**Q故意没有f01，不能声称重放了实际四分支float32运算。** 只以保存的float32数组转float64构造概念对照：

`unit=f11; legacy_reduced=f11+1.5*(f10−f00)`

`a=f11−target; b=1.5*(f10−f00)`

`MSE(legacy_reduced)−MSE(unit)=2mean(a*b)+mean(b*b)`。

96组×3种IMU条件＝288个派生状态/条件记录；保留交叉项的正负，不能把非零文本项或其能量当冲突。paired−static/reversed在每种guidance下分别计算（384行），再做guidance之间的差值之差（192行）。同一组的b相同，所以**原始预测的paired−static/reversed差分会消去b，但MSE差值不必不变**。按来源先均值两次replicate，再按3来源等权，t/config/train-val分开，保留min/max和所有负值方向。

R静止f11与同batch f10是架构一致性检查；不跨batch5/trace batch4做逐位比较，也不将旧checkpoint的D作为新R的同权重对照。Q有限但差的结果不用于临时换权重或改S1；确定的软件、非有限或身份错误才停止后续队列。

## S1及240条同次前向trace

60条与旧R完全对应：val42（六条件36＋reversed6），train18（paired/text/static）。仅guidance改成 **1/1/1**；固定R权重、输入/reference/O[0]、caption、长度、K=1、seed20260926、1000步DDPM、完整noise地址、四分支batched路径及condition cache。**不能改为直接f11的快路径**，也不重新训练或生成旧2.5臂。

每条实际采样在t访问顺序 **900→500→250→50**，旁路保存已有 `x_t,f00,f10,f01,f11,actual_combined_prediction`，以及同次返回预测hash和全部身份。记录发生在posterior更新/加噪之前；detach后的独立快照只读，不替换返回张量，不消费RNG，不改cache，不多跑encoder或MDM。240个trace槽位预先登记；缺失trace不能把交付记为完整，即使motion文件已经保存，也须标注诊断不完整并保留其指标。

可以离线在这些**S1实际状态**上组合两种guidance，报告文本增量RMS/max与组合差；这不是旧2.5轨迹，也不是完整反事实rollout。不得另用GT状态调用模型补轨迹。现有无文本IMU-only在代数上不随wt变化，本干预不能凭自身修复其巨大偏离；该负例及numeric-zero均保留。

## 执行前冻结的软件容差

全部用于软件/算术一致性，**没有新增模型效果阈值**。每项同时保存连续max/RMS、bitwise和分母，不根据结果放宽。

| 检查 | 冻结规则 |
| --- | --- |
| hash、ID、mask、种子、计数 | 精确一致 |
| 原输入 | acceleration精确；orientation/normalized input/reference `atol=2e−7, rtol=0` |
| RIC | `atol=2e−6m, rtol=1e−6` |
| Q static f11/f10；trace中条件关闭的对应分支 | `atol=1e−6, rtol=1e−6` |
| trace对运行的影响 | CPU真实接口fixture中，开/关trace的输出、RNG、cache、计数逐位相同；同次实际combined与实际返回预测hash相同 |
| 完整采样 | 最大坐标差≤1e−5m：新text/bypass 6组、新static/text 12组、新旧IMU-only 6组，共24项 |

float32四分支组合不要求unit与f11逐位相同。使用原逐项组合式，并与float64参考比较，固定逐元素绝对误差界：`gamma16 * max(1,S)`，其中 `u=2^-24, gamma16=16u/(1−16u)`，

`S=|f00|+|wt|(|f10|+|f00|)+(|f01|+|f00|)+(|f11|+|f10|+|f01|+|f00|)`。

这是按操作数规模预先定义的舍入界，不是观测结果后调出来的容差；保存最大error/bound、原始差及finite count。Q的float64 MSE恒等式误差界固定为 `64*eps64*max(1,MSE_legacy,MSE_unit,|cross|,energy)`。全部数组/必要reduction有限；越界作为软件/重放差异停止后续并保留失败，不冒充动作质量负例。未完成对应槽位的检查应记not_comparable，不能补造通过。

## 评价与解释

全部60条新输出加旧R60条对照，按冻结原指标直接对20Hz GT，无空间/时间拟合；val、train、配置、条件分别列n=3来源等权，缺失明确partial n并保留原分母。jerk必须与GT一起看；接触只用于val000433/000382，val000139和全部train为N/A。关节代理不能称为生成IMU一致性。

- 对全部60个匹配条件记录 **S1−旧R**，而非只列paired。
- 在每个guidance内部比较paired−text、paired−static；val加paired−reversed、paired−zero。每个guidance36个配对差值，共72；其S1−旧R差值之差共36。保留原量、逐来源差值、extent/path、输出间平均关节距离，后者只代表敏感性。
- 新旧共120条按预登记seed20260930混排，隐藏guidance/config/condition，原粗动作目标可见。保存10个均匀快照和全帧手距/脚高/root轨迹，先记录观察与uncertain再解码。旧结果已知，不称严格盲评/独立人工验证；不计算新semantic成功率或JSR。
- 重点保留train000506的交替弓步与回收顺序、004728的**右**腿后抬方向/幅度/保持、val000382 wrists的非目标下肢动作、000139两配置步行及000433完整踢腿返回；不能仅凭global降低或更接近新text-only通过认证。

若Q额外文本项增加误差，且S1自由生成相应改善，可作为这个开发面板上guidance失配的有限支持；仍需相对**本guidance的**文本/静止/倒序对照判断动态收益。若Q joint本身差，或训练动作仍不能自由生成，本候选未解决能力问题；若Q好但自由采样差，仍是待解释的状态/rollout失配，不证明唯一机制。不同来源/t方向不一致时完整报告混合结果，不自动续训或扩数据。

仍只有3 train＋3 exposed development-val、固定训练seed、K=1；无泛化、显著性、多样性、严格IMU闭环、相容编辑、真实设备或封存测试认证。本轮是基线诊断候选，不作为新研究贡献。

## 限额与01的下一步

| 阶段 | 硬上限 |
| --- | --- |
| 新训练/新checkpoint | 0 / 0 |
| Q | 480 MDM branch examples、≤576 encoder examples、600秒 |
| S1 | 60×1000步、60000 denoising calls、240000 MDM branch examples、≤120 encoder examples、共享2400秒 |
| trace | 240记录、额外MDM/encoder调用0 |
| 整队 | 240480 MDM、≤696 encoder；单GPU、allocated≤24GiB、**3600秒** |

总时钟在预检/加载前启动；每阶段deadline取阶段和总deadline较早者，加载/保存/检查全部计时。阶段上限合计3000秒，余下总期限不构成额外模型工作额度，不跨阶段挪用。预登记480/60/240槽位；前向前保留计数，首次超时/OOM/非有限/身份/计数/软件一致性错误停止，保存已完成和failed/not-run，不重试、换卡、CPU回退或插入真实warmup。

**下一责任方：01按本包实现新版本入口和CPU故障检查，交付代码/实现manifest；02再做独立软件验收。随后01申请绑定最终实现、本包、唯一新目录和GPU的新授权，获授权后才预检并单次执行。** 本次无需重复询问是否同意开始实现，旧预算不能续用。CPU验收须覆盖真实加载器、batch5 Q分支、共享x_t、null/参考语义、trace无副作用、final200身份、480/60/240分母和阶段/总期限的故障保留；本次12项协议测试不能替代尚未实现的方法测试。

## 本轮完成的CPU验证

[构造器](/home/a200/0proj/ITM/scripts/freeze_b0_sampling_alignment.py) 只用文件/NumPy，不导入神经模型。核对 **726份来源文件**、旧R60槽位和输入/随机身份，重建 **96个Q有效噪声＋12000个唯一采样noise draw**。新增[协议测试](/home/a200/0proj/ITM/tests/test_b0_sampling_alignment_protocol.py) **12 passed**，覆盖分母/条件、原噪声、trace、预算、混排、代数正负例及多个数量级的舍入界。新目录重建的 **14个包文件逐字节相同**。

完整命令、测试与重建hash：[checks目录](b0_sampling_alignment_freeze_checks_20260927_v1/commands.md)。本轮真实模型加载/前向、训练、采样、GPU查询、授权均为0；方法与旧评价文件未改写。首次真实执行后若必须改变本包，另出版本并说明影响，不能覆盖已冻结文件。
