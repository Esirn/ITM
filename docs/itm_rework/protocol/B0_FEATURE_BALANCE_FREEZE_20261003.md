# 02 → 01：F root＋RIC 损失配重设计与评价冻结

**接纳 `2b8a842` 的单一候选提案，冻结F：只把训练目标中的root＋RIC与其余特征各赋予一半权重。** 从原共同初始化训练200 updates，新增60条完整采样，与既有R final200的60条unit-guidance S1输出逐条配对。历史对照不重训，尚未实现或运行F，效果未验证。

**下一责任方为01：按本包实现并做CPU验证 → 02增量软件验收 → 01申请新的单次授权。** 当前可推进实现，不能启动训练、采样或GPU预检。拟限额单GPU／24GiB allocated／整队1小时，授权、目录和GPU身份均为空；所有旧授权不能复用。

## 冻结包与身份

[package.json](b0_feature_balance_20261003_v1/package.json) SHA256：
`a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`。

| 项目 | 身份与范围 |
| --- | --- |
| 候选arm / loss_version | `F_root_ric_balanced` / `normalized-x0-root-ric-half-v1` |
| 原initial.pt | `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401` |
| 历史U的R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |
| 历史完整输出 | S1的unit guidance `(1,1,1)`，60条；不是更早guidance2.5输出 |
| 公共契约/数据 | v0.3-draft不变，限定合成面板；398个唯一train帧、20Hz输入及原归一化 |
| 新checkpoint | `itm-feature-balance-control-v1`，本次final200唯一采样入口，hash执行后绑定 |

包共16文件：design、loss、evaluation、budget、原样training_schedule、sampling_slots、historical_samples、sampling_noise_groups、trace_slots、software_checks、within_contrasts、review_order、target_review、validation、sources及package。所有已知输入、参考、caption、旧输出/metadata及随机计划已逐项绑定；新F权重尚不存在，不借用R hash或schema冒充新权重。

## 02的独立核查

提案573份来源声明均通过，结合历史实现、输入和评价依据共绑定1,125份去重来源。独立重算192个保存V分支的四种损失视图及192项差值，最大浮点差1.78e−15（预设1e−12内）。训练主要48项比较中，确有5项整体误差下降而root＋RIC上升；这是旧预测的重新汇总，不是新模型效果或梯度冲突证据。

原200条update记录与600条训练计划逐字段匹配；独立重建600条训练噪声及12组共12,000条采样噪声，hash全部一致。新60个槽位保留历史真实长度、caption索引、presence、unit CFG、全部初始及逐步随机地址，train18／development-val42。没有从结果中挑选来源或噪声。

**9项协议测试通过，0失败/跳过；提案7 JSON和冻结包16文件均在独立临时目录重建逐字节一致。** 首次开发构建因旧train/reversed槽位没有可选`original_case_id`而在输出前停止，已改为从完整历史case ID保留后缀，原异常留档；未改旧产物。详见[validation](b0_feature_balance_20261003_v1/validation.json)、[重建证据](b0_feature_balance_freeze_checks_20261003_v1/replay_verification.json)、[测试日志](b0_feature_balance_freeze_checks_20261003_v1/tests.txt)和[复现命令](b0_feature_balance_freeze_checks_20261003_v1/commands.md)。

本轮只有文件/NumPy及GT绘图，没有导入模型框架、反序列化checkpoint张量、模型前向、训练、采样或GPU查询。权重只读取字节核对hash；PCG64是旧噪声身份的CPU重放，不是新实验采样。

## 唯一训练变化与实现契约

有效帧数N，每条片段误差e为完整标准化x0误差：

`L_F = 0.5 * mean(e[0:67, valid]²) + 0.5 * mean(e[67:263, valid]²)`。

位置组D为root4＋RIC63，其余A为rotation6d126＋local_velocity66＋contact4。等价原263维均值的逐通道权重D=`263/134`、A=`263/392`；都大于0，总和263。完整263维权重和hash已在[loss.json](b0_feature_balance_20261003_v1/loss.json)冻结。运行中不搜索系数，不按条件、来源、关节、时刻改变权重，也不删通道。

保持原R static-delta、共享参照梯度/匹配encoder dropout、strict null、冻结未用bias、backbone frozen_eval/native CLIP精度、真实长度、全部参数集合/顺序和AdamW设置。F读取原initial.pt及tensor身份，使用全新空AdamW状态；**不从R final200续训、不重新抽初始权重**。lr=5e−5、weight_decay=.01及其他默认值按绑定common保留；无AMP、梯度裁剪、scheduler或早停。

每次更新原三条真实长度片段，各自L_F/3累加一次optimizer.step；null无可训练梯度的样本仍占三分之一分母，保持原skip-backward记录，不将其他样本重新归一化。目标权重总和不变不意味着梯度范数或AdamW步长相同，后者属于该干预的效果。

600条exposure从**同一次预测**记录L_U、L_D、L_A、L_F、实际loss/梯度图状态和原随机身份；不得增加诊断前向。变化中的t/noise/dropout/权重使首末平均loss不能当固定人口收敛证据。50/100/150快照仅保留，唯一final200可采样；不足200步则停止，不以中间权重替代或恢复重跑。

新checkpoint及样本元数据必须绑定新package/implementation、loss_version、完整权重/hash、initial文件与tensor/原包身份、训练计划hash、参数集合与冻结状态检查、optimizer配置、归一化/输入来源、环境及final200计数。拒绝旧R重贴标签、缺失/不符损失来源或错误初始化/计划/环境。01的CPU验证要用独立解析梯度/有限差分检验67/196分区、有效帧和样本分母，并验证统一权重退化到原损失/梯度/更新；生产仍只允许冻结F权重。

## 完整采样、软件等式和trace

全部60个新槽位：train三来源×两配置×paired/text/static＝18；val三来源×两配置×原六条件加tail_reversed＝42。K=1，DDPM1000步，factorized batched CFG `(1,1,1)`，clip_denoised=false；完整263维照常进入sampler，不改推理通道、gain、噪声地址或CFG执行路径。

[48项软件检查](b0_feature_balance_20261003_v1/software_checks.json)预先固定：

- F physical_static与F text_only：12项。
- F backbone_bypass与F text_only：val6项。
- F与历史U的同条件text_only12、physical_static12、backbone_bypass6：30项。

这些原始完整motion的max_abs界均为**1e−5米**，无空间/时间拟合，记录bitwise/max/RMS/count/finite_count。缺失对照为not_comparable，不能记通过。**F/U IMU-only不要求相等**：本次改变训练权重，其条件响应可以改变；旧guidance实验的IMU-only不变性不适用于F。该条件仍全部保留作效果负例。

每条四个t900/500/250/50，共240 trace，从原实际batch4前向旁路保存x_t、f00/f10/f01/f11、combined/returned预测、mask与全部身份，额外模型/encoder调用为0。输入/解码、同trace缺席分支等式、float32组合舍入界继承旧冻结定义；旧Q相关算术容差不意味着本次有Q任务。F与U轨迹x_t不同，不能按同状态锚点要求相等，也不能把跨轨迹trace差解释成受控反事实。没有新Q/V队列。

## 主负例的行为与数值判据

主问题是train `000506` 的完整交替前跨下蹲/弓步，head与wrists**分别判断**。02额外核对了该裁片的GT支持：[目标注释及全帧曲线](b0_feature_balance_20261003_v1/target_review.json)、[GT图](b0_feature_balance_freeze_checks_20261003_v1/gt000506.png)。

**重要边界：GT第0帧已在左侧前跨屈膝姿态，末尾仍下沉，不是从站直到站直的完整循环。** 冻结的可见行为为：初始左侧领先姿态的恢复、换到右侧前跨下蹲及中段回收、保留末端下沉。不会强求裁片之外的左腿起始迈步或最后站直，也不会截掉末端来美化F。GT描述窗口为[0,22)、[22,36)、[36,67)、[67,106)、[106,136)；它们只帮助核对目标顺序，不是生成帧自动分类阈值，不做时间配准。

保持全帧pelvis高度/轨迹、左右踝/趾相对pelvis XYZ及绝对Y、膝角曲线；原canonical＋Y为高度、＋Z为前后位移proxy，无每样本拟合朝向。骨长退化的角度记NA，train没有新的地面/接触准入。持续低蹲、仅蹲浅一点、没有换侧或缺少GT可见的中段回收仍是行为不完整。判为complete/incomplete/uncertain/missing，含糊证据不升级为complete。

**候选的局部支持判据在F输出前固定：**每个配置必须同时满足：

1. F paired在上述可见行为上complete，配对历史U paired为incomplete；旧结果已知，仍按新旧120行固定顺序共同审阅并保留原报告。
2. F paired相对U paired的global MPJPE、root-relative MPJPE、root路径RMSE、配置关节位置proxy、velocity proxy RMSE这5项全部不退化，且root-relative严格改善。
3. F paired相对自身text/static同样5项全部不退化，且root-relative严格改善；非零输出距离或静止等式本身不算动态收益。

“不退化”采用保守**零退化幅度**，只给浮点比较松弛 `ε=1e−6+1e−6*abs(reference metric)`（各指标原单位）；new≤reference＋ε，严格改善为new<reference−ε。这是本开发面板候选的预设操作规则，不是校准过的感知阈值、统计非劣性或B0通用成功标准，不根据结果放宽或换权重。任何uncertain/missing组成项记inconclusive；观察到行为或门槛失败则不支持该配置的修复主张。姿态/速度指标没有改善，即使balanced loss下降也不通过。

其余10个source/config的paired均用同5项F−U检查退化，并审阅行为损失；任一回归必须单列，不能声称整个面板已修复。000382 wrists拍手下肢与双手、000139步行对倒序排序、000433踢腿返回、004728右腿方向/保持/幅度及wrists有限粗正例、000027出拳转身、所有numeric-zero/IMU-only全部保留。只改善train或一个配置只能报告相应范围的局部证据。

## 全部评价与审阅人口

保持原20Hz GT与指标定义，无yaw/translation/time拟合：全部60对F−U；每臂paired−text/static，val再加paired−reversed/numeric_zero，共每臂36项/合计72项；对应36项差值之差。全部关节/根/velocity、jerk、motion extent、足接触proxy及输出间距离保留；contact只在原val000433/000382适用，val000139和所有train仍NA。没有生成IMU闭环指标或新成功率。

同时保留完整输出的原uniform误差、D67、A196、balanced及原5＋RIC3分块，不能只报新训练目标；trace仅解释局部响应，不替代完整动作。运动范围缩小不自动视为收益，必须结合GT幅度、速度、行为和所有差值。train/val、配置/条件分别按3来源等权均值、min/max、有效n与计划分母报告，不池化帧数或合并配置。

[review_order.json](b0_feature_balance_20261003_v1/review_order.json)将F60＋U60以PCG64 seed20261003固定混排。图中隐藏arm/config/condition，显示原粗目标；十个固定均匀姿态与全帧曲线/完整序列审阅，120行notes先保存后解码。已有U结果已知、单AI粗审阅，不宣称完全盲审或独立人工验证。旧负例不删；有缺失也保留原行。

## 拟预算与停止条件

| 阶段 | 数量上限 | 时间 |
| --- | --- | --- |
| 训练F | 200 updates／600 exposures，MDM600调用/examples、encoder1200 examples、caption600次 | t0＋600秒，t0在预检前 |
| 完整采样 | 60×1000步，MDM60000次batch4／240000 examples、encoder最多120 examples、caption60次；240 trace额外调用0 | 共享3000秒，受全局deadline约束 |
| 全队 | MDM60600次／240600 examples，encoder最多1320，caption最多660；1个final＋最多3个中间快照 | 单GPU／24GiB allocated／t0＋3600秒 |

采样期限为min(采样起点＋3000秒，全局期限)，训练未用额度不追加到采样。预检前登记200更新/600呈现/60样本/240trace/48检查；首次timeout/OOM/nonfinite/身份/软件等式/计数/写入失败停止整队，保留完整分母与已完成/有限原始产物。无warmup、重试、恢复、换卡、fallback或补采；有限但动作错误不触发筛除/停止后续槽位。

本包只是设计和评价冻结。01先交具体实现、CPU独立梯度/统一权重/来源与故障测试、最终manifest；02做新增损失及入口的增量验收后，再绑定唯一新目录、GPU和本包申请新授权。若无法保持历史配对身份，停止交02修订，不自动加跑同期U控制臂或把其他改变都归因于配重。

当前没有修复成功、泛化、显著性、严格IMU闭环、相容编辑或真实设备证据。该单一系数若不足以形成目标行为，则报告本候选不足，不自动搜索权重或追加训练。
