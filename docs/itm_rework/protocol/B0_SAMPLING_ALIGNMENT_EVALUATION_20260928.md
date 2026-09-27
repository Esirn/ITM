# 02 → 01：固定 R guidance 对照独立评价

读取 **`f6da7a8`**，按 `1680618` 冻结协议和 `a90d3e7` 软件验收评价全部480 Q、60新采样、240 trace及60旧R对照。**交付、身份和数值一致性通过；guidance 2.5→1带来局部改善与退化，未解决完整动作能力和稳定动态IMU收益，B0联合控制有效性仍未认证。** 不能把额外文本引导认定为统一根因，也不能据位置均值下降宣布修正成功。

本轮仅CPU读取既有数组、指标重算、绘图及单AI粗审阅；新模型前向、训练、扩散采样、GPU查询/预检均为0。未修改方法代码、冻结条件/阈值、旧产物、数据准入或授权。**02评价交付完成，下一责任方为01：针对下述具体负例提出最小后续诊断/修正；不追加运行。**

## 完整性与独立重算

- **480/480 Q、60/60 S1、240/240 trace完整**，60旧R对照全部保留。新样本val42/train18，旧对照同分母，失败/缺失均为0；有限但错误的动作仍是有效输出，未剔除。
- 独立核对implementation绑定743文件和运行目录1805文件，共2548个路径；61份receipt对应480＋240000个MDM分支、696个encoder examples，更新0，均在绑定deadline/显存上限内。1621份receipt产物hash通过，840个NPZ、3900数值数组、91,329,180元素全部有限。
- 原始30Hz IMU经独立插值重建20Hz，核对480 Q和60 S1输入、旧R对应输入、原始O[0]静止参考、caption、GT及mask；398帧train-only统计一致。固定R final200 `91744ac5…`、冻结包 `537895c4…`、implementation `793fdd9a…` 不变。
- 12组/12000个唯一完整采样noise draw重建通过；新旧plan一致。96组Q噪声及cosine→identity SpacedDiffusion构造独立重建，全部保存x_t最大差 **0**。96组Q static/text逐位相同。
- 重算288个Q条件分解、384个组内差值、192个差值之差；保存原始480分支MSE和逐来源/t结果。240 trace的实际返回hash、四分支组合、缺席条件等式通过，最大error/bound **0.068403**。
- 24项完整样本检查与方法报告的max/RMS/count/reference逐项完全一致：18项逐位相同，其余6项最大坐标差 **1.66893e−6 m**，保持原1e−5 m界。S1 CPU RIC重放最大差 **1.90735e−6 m**；含旧R的120条最大差 **2.86102e−6 m**，均满足原atol2e−6、rtol1e−6。
- 18项评价解析测试通过；最终评价运行时禁止nn.Module前向与CUDA查询。先完成的样本/差值与最终完整CPU重跑逐字节一致；旧R60条指标与前批评价完全相同。开发期语法及schedule构造错误的修正记录保留，未归为方法失败、未放宽阈值。

耗时2043.614606秒（34.06分钟）、allocated峰值535660544 bytes（0.499GiB）取自原运行记录，本次没有查询GPU。该单次授权已经消耗。

证据：[运行核验](b0_sampling_alignment_evaluation_20260928_v1/execution_audit.json)、[样本/分母/指标](b0_sampling_alignment_evaluation_20260928_v1/samples.json)、[全部配对与差值之差](b0_sampling_alignment_evaluation_20260928_v1/contrasts.json)、[24项重放](b0_sampling_alignment_evaluation_20260928_v1/consistency_replay.json)、[命令及修正记录](b0_sampling_alignment_evaluation_20260928_v1/commands.md)。

## Q：训练与验证的方向不同

下表为paired的 `MSE(legacy_reduced)−MSE(unit)`；正值表示额外文本项增加给定GT噪声状态上的预测误差，负值表示降低。每来源先平均两次noise，再对3来源等权，不跨t合并。

| split/config | t50 | t250 | t500 | t900 |
| --- | ---: | ---: | ---: | ---: |
| train head | +0.0001365 | +0.0001775 | +0.0003758 | +0.0061182 |
| train wrists | +0.0001537 | +0.0001857 | +0.0002515 | +0.0057100 |
| val head | −0.0000136 | −0.0000194 | −0.0001202 | −0.0003222 |
| val wrists | −0.0000605 | −0.0001546 | +0.0000598 | +0.0028218 |

**额外文本项并非总是有害。** train各格均值支持去掉它可减小此定点误差，但val head各t均值反而受益于原额外项；val wrists随t变号。分解中的误差交叉项有正有负，不能把非零文本能量当作冲突证明。Q没有f01，以上是保存float32预测转float64后的概念重组，不是原四分支float32采样的重放。

**训练条件的定点响应确实存在，却没有转为可靠完整生成。** unit paired相对static/reversed，train两配置、三来源在四个t的两噪声均值均更低；例如t900来源均值paired−static为head −0.04919、wrists −0.06970，paired−reversed为−0.05425/−0.08564。但000506自由输出仍无法完成交替弓步，故不能据Q改进宣布训练动作已经掌握。

val head paired−static均值改善主要来自000139，000433和000382各t反而为正；paired−reversed在t50/250/500均值更差，t900仅−0.0000446。val wrists paired−static在四个t均值为 **+0.00871至+0.01461**，尤其000382各t为 **+0.02409至+0.04891**；paired−reversed四个t均值也更差。这与腕部拍手的自由输出负例相符，不能只归因于额外文本引导。

原始数值、来源min/max、288分解及864派生/对照行见[Q结果](b0_sampling_alignment_evaluation_20260928_v1/probes.json)、[Q对照](b0_sampling_alignment_evaluation_20260928_v1/probe_contrasts.json)。共同文本增量在原始paired−comparator预测差中消去，但平方误差差值不必不变；全部差值之差仍保留。

## 完整自由采样：改善主要集中于步行的全局位置

位置单位m、速度m/s；每格val3来源等权。无平移/yaw/时间拟合，head/wrists指标只是关节位置代理。static与同臂text-only相等，因此表中以text-only代表该数值基线。

| config | guidance/condition | Global MPJPE | Root-relative | Root RMSE | Joint proxy | Velocity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| head | oldR paired | 0.72900 | 0.12446 | 0.74292 | 0.69083 | 0.86391 |
| head | S1 paired | 0.44564 | 0.13235 | 0.41157 | 0.44943 | 0.88404 |
| head | S1 text/static | 0.52508 | 0.14334 | 0.51562 | 0.53823 | 0.90240 |
| head | S1 reversed | 0.43087 | 0.12475 | 0.39815 | 0.43768 | 0.84463 |
| wrists | oldR paired | 0.91420 | 0.19643 | 0.86503 | 0.96923 | 0.71767 |
| wrists | S1 paired | 0.81904 | 0.20370 | 0.77188 | 0.88752 | 0.73754 |
| wrists | S1 text/static | 0.81216 | 0.13357 | 0.84512 | 0.95355 | 0.80232 |
| wrists | S1 reversed | 0.82139 | 0.20198 | 0.79157 | 0.88736 | 0.81290 |

- 相对旧R，S1 paired的global均值head下降0.28336、wrists下降0.09517；但relative分别**上升0.00790/0.00727**，velocity也**上升0.02012/0.01986**。head全局改善来自000139的−0.87710，而000433/000382分别变差+0.00437/+0.02264。wrists步行−0.46860抵消了kick **+0.21995**，不能只报告均值。
- 文本基线本身也变化：head text-only的global均值从0.75461降到0.52508，wrists从0.80717变为0.81216；因此新旧paired差值不能独占为IMU收益。
- S1 head相对自身static，global/relative/velocity均值改善0.07945/0.01099/0.01837；但相对reversed全部变差 **+0.01477/+0.00761/+0.03940**。000139对static有局部收益，对reversed global/relative仍变差+0.03129/+0.00961；并未证明时间顺序一致的稳定收益。
- S1 wrists相对static，global **+0.00688**、relative **+0.07013**；拍手000382单例global **+0.25592**、relative **+0.20672**，审阅仍见双腿抬起/屈曲。相对reversed的global均值仅−0.00236、relative仍+0.00172；该拍手单例两项均更差。
- 两配置IMU-only保持数值不变，wrists global仍 **2.28276m**；这些巨大偏离未因text guidance变化得到修复。numeric-zero等全部负例也保留。没有文本时改变wt不应带来改善，这一等式是软件事实。
- S1 paired jerk head/wrists为44.21280/35.60305，GT为30.48664；旧R为47.24833/33.51624。合法GT-contact速度代理由0.13839/0.12543变为0.15404/0.19904（每格仅n=2）。不把较低jerk单独解释为更好；000139及所有train contact均为N/A。

全部40格、60个S1−oldR配对、两臂共72个内部对照及36个差值之差见[完整表](b0_sampling_alignment_evaluation_20260928_v1/tables.md)和[逐条CSV](b0_sampling_alignment_evaluation_20260928_v1/per_case.csv)。所有轴向extent、pelvis path、输出距离及其差值保存在JSON；输出距离只表示敏感性。

## 训练动作与120条固定顺序审阅

train paired全局/relative均值：head由0.62045/0.21809变为 **0.85650/0.19392**，wrists由0.40435/0.14320变为 **0.35254/0.12403**。两配置相对自身static有指标收益，但依然不能将其解释为完整训练动作已学会。

1. **000506交替前弓步**：S1 head是初始踏步后长期低坐/低蹲；wrists短暂站起后再次持续低姿态，没有完整交替及回位。新旧global分别变差+0.05633/+0.01601，wrists relative虽改善0.03826，不能抵消行为缺失。
2. **004728右腿后抬保持**：S1 head出现短促多次高抬而非稳定保持，root移动增大，global比旧R差0.60461；wrists仍有右腿高举、保持、放下，global/relative改善0.04523/0.02137。这是保留的局部正面迹象，但后抬方向与过高幅度仍uncertain，不能认证整项动作正确。
3. **000382双腕拍手**：持续抬腿/下肢变形未消除。head姿态较稳定，但手臂结束动作也不完全明确。**000139步行**两配置paired均有粗步行，静止/文本也能步行；不能将步行出现本身归为动态IMU控制。
4. **000433踢腿回位**：S1 head晚段高抬腿，到末帧仍未充分回落；wrists多为低幅脚部动作，清晰完整kick-return未确认。全部numeric-zero/IMU-only异常保留。000027存在出拳/转身粗迹象，也有额外跳跃和顺序uncertain。

20页图均按冻结120行混排，显示10个均匀姿态及全帧手距/脚高/root曲线，隐藏arm/config/condition。全部观察先写[原始笔记](b0_sampling_alignment_evaluation_20260928_v1/review_notes_hidden.txt)，再生成[标签映射](b0_sampling_alignment_evaluation_20260928_v1/review_unmasked.json)。旧结果已知且读取schema时见过S001映射，**不是完全盲评或独立人工验证**；单AI、投影姿态/曲线不足以认证细节/接触/自然度，不报告semantic成功率。图路径/hash见[绘图清单](b0_sampling_alignment_evaluation_20260928_v1/review_rendering.json)。

## Trace与后续责任

240 trace支持检查实际S1状态上的四分支行为：paired在t900的额外文本增量RMS来源均值，train head/wrists为0.08553/0.08729、val为0.06346/0.07643；t50分别约0.01285/0.01314及0.00810/0.00859。文本项确实会改变局部预测，但**非零不等于冲突**；没有旧guidance轨迹，也没有用这些状态做完整反事实rollout。trace不提供独立因果或自由输出GT误差证据，见[全部trace统计](b0_sampling_alignment_evaluation_20260928_v1/traces.json)。

**本候选没有解决B0关键能力问题，也不支持继续无目标地搜guidance。** 当前证据指向至少两项仍需区分的问题：train已有GT噪声状态响应，但自由生成缺失完整动作；val腕部000382在unit Q及自由输出中均劣于static，不能只用采样额外文本解释。01下一步应先利用现有预测/trace/训练记录，给出能区分这些现象的最小诊断或修正提案，列清预期证据、保留负例和拟预算。不能直接宣称“扩大覆盖”或“继续训练”已获因果支持。

若确需新模型前向/训练/采样，再由02按具体提案冻结评价并申请新授权。当前不追加作业，不复用消耗过的预算；无新增加载器/输入/限额软件返工项，无需重复往返验收。边界仍为3 train＋3已暴露development-val来源、单训练seed、K=1，不作泛化/显著性/多样性、严格生成IMU闭环、相容编辑或真实设备认证。
