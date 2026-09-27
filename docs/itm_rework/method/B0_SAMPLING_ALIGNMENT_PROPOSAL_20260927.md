# 01 → 02：固定 R 权重的最小采样修正提案

读取 **`c6aef58`**。接受本批结论：交付完整，模型有效性仍未证实；L的步行→持续抬腿、R双腕拍手的下肢退化、训练000506/004728未可靠生成均须保留。**下一步建议先固定R final200，只检验采样 guidance `2.5/1/1 → 1/1/1`，不重新训练。** 配套480个R最终模型定点探针及60条完整采样，拟上限单GPU / 24GiB allocated / 总1小时。

这是具体开发提案，**尚未由02冻结，也不是授权或已验证修正**。本轮只用CPU读取旧数组/训练记录、核对代码及NumPy代数；预训练模型加载/前向、训练更新、扩散采样、GPU查询均为0。没有修改模型/运行器、公共协议、输入或旧结果。

## 现有证据与尚缺证据

1. R减轻旧共同偏置，但步行恢复本身不能证明动态IMU收益：text-only本来就会步行。R head的局部正面迹象和wrists拍手的负面结果必须同时保留。R−L包含参照差分与去bias，不能进一步拆分为单一成分因果。
2. 不能据“训练误差下降”说训练动作已学会。现有D仅使用旧B0 `44d2c330…`；**没有L/R final200在固定GT噪声上的单步诊断，也没有旧R自由轨迹的逐分支预测**。400个update记录是在变化的权重、t、噪声、dropout下产生的，不能把首末均值当作固定人口收敛曲线。
3. 本轮逐项核对两臂各600条呈现仍匹配冻结计划。每臂条件人口为joint495、text-only58、IMU-only43、both-off4；这不是充分优化的证据，也不支持仅扩大数据覆盖就能解决训练动作失败。
4. 原输入与静止参照不是相同数组。本轮仅从旧NPZ计算active归一化特征差的RMS（无量纲）：

| 来源 | head paired−reference RMS | wrists paired−reference RMS |
| --- | ---: | ---: |
| train 000027 出拳/转步 | 0.9140 | 1.0138 |
| train 000506 交替弓步/蹲起 | 0.3172 | 0.2147 |
| train 004728 右腿后抬保持 | 0.2499 | 0.6564 |
| val 000433 踢腿返回 | 0.2138 | 0.3658 |
| val 000382 拍手 | 0.0380 | 0.5965 |
| val 000139 步行 | 0.2838 | 0.2327 |

这只排除“这些输入与参考数值相同”，不说明信号可辨识、编码器利用充分或属于训练分布。head拍手信号比wrists弱，不能要求同等手部控制。完整12项输入统计及02对应原始指标见 [existing_evidence.json](b0_sampling_alignment_20260927_v1/existing_evidence.json)；没有重新定义公共指标。

## 为何先检验 guidance

`run_mdm_correction_worker.train()` 的joint呈现监督单个原始联合输出 `f11(x_t,t,u,c)` 的normalized-x0 MSE。采样沿用四分支公式：

`f = f00 + wt(f10−f00) + wi(f01−f00) + wj(f11−f10−f01+f00)`。

当前 `wt/wi/wj=2.5/1/1`，实数代数上为：

`f_current = f11 + 1.5(f10−f00)`。

因此IMU-only项抵消；不能仅用IMU-only训练样本少来解释paired劣化。额外文本项也不是代码错误或已证实根因：预训练骨干使用文本CFG有其既有行为，而联合输出受到IMU残差影响后，两者的误差方向未必相容。当前保存结果不足以判断。

**S1候选固定 `1/1/1`，在实数代数上还原 `f11`**。它是一个与训练输出对应的结构性取值，不做guidance网格、按片段挑权重或再选checkpoint。float32组合可能有舍入差，不能承诺与直接单分支调用逐位相同；仍使用相同四分支batched路径，不趁本轮更换执行方式。

此改动也会把text-only/static的文本引导从2.5改为1，所以必须重算同尺度对照及差值之差，不能只比较新paired与旧paired。对真IMU-only，文本被关闭、`f10=f00`，该改动在代数上无影响；**它不能单独解决R wrists IMU-only已有的巨大偏离**。该负例继续保留，用作固定权重/噪声下的预期不变检查。

## Q：R final200的固定定点诊断，480 branch examples

唯一权重：`outputs/itm_rework/method/correction_authorized_20260927_v2/R_static_reference_delta/model.pt`，SHA256 **`91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`**。不重新训练、不选epoch，也不以Q结果改变下面S1配置。

沿用原D exact-length的12个source/config、t=`{50,250,500,900}`、两个noise replicate，共96组相同GT噪声状态；保留原noise_key、GT、normalization、长度和参考语义。每组5个真实branch examples：

- `f00`：文本/IMU都关闭；`f10`：只开文本。
- `f11(paired)`、`f11(physical_static)`、`f11(tail_reversed)`：文本相同，分别使用三种原输入变换。

合计 **96×5=480**，全部eval、逐组同一x_t、真实长度、无padding实验。IMU encoder只为三种开启条件计算观测/参考，上限96×3×2=576 examples；f00/f10用严格null，不另行编码。新Q不是旧B0探针的重复，直接检验产生当前负例的R最终权重。

保存每个分支的float32预测、x_t、target、mask、条件/模型/噪声身份。从同一组已保存分支离线组成两个guidance的输出，不增加MDM调用。对每种u定义 `a=f11(u)−x0`、`b=1.5(f10−f00)`，记录：

`MSE(f_current,x0) − MSE(f11,x0) = 2 mean(a*b) + mean(b*b)`。

所有来源/config/t/noise逐行保留，train/val分开；paired−static/reversed与两种guidance的差值之差均列出。static的f11与f10应在既定数值容差内相同，这是R架构检查；不是学得的能力。Q是给定GT噪声状态的诊断，不能替代自由生成。

明细：[proposed_probe_slots.json](b0_sampling_alignment_20260927_v1/proposed_probe_slots.json)，状态为提案。

## S1：同一R模型，60条完整采样

一对一复用旧R的60个槽位：42 val（原36＋6个tail-reversed）和18 train（3来源×2配置×paired/text-only/static）。不删000433、不只选两个显眼负例；train没有旧reversed采样，保持双方同条件人口，不临时补不对称对照。

仅guidance变为 `1/1/1`。固定同一R checkpoint、原MDM、input/reference/O[0]、caption、长度、K=1、seed=20260926、全部DDPM噪声地址、1000步、四分支batched执行。旧R `2.5/1/1` 的60条产物作为完整历史匹配对照，**不重跑旧臂、不生成新权重**。新case ID与旧comparison_case_id显式映射，random_plan_case_id完全保留。

在60条新采样实际经过的t=`{50,250,500,900}`，旁路保存本次前向已有的四分支输出、x_t、实际组合输出及身份，共240个trace记录，**额外模型调用为0**。纯读取并detach，不改返回张量、RNG、缓存或梯度状态；不能再用GT状态调用模型“补轨迹”。可在S1轨迹的相同状态上离线重组旧guidance，观察项的尺度/方向；这不是旧2.5轨迹，不能宣称获得两条历史轨迹的逐步因果比较。

明细：[proposed_sampling_slots.json](b0_sampling_alignment_20260927_v1/proposed_sampling_slots.json)。每项都绑定旧R实际产物、同一模型hash和完整输入/噪声身份。

## 请02冻结的评价与解释边界

沿用既有连续指标、无拟合的20Hz GT、原contact适用范围和逐条动作审阅；不由01设置新的“成功率”或事后数值阈值。需预先固定以下比较与容差：

1. 对每个source/config比较S1与旧R的paired，并分别在**各自guidance下**比较paired−text-only/static；val再比较paired−reversed，报告这些差值的S1−R差值之差。原12个source/config全列，train与val分开，保留所有负值方向和分母。
2. 必须审阅000506的交替/回收序列、004728的右腿方向/幅度/保持、000382 wrists的非目标下肢动作及000139两配置的步行。位置、root-relative姿态、速度/幅度与动作同时看；仅global误差变小或输出靠近新text-only不能认证动态收益。
3. 数值检查沿用原null/bypass、R-static/text-only最大坐标差≤1e−5 m并报告bitwise；IMU-only跨guidance预期不变、Q静止分支/组合代数与trace原样回传的dtype容差由02在执行前固定，不因实际结果放宽。保留numeric-zero与IMU-only六条件负例，不能取消这些槽位来美化表现。
4. 若Q显示额外文本项增加误差且S1自由生成改善，记为支持本开发面板上的guidance失配假设；**仍须动态对照优于对应文本/静止/倒序并且动作正确**，不能直接升级为泛化或严格IMU闭环结论。
5. 若Q的joint本身差，或S1仍不能生成训练动作，则本候选失败；后续再讨论优化/条件表示/目标或时变注入，不自动加训练时长或扩数据。若Q好而自由采样仍差，只能定位到尚需解释的状态/采样失配，不能凭本轮证明唯一机制。若仅R先验改变，仍不认证动态控制。

这60条继续属于已参与选择的开发面板，单seed/K=1。改动只是一项基线诊断/修正候选，不作为新研究贡献，也不代替后续同类实例donor和独立封存评价。

## 拟预算与实现顺序

| 项目 | 拟硬上限 |
| --- | --- |
| 新训练 / 新checkpoint | **0 updates / 0** |
| Q | 480 MDM branch examples、576 IMU encoder examples、600秒 |
| S1 | 60条×1000步；60000 denoising calls、240000 MDM branch examples、120 encoder examples；共享2400秒 |
| trace | 240记录，从上述采样前向中取得；额外调用0 |
| 合计 | 240480 MDM branch examples、最多696 encoder examples；单GPU、allocated≤24GiB、**3600秒**从预检前起算 |

所有加载/保存/验证计入阶段和总时限，不转移剩余额度。首次超时/OOM/非有限值/身份或计数异常停止整队，保留480/60完整分母及已完成产物，不重试、换卡或CPU回退。有限但差的Q结果不用于临时选择S1参数。旧队列未用完的预算不续用。

额度参考上一轮实耗：480个D探针约133秒、120条采样约2227秒。本提案的R探针编码成本不同，不能线性保证耗时；因此仍单独保留10分钟Q、40分钟采样和1小时整队硬上限。

次序：**02冻结具体评价/槽位/噪声/容差 → 01实现新版本入口及CPU故障检查 → 02软件验收 → 01申请绑定最终实现、唯一目录和GPU的新授权 → GPU预检/单次执行 → 02独立评价。** 本轮不请求提前授权。

实现只需在新协议入口显式消费固定guidance及R checkpoint，增加Q分支导出和无额外调用的trace观察；原D/L/R运行器保持历史语义，不能直接复用它的旧授权。CPU检查需覆盖两组guidance组合、严格null、trace不改变输出/随机流、最终权重身份、480/60计数和共享限额故障保留。

机器预算：[proposed_budget.json](b0_sampling_alignment_20260927_v1/proposed_budget.json)。CPU生成器：[build_proposal.py](b0_sampling_alignment_20260927_v1/build_proposal.py)；复现：`OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/b0_sampling_alignment_20260927_v1/build_proposal.py`。它只读取已有JSON/NPZ和文件hash、计算NumPy统计/代数、写入01提案，不导入或调用神经模型。输入证据身份见 [sources.json](b0_sampling_alignment_20260927_v1/sources.json)，计数/代数核验见 [build.log](b0_sampling_alignment_20260927_v1/build.log)。

最终提案一致性核对：[verification.json](b0_sampling_alignment_20260927_v1/verification.json)。60项历史槽位的未修改字段逐项相同，480项分支/96组噪声身份、预算计数与424份来源文件hash通过；这是提案检查，不是新模型实验或软件执行验收。原始生成时曾遇到历史新增槽位没有`original_case_id`的字段差异，现以完整旧`case_id`前缀替换并显式保留comparison_case_id；没有补造旧字段或修改公共槽位。
