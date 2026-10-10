# 固定状态训练进度与动态收缩诊断：设计冻结

本方案回答：在**相同保存状态、相同输入和相同目标**上，F 的训练进度是否继续改变局部预测；整体误差下降是否伴随时间变化收缩；P 对静止/倒序输入的局部收益是否保留。它不是续训方案，也不证明 200 步充分、追加训练有用或某个中间 checkpoint 可以修复完整动作。

最终采用v2冻结包。v1构建与27项CPU测试已通过，但复核发现CPU审计把6份不同checkpoint误记为6次反序列化：共同initial在identity校验时重读一次，实际7次。v2更正为6份文件/7次CPU读取；人口、比较、权重、模型/GPU调用与拟预算不变。v1包和原代码快照保留，未执行任何GPU任务；不复用或覆盖它。

已有 F 完整评价关闭该候选：train000506 head/wrists 数值门槛通过而弓步不完整，wrists 收缩；其他10个paired配置9个数值回归。原行为与数值联合判据、负例、caption、数据准入及论文主张不变。见 [F评价](B0_FEATURE_BALANCE_EVALUATION_20261008.md)。本设计是合并主会话的自检冻结，不称独立02验收。

## 固定人口与唯一变化量

| 内容 | 固定人口 |
| --- | ---: |
| GT 加噪 Q 保存状态 | 96：6来源 × 2配置 × 4t × 2既有噪声种子 |
| 历史 U paired 轨迹保存状态 | 48：同6来源/2配置/4t |
| 本次 F paired 轨迹保存状态 | 48：同6来源/2配置/4t |
| 权重快照 | I、F050、F100、F150、F200、U200，共6份 |
| 状态/快照组；T/P/S/R预测 | 192×6=1152组；4608分支 |

三个人口在不同状态上，不解释人口之间的误差差距为分布偏移的因果效应。跨 checkpoint 比较必须读取同一条状态的**完全相同字节**；只改变已保存 E/adapters 权重。来源不按误差筛选，保留3 train与3 development-val，head/wrists分别报告，t为50/250/500/900。所有历史Q噪声已生成并落盘，本轮不调用 q_sample、不重建噪声。

每个状态绑定 x_t、GT target、有效mask、原NPZ/JSON和哈希。四分支遵循已执行V的方法：T真实caption且严格IMU null，P原paired，S零raw加速度＋原始O[0]恒定姿态，R保持第0帧的尾部倒序 `[0,T−1,…,1]`；reference始终原始O[0]，mask/time/caption不变。T保留输入数组用于审计，但不编码E、不注入bias。没有额外numeric-zero或IMU-only探针，其旧完整行为负例仍引用原F/U120行。

I直接加载旧initial的权重，adapters全零。旧元数据reference_mode=none保留；**仅诊断runtime view改为static_delta**，与F/U一致。不创建新初始化、权重文件或随机过程。CPU审计确认六份权重不同、F优化器step分别50/100/150/200、张量键/shape/dtype/finite一致、冻结adapter bias不变；参数漂移不是收敛证明。

## 原始调用、缓存和软件界

复用旧非链式MDM loader。backbone/CLIP加载一次，快照间仅交换审计过的E/adapters，全部eval、float32控制和denoiser、无AMP/梯度。CLIP维持旧native dtype及float32 embedding返回；原1000步identity timestep map。每组原始MDM batch4一次，无CFG合成、后验步、解码、warmup或验证额外前向。

每个source/config的caption按原batch1路径编码一次，跨t、人口和快照共享，共12次。控制缓存按checkpoint/case/input哈希/归一化/config/dtype/eval绑定；**每次换权重清空控制缓存**。每快照12case×P/S/R各观测和reference两次batch1，共432次E调用。T控制严格为零，E调用0；不把不同快照的E输出错误复用。

预登记2880软件检查：1152个S对T，192个I的P对T，960个其余快照T对相同状态I的T，384个U200/Q的T/P/S/R对旧Q预测，96个U200/Utrace的T/P对f10/f11，96个F200/Ftrace的T/P对f10/f11。**逐元素界仍为 `1e−6+1e−6*abs(reference)`**，报告bitwise/max_abs/RMS/count/finite/pass。原Q为batch5，本轮为batch4，不预设bitwise一致，也不允许失败后放宽界。输入重建沿用V的orientation/normalized atol=2e−7、rtol=0，其余输入精确身份一致。

所有有限原始预测先独立落盘，再判断锚点。检查失败保留raw、标记diagnostic_only并停止整队；不得将已有输出记成未执行。保存每快照/case控制与caption缓存身份、原状态引用、目标和mask描述、完整账本、预留/实际/完成计数、耗时、allocated峰值与停止原因。

## 预先固定评价与解释

4608分支报告uniform、D67、A196、固定balanced=.5D67+.5A196、原5主块和RIC3子块。4608个臂内对照为每组P−S/P−R/P−T/R−S。另4608个同状态跨权重对照固定为F050−I、F100−F050、F150−F100、F200−F150、F200−I、F200−U200，各192状态×4分支；不挑最好的中间快照。

所有差值在float64重算 `e=baseline−GT,d=prediction−baseline, ΔMSE=2mean(e*d)+mean(d*d)`，保留交叉项、残差能量和原预测差RMS/max/count。5主分区按dims/263只加和一次，RIC3为子分区，不与主RIC重复加和。GT是固定时间索引参考，**不是自由轨迹状态的条件期望标签**。

为诊断收缩，新增描述性时间分解，**不更改旧主要成功规则**：每个坐标在有效帧上求均值，`MSE(p,g)=MSE(mean(p),mean(g))+MSE(p−mean(p),g−mean(g))`；后项等于预测时间能量＋GT时间能量−2协方差。独立恒等式使用64eps float64界。报告这两类误差、两类时间能量、协方差、能量比和有效相邻帧差RMS；GT能量为0时比例NA，不跨无效帧形成邻接。GT时间均值只是静态参考，不是新模型输出，也没有拟合/时空对齐。单位仍为归一化HumanML特征，不是米/物理IMU误差。

若平均误差下降而预测时间能量收缩、时间误差未改善，可记录这些x0预测的动态收缩线索；结合已失败完整动作解释，不能断言唯一原因是静态条件平均解。若固定Q继续改善而轨迹状态不改善，只支持局部状态依赖；不能单独判定分布偏移的因果作用。若P−S/P−R有利而完整弓步仍失败，只说明局部收益不足。

逐source/config/t/family/branch/checkpoint保留行。Q两噪声先在source内等权平均，再按同split/config/t/family的3来源等权报告mean/min/max、planned_n=3、valid_n和缺失理由。不跨t、人口、配置或split池化，不将相关状态作为独立生成样本。F150→F200的方向/幅度是训练进度描述；**没有收敛/平台阈值，不自动授权追加updates或完整中间采样**。

全部train000506、train004728右腿保持、val000382拍手下肢、val000139顺序回归、val000433踢腿返回和旧numeric-zero/IMU-only负例继续保留。无泛化、显著性、新动作成功率、真实设备、相容编辑或严格IMU闭环认证。

## 一次运行边界与交付顺序

| 限额 | 数量 |
| --- | ---: |
| MDM调用 / examples | 1152 / 4608 |
| batch1 E / caption调用 | 432 / 12 |
| GPU / allocated | 1 / 24GiB |
| 整队绝对期限 | 600秒，从预检前计起，含加载、保存和核验 |
| 新训练/权重/完整采样/噪声/解码 | 全部0 |

预检前登记192状态/1152组/4608分支/2880checks。首个身份、超时、OOM、非有限、计数或软件失败停止全部；有限质量差不提前停止。无warmup/重试/fallback/换卡/期限重置，保留所有失败/未执行分母。

先交付冻结包、CPU审计与可重放验证；下一步实现有界worker/supervisor，完成增量身份/缓存/故障测试和最终依赖manifest，再创建唯一目录及具体GPU执行记录。用户已长期授权GPU由主会话自主判断，[授权记录](../GPU_EXECUTION_POLICY_20261008.json)适用，不需再次请求手动批准。协议冻结本身不是已运行证据；尚未实现入口不得伪称可执行或启动旧V入口替代。
