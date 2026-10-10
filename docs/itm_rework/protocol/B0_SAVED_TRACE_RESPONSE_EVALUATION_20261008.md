# B0 已保存 F/U 轨迹响应分解与最小前缀对照

日期：2026-10-08。实验主会话自检/公式重放，不称独立审阅。此阶段只有 CPU 数组分析，没有 GPU 查询、模型加载/前向、训练、噪声生成、采样或解码。

## 已完成的范围与身份

分析前冻结 [CPU plan](b0_saved_trace_response_20261008_v1/plan.json)，SHA256 `d64d2e1cb5a2a94bb6bcf3d8cd50543889005d21a7c246ed818b7938866c261b`，1103 个来源。复用原 F/U 的全部 60 配置/每臂、四个 t=900/500/250/50，共 **480 轨迹点、240 配对、120 完整输出，全部 valid**。七种原条件及缺失分母均保留：paired/text-only/physical-static 各12配置，IMU-only/bypass/numeric-zero/tail-reversed 各6配置，每臂同数。

原 F/U checkpoint、输入指纹、t、噪声计划、branch order、数组描述符、完整帧与返回数组指纹逐项核对；原 CFG float32 算术界和 absent/static 分支 `1e-6 + 1e-6*abs(ref)` 规则保持不变。GT 沿用原 Mean/Std 的归一化算术，不重新拟合、对齐或生成标签。

结果：[完整轨迹](../../../outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1/traces.json)、[完整输出](../../../outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1/samples.json)、[保留分母的配对](../../../outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1/pairs.json)、[分组统计](../../../outputs/itm_rework/protocol/saved_trace_response_evaluation_20261008_v1/aggregates.json)。来源等权，保留 split/config/condition/t/arm，不跨 t 池化，不把帧或去噪点当独立动作。

## 分解如何解释

对原有全部11个 feature view，将每个坐标按有效帧时间均值中心化，定义时间能量 `E(z)=mean(center(z)^2)`。四项是基础 `f00`、文字 `f10-f00`、IMU `f01-f00`、交互 `f11-f10-f01+f00`；四项能量加六个两倍协方差重构 `E(f11)`。不能只看增量 RMS：条件响应可以具有非零能量，同时通过负协方差降低总时间能量。

另报告有文字时的 IMU 变化 `E(f11)-E(f10)=2Cov(center(f10),center(f11-f10))+E(center(f11-f10))`，以及其余四个同状态增量；保留均值变化与未中心化 RMS。所有量是归一化 HumanML263 特征，不是米、真实 IMU 闭环误差或语义正确率。自由轨迹上的 GT 只是固定时间参考，不是条件期望标签。

## 主 wrists 的缩减早于现有最早保存点

train000506/wrists、paired、root＋RIC67 的时间能量/GT 如下。**F/U 输入状态不同，这张表是描述，不是同状态权重干预。**

| t | F基础 f00 | F文字 f10 | F联合 f11 | U联合 f11 | F的局部IMU能量变化/GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 900 | 0.352965 | 0.362288 | 0.363517 | 0.978249 | +0.001228 |
| 500 | 0.068693 | 0.068249 | 0.058820 | 1.233554 | −0.009429 |
| 250 | 0.029743 | 0.030084 | 0.026918 | 1.278558 | −0.003166 |
| 50 | 0.003673 | 0.003827 | 0.004431 | 1.243606 | **+0.000604** |

最早保存的 t900，F 的基础/文字分支已随该轨迹状态显著缩减。四个稀疏保存点不能定位此前哪一步首先分化；500/250 的局部 IMU 响应继续降低时间能量，但末段 t50 的响应反而略增能量。t50 中该正变化由协方差项 `+0.000122` 与增量中心化能量 `+0.000482` 构成；不是“最后一步残差把动态一次性抵消”。U 的 t50 IMU 变化为 `−0.032937`，仍保留约1.244倍 GT 时间能量，说明增量符号也不能单独认证塌缩。

已存完整输出的相同 view：F paired **0.001069**、U paired **1.111474**、两臂 text-only/physical-static **0.704812**。30个 text-only/static/bypass 完整配置跨 F/U 的特征数组全部 bitwise 相等。这个完整路径差异和局部末步分解共同支持追查早期累计状态演化；它们不识别唯一机制，也不证明关闭 IMU 能修复语义动作。

![主配置保存轨迹分解](../../../outputs/itm_rework/protocol/saved_trace_response_panel_20261008_v1/primary_saved_response_energy.png)

head 不呈相同收缩形态：主 F paired t900=1.436202、t50=1.944974、完整输出=2.019157；原“缺少中段回收”失败仍保留。train000027/head 在 t900 也有 F/U 分化（0.2422/0.6507），因此下一对照保留全12配置，不只挑主 wrists。[全96 paired轨迹表](../../../outputs/itm_rework/protocol/saved_trace_response_panel_20261008_v1/all_paired_trace_panel.json)和[72行完整 paired/text/static 参照](../../../outputs/itm_rework/protocol/saved_trace_response_panel_20261008_v1/all_final_reference_panel.json)均已导出。val拍手下肢、val000139顺序、踢腿返回、train004728保持及原 numeric-zero/IMU-only 等负例不变；不以时间能量覆盖原 F 行为判决。

## 软件验证

响应/缺失分母/中心化/掩码合成测试8项通过（0.10秒）；前缀噪声次序测试4项通过（0.07秒）。独立 NumPy 协方差/Gram 矩阵公式重放 **388080 个标量**，1103来源和7结果文件哈希通过，最大绝对计算差 `4.80e-14`，30个上述完整数组等式均通过。[验证记录](../../../outputs/itm_rework/method/saved_trace_response_verification_20261008_v1/verification.json)。这是主会话另一计算路径自检，不是新的独立实验或统计审阅。

首次验证器把小差值本身作为舍入尺度，出现 `2.22e-14` 的相减误差；已保留[原验证器](../../../outputs/itm_rework/method/saved_trace_response_verification_20261008_v1/verifier_before_context_bound_fix.py)与[失败记录](../../../outputs/itm_rework/method/saved_trace_response_verification_20261008_v1/first_verification_failure.json)，改为原有64eps等式要求的端点/协方差/能量尺度后完整通过。评价数组、评价公式和科学容差没有改变。

## 下一项机制对照已冻结，尚未实现或执行

[新前缀 package](b0_prefix_mechanism_20261008_v1/package.json) SHA256 `8b32b863da7a530cc71350384663678adfd072f8db53bbb01add176b3040853c`，2639依赖；9个 payload 字节重建和全部来源/产物哈希验证通过。不是 runtime 验收。[冻结验证](../../../outputs/itm_rework/method/prefix_mechanism_freeze_verification_20261008_v1/verification.json)。

- 原6来源×head/wrists＝12配置，从同一旧初始噪声分别运行 F200 paired、U200 paired、公共 text-only，共36短路径；完整1000步扩散 schedule 不变，仅在 t900 后截断，共100个前向/路径，保存终端 x899，不解码动作。
- 保存 t999/975/950/925/900 的180个状态；每个状态再用 F200/U200、同一 paired 输入做 factorized batch4 前向，共360固定状态组。原生 caption batch1，不走旧 Q 的 batch5 路径。保留相同 state 上的条件/权重响应与不同 path 状态演化的区别。
- 12个旧噪声计划的初始＋100个 posterior draw 地址/哈希已从 F/U paired/text metadata 核对一致。以后运行只按旧地址重生成并逐个核对，不引入新 seed。总3960 MDM调用/15840examples，E48、caption12，600秒/24GiB allocated cap，共享预检政策继续有效。
- t900 对48组原保存状态进行288个数组锚点；固定状态自身返回、不活动 adapter 与初始噪声核验共1380核心等式，另保留逐步 CFG 与 text-only absent checks。第一故障停整队，保留全部计划分母；回放不符时不得作主机制解释。

**唯一下一任务：实现该已冻结短前缀 worker/supervisor/CPU验收与评价入口；通过软件验收后按长期授权自主判断共享GPU运行，再完成全结果评价。** 不追加训练、不扫参数、不运行 t900 以下步骤。原 F 主配置行为仍为 not_supported，未修改论文主张。
