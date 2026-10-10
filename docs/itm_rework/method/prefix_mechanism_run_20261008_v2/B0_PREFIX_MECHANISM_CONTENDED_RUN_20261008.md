# B0 高利用率共享运行：实际执行、原时限停止与完整诊断评价

日期：2026-10-08。用户明确要求在GPU利用率99%时也运行；已取消利用率准入门槛，并实际执行真实模型。**队列完成3922/3960次前向，在原600秒绝对预算处停止；这次不是预检拒绝。**

## 资源政策与身份

[新政策](../GPU_CONTENDED_EXECUTION_POLICY_20261008.json) SHA256 `eb89f0f66c86a15368a3d0a4bf5db35027b380a642f2c43c04c59624cbd30d72`，保存完整用户指令。旧50%门槛只用于历史尝试；新准入仅要求3次观测空余显存≥4096MiB，GPU/显存活动率及compute进程照常记录，不因高利用率拒绝。未更改其他进程、优先级、GPU全局配置或MPS设置。

99% GPU利用率是采样时段内有kernel执行的时间比例，并非某任务占有99%的固定运算配额；调度/吞吐不保证固定比例。显存容量仍是另外的约束，并不能保证任何共享任务永不故障。[NVIDIA nvidia-smi定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)。

原科学 package `8b32b863…`、权重/12配置/旧噪声地址/4分支路径/软件容差/指标均不变。新[实现v2](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/implementation.json) SHA256 `48a12948ade4a006e70b28cdc44fab8d3bb3fff2c63d15933e815c574118eff9`，2657依赖；[来源差异](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/source_delta.json)只有新政策、预检、父入口和对应测试，科学worker/guard/原评价/原测试逐字节不变。

高利用率预检与parent政策/预算一致性9项CPU检查通过（0.94秒），沿用前次7项完整worker/故障/父入口验收，不重复无变化的完整合成队列。[CPU验收](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/CPU_acceptance.json)、[本次执行记录](B0_PREFIX_MECHANISM_EXECUTION_20261008_V2.json)、[高利用率父入口](../../../scripts/run_mdm_contended_prefix_mechanism_bounded.py)、[预检](../../../scripts/itm_contended_gpu_preflight.py)。长期GPU授权有效，无需逐次批准。

## 实际运行与停止

选定原GPU UUID，预检3次GPU利用率99/99/98%，空余7560MiB，显存活动率28/16/28%，有两个既有compute进程；4次只读NVML查询。队列实际获得计算时间，累计完成MDM **3922calls/15688examples**，E48、caption12、旧噪声重生成3636；自身峰值 allocated **533909504bytes＝0.49724GiB**。

36条前缀路径/3600步和180指定状态全部执行完成，所有48组t900历史锚点的288个数组检查、12组共享初始噪声检查通过。固定状态组321个完成元数据，1个已保存原始预测但未完成元数据，38个尚未执行；322组模型调用和raw数组可审计。7380项软件等式中 **7219completed、0failed、161not-run**。

父supervisor在原冻结600秒处终止本项目worker进程组，报告 `wall_timeout_process_group_killed`；总600.611883秒包含杀进程/账本重放清理，不是临时延长截止时间。没有OOM或软件等式错误；没有修改原失败或在同队恢复/自动重试。[原状态](../../../outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v2/runtime/status.json)、[完整账本journal](../../../outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v2/runtime/prefix.receipt.journal.jsonl)、[运行摘要](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/run_summary.json)。

另做1次只读查询，预检中的两个原PID都仍存在；这里只证明进程context仍在，不证明其训练健康/进度或未来不会故障。[查询记录](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/post_run_process_presence.json)。此阶段NVML查询总5次。

## 全分母评价与自检

按既有整队规则，**所有本队raw产物仍为 diagnostic_only**，不因前缀已完成而把整队改为完成：180状态全部diagnostic-only；360组＝322diagnostic-only＋38missing；1440分支＝1288diagnostic-only＋152missing；720权重对照＝568diagnostic-only＋152not-comparable；180路径对照和36终端全部diagnostic-only。原F行为not_supported与负例不变，此阶段主机制判决inconclusive。

[最终CPU评价v3](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v3/summary.json)及[完整artifact index](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v3/artifact_index.json)保留全部分母。独立公式重放核对2657来源/8215已保存产物、7219已执行软件等式、3600条posterior状态hash连接、3636旧噪声描述符和 **329034个指标标量**，最大绝对差 `3.73e-14`；无新模型、GPU查询或噪声生成。这是主会话另一公式自检，不是独立审阅或完整队列认证。[diagnostic验证](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/diagnostic_verification.json)。

自检发现原CPU评价的辅助 `x_t_difference_rms` 使用float32差/平方/均值，与既有float64指标约定不一致。保留原绑定[评价脚本](../../../scripts/evaluate_b0_prefix_mechanism.py)、[评价v2](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v2/artifact_index.json)和[首次核验失败](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/diagnostic_verification_first_failure.json)，另存[float64修正版](../../../scripts/evaluate_b0_prefix_mechanism_v2.py)。432个辅助RMS格修正，最大变化 `1.393e-8`；9个其余评价payload逐字节相同。模型raw、软件容差、主要MSE/时间能量及其对照不变。[精度修正核对](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v2/CPU_evaluation_precision_fix.json)。

## 仅作诊断的前缀线索

主train000506/wrists、root＋RIC67预测时间能量/GT，下表来自180指定状态；所有单元仍为diagnostic-only，不作完整机制或行为修复认证。

| t | F paired | U paired | 公共text-only |
| --- | ---: | ---: | ---: |
| 999 | 0.069870 | 0.075602 | 0.102569 |
| 975 | 0.243314 | 0.309391 | 0.209540 |
| 950 | 0.607029 | 0.916844 | 0.515337 |
| 925 | 0.407335 | 0.860360 | 0.388309 |
| 900 | 0.363517 | 0.978249 | 0.308388 |

同一初始x999与旧逐步噪声下，初始文字分支相同，之后路径已有分化；F不是从最初一直单调缩减。text-only在t900也比U低，但原完整text-only输出保留GT时间能量约0.705，而F末段约0.001，故低的早期预测能量不能单独识别终段塌缩机制。head走势不同，不能给两种配置同一个静态解释；归一化feature能量不是米或动作正确率。跨路径状态变化与同状态权重变化继续分开，缺失对照不补为成功。

![diagnostic-only前缀图](../../../outputs/itm_rework/protocol/prefix_mechanism_diagnostic_panel_20261008_v2/primary_prefix_energy.png)

[全180前缀表](../../../outputs/itm_rework/protocol/prefix_mechanism_diagnostic_panel_20261008_v2/all_prefix_panel.json)、[1288已保存同状态分支表](../../../outputs/itm_rework/protocol/prefix_mechanism_diagnostic_panel_20261008_v2/all_same_state_panel.json)。这些表来自主要payload未改变的评价v2，和修正版v3一致。

## 唯一下一任务

CPU已保存[适应高负载的资源预算设计](B0_PREFIX_MECHANISM_LONGER_BUDGET_PLAN_20261008.json)：拟在**新队列开始前**声明1200秒绝对预算，保持同一科学package/3960calls/全部检查及指标/24GiB cap，并从头运行完整队列；不把本队剩余38组补齐后拼接为完成。该设计还未实现/验收/执行，不是新的模型结果。

下一步只改独立parent/资源预算记录并CPU验收，再在长期授权下运行一个新完整队列和评价；原v1/v2及其失败分母保留，不临时修改当前截止时间，不追加训练/采样t900以下/改论文主张。高利用率准入权限已明确，不需要用户再次批准GPU。
