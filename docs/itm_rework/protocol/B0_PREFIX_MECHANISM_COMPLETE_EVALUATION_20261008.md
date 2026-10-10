# B0 完整前100步对照与保存预测差异分解

日期：2026-10-08。用户要求继续已准备的1200秒预算方案。**新队列从头完整完成，3960前向/7380软件检查全部通过，所有计划评价槽有效；旧超时产物仍保持diagnostic-only，未恢复或拼接。** 此阶段没有训练、完整动作解码或论文主张修改。

## 身份、预算和执行

科学package仍为 [原前缀协议](b0_prefix_mechanism_20261008_v1/package.json)，SHA256 `8b32b863da7a530cc71350384663678adfd072f8db53bbb01add176b3040853c`；F200/U200原权重、12配置、caption batch1/MDM batch4、旧噪声地址/100步posterior顺序、各检查和容差均不变。

用户continue对应[预声明1200秒预算修订](../method/B0_PREFIX_MECHANISM_1200S_BUDGET_20261008.json)，SHA256 `6170c5a16a3110f1776304b3691ee50b81311d9a33cc01979ac499d8b026a4ab`；它只覆盖新尝试的壁钟时限，保留24GiB allocated cap、首故障停止、独立目录、不干预其他进程和高利用率准入。

[最终implementation v3](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/implementation.json) SHA256 `be04358cbeca4d35b4aeed2d3222f4f1013105cf6ab930e011eb8c16a113194f`，2659依赖。[父入口](../../../scripts/run_mdm_long_prefix_mechanism_bounded.py)的4项CPU检查通过（8.75秒）：1200秒修订身份/不变科学计数，父和worker同一deadline且不重置，1200边界在worker启动前停止并保留全分母，真实CPU准备绑定修正版评价且不构建模型或查询GPU。沿用已有7项worker/故障验收＋9项资源验收，不重跑无变化的完整合成队列。[验收](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/CPU_acceptance.json)、[源码差异](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/source_delta.json)、[执行记录](../method/B0_PREFIX_MECHANISM_EXECUTION_20261008_V3.json)。

真实运行 `prefix_mechanism_authorized_20261008_v3` **517.943504秒**，peak allocated **533909504bytes＝0.49724GiB**。预检空余23389MiB，GPU活动率40/41/40%，有1个既有compute进程；此时的低活动率是实际观测，不是恢复了旧利用率门槛。4次NVML只读查询，没有自动等待空闲或干预他人。

MDM3960calls/15840examples、E48、caption12、旧噪声重生成3636，各reserve/actual/completed完全一致。36路径、3600步、180状态、360固定状态组/1440分支、7380等式全部completed；无失败或未执行分母。[队列状态](../../../outputs/itm_rework/method/prefix_mechanism_authorized_20261008_v3/runtime/status.json)、[摘要](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/run_summary.json)。

## 全评价和公式重放

[评价v4](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v4/summary.json)：180轨迹状态、360组/1440主分支、720同状态权重对照、180路径对照、36终端全部valid，沿用已修正的float64辅助RMS。[全部产物](../../../outputs/itm_rework/protocol/prefix_mechanism_evaluation_20261008_v4/artifact_index.json)。valid表示执行/来源/软件等式可用，不表示动作质量。

独立计算路径重放2659依赖、8292worker产物、全部7380软件等式（含48组历史t900锚点/288数组）、3600条状态hash连接、3636旧噪声描述符和 **357876指标标量**，最大绝对差 `3.73e-14`；没有额外前向/噪声/GPU查询。[验证记录](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/result_verification.json)。主会话自检，不称独立审阅或统计显著性。

## 前缀阶段事实

主train000506/wrists的归一化root＋RIC67预测时间能量/GT：

| t | F paired | U paired | text-only |
| --- | ---: | ---: | ---: |
| 999 | 0.069870 | 0.075602 | 0.102569 |
| 975 | 0.243314 | 0.309391 | 0.209540 |
| 950 | 0.607029 | 0.916844 | 0.515337 |
| 925 | 0.407335 | 0.860360 | 0.388309 |
| 900 | 0.363517 | 0.978249 | 0.308388 |

相同初始x999和逐步噪声下，F/U/T的初始文字分支相同，随后路径已分化。F先增加再下降，不能称全前缀单调缩减；text-only在t900也低，却在原完整输出保留约0.705倍GT时间能量，F完整输出约0.001。因此早期低能量本身不是终段塌缩的充分判据。head前缀形态不同，原完整动作缺少中段回收的失败不因较高动态能量而改变。

![完整前缀](../../../outputs/itm_rework/protocol/prefix_mechanism_panel_20261008_v3/primary_prefix_energy.png)

[全180状态表](../../../outputs/itm_rework/protocol/prefix_mechanism_panel_20261008_v3/all_prefix_panel.json)、[全1440同状态分支表](../../../outputs/itm_rework/protocol/prefix_mechanism_panel_20261008_v3/all_same_state_panel.json)。GT只作固定时间参考，不是自由轨迹的条件期望标签；特征能量/RMS是归一化HumanML单位，不是米或IMU闭环误差。

## CPU附加分解：局部差异与已累积的状态差异

该附加分析在已有v2部分诊断暴露后设计，在读取完整v3效果前冻结，不伪称盲法预注册，也不新增主行为判据。采用[CPU plan v2](b0_prefix_attribution_CPU_20261008_v2/plan.json)，SHA256 `8a2e6e2925e1fc000876890b8cbbc9166158de872c521aec856ec74261f70a21`：6动作来源/12配置、5个原定t、F−U与F−text两对照、两种参考顺序，共240行，全部valid。未执行的v1把12配置写作source_count，v2只明确6来源/12配置；选择器和计算不变，v1保留。

记`P(w,x)`为paired预测，`T(x)`为严格无IMU文字预测。F−U的一种顺序为：

`P(F,xF)−P(U,xU) = [P(F,xF)−P(U,xF)] + [P(U,xF)−P(U,xU)]`。

第一项是在同一输入state切换checkpoint的即时预测差；第二项是在U权重固定时，已有路径state变化引起的输出差。另一顺序先改变state再在xU切换权重；两种都报告。F−text同理分成固定state的IMU条件差和固定条件的state输出差，另保留反向顺序。直接/状态分量的平方能量和两倍协方差重构总差异能量，并共同重构GT参考MSE变化；有符号协方差使它们不能解释为相加的“贡献百分比”。

主wrists、root＋RIC67、F−U、直接项在xF评估的差异RMS如下：

| t | 同状态checkpoint差 | 固定U权重的state输出差 |
| --- | ---: | ---: |
| 999 | 0.024610 | 0.000000 |
| 975 | 0.054351 | 0.081438 |
| 950 | 0.053542 | 0.240376 |
| 925 | 0.042049 | 0.429239 |
| 900 | **0.033956** | **0.573888** |

t900另一顺序为0.033487/0.569601；head两顺序分别0.030314/0.214052与0.030842/0.211718。全12配置在t900、两顺序的state项RMS都大于即时checkpoint项，但幅度因source/config而异，不作显著性认证。

这支持继续追踪去噪过程的state演化。**不能据此宣称权重无关或证明唯一状态分布机制**：这些state本身正由此前不同权重/条件的完整前缀演化产生；算术分解与同state局部操作没有独立识别“过去权重的因果贡献”。主wrists F−text t900的即时条件差0.156243、state输出差0.565590，也不能从能量/RMS大小单独认证IMU移除会修复动作。

![两顺序差异分解](../../../outputs/itm_rework/protocol/prefix_attribution_panel_20261008_v1/primary_algebra_rms.png)

[全部240行](../../../outputs/itm_rework/protocol/prefix_attribution_evaluation_20261008_v1/rows.json)、[分组均值/min/max/n](../../../outputs/itm_rework/protocol/prefix_attribution_evaluation_20261008_v1/aggregates.json)、[紧凑全表](../../../outputs/itm_rework/protocol/prefix_attribution_panel_20261008_v1/all_algebra_panel.json)。5项合成代数/抵消/均值/掩码/顺序测试通过（0.07秒），另一次CPU角色交换断言通过；独立Gram/dot-product及分组公式核验 **108720标量**、300个已有预测输入，最大绝对差 `4.44e-15`。[分解验证](../../../outputs/itm_rework/method/prefix_mechanism_implementation_20261008_v3/attribution_verification.json)。新模型/GPU/噪声调用均0。

## 研究判断和唯一下一步

完整前缀与代数队列已可用，早期路径分化和即时/累积差异的描述有支持；**完整弓步行为修复仍为原not_supported，未认证充分训练、泛化、唯一因果机制或论文主结果**。原val拍手下肢、val000139顺序、踢腿返回、train004728保持等负例和既有numeric-zero/IMU-only面板不变。

已保存[晚段CPU计划](b0_late_attribution_CPU_20261008_v1/plan.json)：只读复用已完成原生固定状态诊断中的F/U paired states，在t900/500/250/50、6来源/12配置做同一双顺序分解，共96行，并读取既有text-only后段能量曲线作描述参照。不需要任何新前向/采样/噪声/解码。

**唯一下一任务：实现并完成该96行CPU晚段分解和text-only参照，核验全部来源/身份/代数后，再选择最小GPU干预。** 不重跑已完成前缀、不追加updates/配重搜索、不中间checkpoint采样，不凭本阶段结果直接改方法或论文主张。
