# 训练进度与动态收缩诊断：冻结及CPU准备完成

**已完成下一研究阶段的设计冻结与CPU准备，尚未运行新模型诊断。** 不追加训练、不搜索损失配重、不重采样，也不选择中间checkpoint。下一任务是按本包实现有界执行入口和增量软件测试，再按用户长期GPU授权自行执行一次。

最终采用 [v2机器包](b0_training_sufficiency_20261008_v2/package.json)，SHA256：
`471376c4799b80975c09faee4a80e2adfa220a6119a73a325761ed2a89029593`。
见 [完整设计](B0_TRAINING_SUFFICIENCY_DESIGN_20261008.md)。主会话自检/重放，不称独立02或人工验收；原F的head/wrists弓步失败结论和论文主张不变。

## 完整冻结人口

192个已有状态：全部Q96 GT加噪状态、历史U48 paired轨迹状态和本次F48 paired轨迹状态。每个状态逐一绑定原数组、target/mask、输入/caption/timestep、文件和数组哈希，分别读取相同字节比较6个快照 I/F050/F100/F150/F200/U200。四分支固定为T/P/S/R，1152组/4608预测，2880软件检查。12source/config输入case覆盖3 train＋3 development-val及head/wrists；不按误差筛选。

I使用原zero-adapter张量，仅诊断runtime reference_mode改为static_delta，不重新初始化。F四快照的身份、shape/dtype/finite、冻结bias、训练参数、optimizer步数/人口及完整训练来源均通过CPU核验；U200按原validator核验。原F1145份运行依赖均未改，合并新来源后2533份全部通过。

提出的**单次运行限额**：1152次batch4 MDM/4608examples，432次batch1 E、12次batch1 caption；单GPU/allocated≤24GiB，整队600秒，从预检前开始并含加载、保存及核验。新训练/权重/完整采样/q_sample/后验/噪声/CFG/解码/warmup/重试均0。完整人口先登记，首个规定故障停止，有限质量差不停止，失败和未执行分母保留。

固定比较包括P−S/P−R/P−T/R−S，以及同状态F050−I/F100−F050/F150−F100/F200−F150/F200−I/F200−U200。每类4608行。新增时间均值误差＋时间变化误差分解，用于描述x0预测收缩，不改变主要完整动作成功规则。没有平台/收敛阈值，没有“最好checkpoint”选择器。

## 真实已有数组中的描述性线索

以下为train000506、t50的paired trace **保存x0预测**，单位为预训练归一化HumanML root＋RIC67特征。不是完整解码动作的物理量，不是新的同状态跨权重结果。

| 状态来源/配置 | 总MSE | 时间均值误差 | 时间变化误差 | 预测时间能量/GT时间能量 |
| --- | ---: | ---: | ---: | ---: |
| U/head | 4.040329 | 1.020007 | 3.020321 | 1.787710 |
| F/head | 4.349688 | 1.146796 | 3.202891 | 1.944974 |
| U/wrists | 5.584080 | 3.508034 | 2.076047 | 1.243606 |
| F/wrists | 2.992933 | 1.919157 | 1.073776 | **0.004431** |

F/wrists预测时间能量0.004912、GT为1.108565，约GT的0.44%；时间变化误差1.073776仍接近静态GT均值参考误差1.108565。这与已审阅的完整动作收缩一致。F/head没有同样的全局时间能量收缩，保留其不同失败形态。

U与F轨迹的x_t不同，以上并列**不能归因于权重差异或训练阶段**。本包将所有快照放到每条相同状态上，补齐这一控制。局部预测进步也不能证明额外训练可修复完整弓步，或200步已充分收敛。

全部96条已有paired trace时间分解已保存，并以独立NumPy公式重算1152个view单元；没有新模型前向。见 [原数组描述性表](b0_training_sufficiency_20261008_v2/existing_trace_temporal_baseline.json)。

## 软件与重放证据

- **28 passed，3.82秒**：时间分解/静态收缩反例、分块加和、有符号差值、NA/有效mask/不跨缺帧、坏输入拒绝、固定人口、同状态6快照、缓存计数、禁止覆盖/选择/新训练，以及坏checkpoint身份/step/bias/nonfinite拒绝。真实CPU审计测试禁止nn.Module构建和GPU查询，核对7次CPU读取、6份不同文件。
- 原始来源2533份、状态192、计划组1152和checks2880逐一验证；已有96行/1152view用独立公式重算通过。自检不冒称独立审阅。
- 在新目录重建，**12份文件全部逐字节一致**，包SHA保持不变。NumPy冻结与重放入口均禁止神经框架导入和子进程启动。

证据：[测试日志](../../../outputs/itm_rework/protocol/training_sufficiency_preparation_20261008_v2/tests.log)、[NumPy核验](../../../outputs/itm_rework/protocol/training_sufficiency_preparation_20261008_v2/result_verification.json)、[checkpoint CPU审计](../../../outputs/itm_rework/protocol/training_sufficiency_preparation_20261008_v2/checkpoint_audit.json)。

v1曾完成构建和27项测试，随后发现CPU审计的读取次数字段误将“6份文件”记为“6次读取”；identity校验重读initial，真实为7次。v2更正计数，增加实际调用计数测试。旧包、原代码快照和日志原位保留；**两个版本均没有GPU执行**，人口/科学规则/拟预算完全相同。

## 唯一下一任务

实现该包的worker/supervisor：加载一次backbone、按快照交换E/adapters，caption跨快照共享但控制缓存必须换权重即清空；先保存raw再判锚点；完整预登记/计数/首故障账本和绝对期限。完成增量CPU缓存/身份/故障测试、最终依赖manifest、唯一run/GPU执行记录后，按[长期GPU委托](../GPU_EXECUTION_POLICY_20261008.json)自主安排一次预检与运行，不再向用户请求逐次GPU批准。

此时没有新worker或GPU运行结果，不得把旧V入口的完成记录充作本方案执行。后续不得从这些局部诊断直接改论文主结果、追加训练或采样中间checkpoint。
