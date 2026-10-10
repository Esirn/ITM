# 共享GPU下完成固定状态诊断：轨迹收缩线索，不认证训练充分

**已有计算进程不妨碍低负载共享。本轮已在其他进程持续存在的GPU上完成完整诊断，139.13秒、allocated峰值0.507GiB，全部1152组/4608主分支/2880软件检查通过。** 用户明确允许共享使用；新预检不再因进程存在自动拒绝，采用当前资源观测，保留科学规则、唯一目录、绝对期限与首故障停止。

**研究结论：不能将完整弓步失败直接归因于“200updates欠训练”，也不能据此直接追加updates。** 主wrists在GT加噪状态上预测动态仍接近GT；在已经收缩的F轨迹末段状态上，I和旧U也预测近静态。不同状态人口仍不是受控的分布偏移因果比较。原F完整动作评价的head/wrists不支持修复结论不变，没有新完整动作、checkpoint选择或论文主张。

## 共享资源与完整性

| 项目 | 实际证据 |
| --- | --- |
| GPU | `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`，RTX4090，原cuda:0对应物理GPU |
| 其他compute进程 | 预检**1个**，保持原有进程，不停止/修改它 |
| 三次资源观测 | free=23389MiB；GPU utilization=42/38/39%；memory-controller utilization=0/0/0% |
| 自身allocated峰值 | **544594944 bytes，0.507GiB**；24GiB是上限，不是预留量 |
| 总期限/实际耗时 | 600秒，预检前起算；**139.132800秒**，含加载、编码、保存与核验 |
| 状态/组/主分支/check | **192/1152/4608/2880全部completed**；缺失0 |
| MDM/E/caption | 1152调用/5184examples；432调用/432examples；24调用/72examples |
| 辅助f00 | 576个，只恢复原生Q数值路径，不增加主要行为/成功指标 |
| 新训练/完整采样/噪声/解码 | 全部0；6个已有权重，无中间checkpoint选择 |
| 产物/验证 | 3300运行产物，2570绑定依赖全部核验；4608预测、4608臂内、4608同状态跨权重对照全部valid |

[共享GPU政策](../GPU_SHARED_EXECUTION_POLICY_20261008.json)来自用户新明确指令。当前门槛是连续3次、间隔2秒，free≥4096MiB，GPU与显存控制器利用率均≤50%；原始查询stdout/stderr保存。利用率是近期活动代理，不是精确空闲SM数量或吞吐保证。GPU上有进程本身不再拒绝，科学/OOM/非有限/计数/身份/超时/软件故障仍首故障停止。

最终[原生路径v4包](b0_training_sufficiency_20261008_v4/package.json) SHA256：`95283f4c9744c6624ca3d4e6f0d2a199427f85e50edb2c975039512208b0416a`。
[实现manifest](../../../outputs/itm_rework/method/training_sufficiency_native_implementation_20261008_v1/implementation.json) SHA256：`cb30fb6d035e96c129ea67c5924ee8ef4c35648a97cb2077c7385227b7a41f28`。
[具体执行记录](../method/B0_NATIVE_TRAINING_SUFFICIENCY_EXECUTION_20261008.json)由主会话按既有长期委托和新共享指令创建，无逐次手动批准。

运行：`outputs/itm_rework/method/training_sufficiency_native_authorized_20261008_v1`。
评价：[全部主分支](../../../outputs/itm_rework/protocol/training_sufficiency_native_evaluation_20261008_v1/branches.json)、[臂内对照](../../../outputs/itm_rework/protocol/training_sufficiency_native_evaluation_20261008_v1/within_contrasts.json)、[跨快照对照](../../../outputs/itm_rework/protocol/training_sufficiency_native_evaluation_20261008_v1/checkpoint_contrasts.json)、[3来源等权聚合](../../../outputs/itm_rework/protocol/training_sufficiency_native_evaluation_20261008_v1/aggregates.json)、[完整性](../../../outputs/itm_rework/protocol/training_sufficiency_native_evaluation_20261008_v1/summary.json)。

## 历史Q数值路径核验与修正

共享预检首次实际放行的v2队列完成961次MDM/3844examples，109.58秒、allocated0.502GiB后，按原1e−6界在`U200-S001-historical_T`停止。max_abs=0.000123620、RMS=0.000017573。保留960completed/1failed/191not_run、2018checks通过/1失败/861未执行，有限输出全部diagnostic_only；[部分队列评价](../../../outputs/itm_rework/protocol/training_sufficiency_evaluation_20261008_v2/summary.json)没有充作本次成功结果。

代码核对发现旧Q使用**原生CLIP batch5＋MDM batch5**，新诊断缓存是CLIP batch1＋MDM batch4。真实文本、state/checkpoint相同也不保证原生半精度CLIP在不同batch下逐位相同。不是GPU资源不足或模型行为退化的判定。

另冻结并执行单状态路径核验（不是恢复旧队列）：4次MDM/18examples、2次caption/6examples、6次E，120秒上限，实际11.29秒。原生旧路径A与历史Q的5个预测**逐位一致**；新路径C与失败队列4个预测也**逐位一致**，9个强制锚点全通过。改变caption batch、保持MDMbatch5时，预测max_abs变化0.000200748；CLIP embedding batch1/batch5最大差0.00390625。固定caption、改变MDM batch也有约1.9e−6小差异，其中一个组合越过原界，因此修正恢复**完整**原生路径，未只修caption或放宽容差。

见[独立路径结果](../../../outputs/itm_rework/method/caption_path_authorized_20261008_v1/probe/result.json)、[冻结计划](../../../outputs/itm_rework/method/caption_path_implementation_20261008_v1/plan.json)及[执行记录](../method/B0_CAPTION_PATH_EXECUTION_20261008.json)。这是主会话的受控路径自检，不是独立审阅。

原生方案显式冻结：Q仍用5分支f00/T/P/S/R，caption原生batch5；轨迹仍用T/P/S/R batch4、caption batch1。24个caption cache（两条数值路径×12case）；E仍按case/checkpoint共享，432次。主要状态/权重/4608行/2880checks/指标/容差不变；新增576辅助f00使MDMexamples由4608变5184，caption由12调用/12examples变24调用/72examples，整队600秒不变。v3准备曾继承旧2533来源计数，v4更正为2543冻结来源；v3未执行，数据/路径/配额不变。旧包、失败目录、原代码快照均保留。

## 同状态的主wrists结果

train000506、paired、t50，以下是归一化root＋RIC67特征MSE与预测时间能量/GT比例。Q先平均两个固定噪声；U/F轨迹各一个已有状态。**在同一人口内比较权重时，输入state完全相同；跨人口并列是描述量，不是因果状态分布实验。**

| 固定状态人口 | I | F150 | F200 | U200 |
| --- | ---: | ---: | ---: | ---: |
| GT-noised Q：MSE | 0.014007 | 0.006788 | 0.006457 | 0.007103 |
| GT-noised Q：时间能量/GT | 1.071234 | 0.993401 | **0.986352** | 1.011492 |
| U轨迹：MSE | 5.645092 | 5.612711 | 5.499613 | 5.584080 |
| U轨迹：时间能量/GT | 1.276542 | 1.238279 | 1.222916 | 1.243606 |
| F轨迹：MSE | 3.037175 | 3.048482 | 2.992933 | 3.019056 |
| F轨迹：时间能量/GT | **0.003827** | 0.003782 | **0.004431** | **0.004337** |

F200在GT加噪参考状态保留约98.6%的GT时间能量，局部预测较I改善；但在已收缩的F轨迹末段状态上，I/F/U均约0.4%。不能将其解释为只有F200权重在最后一步瞬时引发静态输出，也不能宣称权重完全无关：生成这些轨迹状态的完整采样过程不同，尚未做受控完整轨迹干预。

同一F保存轨迹的四个t上，F200 paired时间能量/GT为t900=0.363517、t500=0.058820、t250=0.026918、t50=0.004431；I/U也显示相同人口上的变化。这是不同t保存状态上的描述，不将4个状态当4个独立动作，也不从中识别单一因果机制。

![同状态跨权重曲线](../../../outputs/itm_rework/method/training_sufficiency_native_implementation_20261008_v1/primary_same_state_progress.png)

全主head/wrists及所有6快照数值见[两配置同状态表](../../../outputs/itm_rework/method/training_sufficiency_native_implementation_20261008_v1/primary_same_state_table.json)。head不呈现与wrists相同的时间能量收缩形态，原缺少中段回收的完整动作失败仍保留；不以“有时间变化”认证正确动作。

## 训练充分性与负例边界

最后50updates没有统一单调改善。GT-noised Q、paired、3 train来源等权且两噪声先在source内平均，F200−F150 balanced差值如下（负为局部参考MSE改善）：

| 配置 | t50 | t250 | t500 | t900 |
| --- | ---: | ---: | ---: | ---: |
| head | +0.000097 | +0.000272 | −0.000159 | −0.005001 |
| wrists | +0.000993 | +0.000942 | +0.000281 | −0.011815 |

因此单一主source某个t的下降不能代表整个train面板已收敛，或追加训练会修复完整动作；没有新增平台阈值、显著性检验或checkpoint挑选。所有source/config/t/family/branch保留，Q两种噪声不作为两个人，3来源均值/min/max/有效n/计划n仍不跨t或split池化。

val000382拍手下肢、val000139顺序回归、val000433踢腿返回、train004728右腿保持、原numeric-zero/IMU-only等负例继续保留，未用本诊断覆盖原F120行审阅。GT是固定同时间参考，不是自由采样状态的条件期望标签。所有特征分解均为归一化HumanML单位，不是米或真实IMU闭环误差。

## 软件重放与下一研究任务

共享预检23项CPU增量检查通过；单状态路径5项fixture检查通过；原生队列3项CPU检查包含完整1152fixture组/2880checks、24caption/72examples、5184MDMexamples及576辅助f00；父入口3300必需产物从真实fixture receipt核验。继承已有故障/超时/日志/身份测试，不称新统计独立验收。

最终用独立NumPy公式核验2570依赖、3300产物、2880强制软件等式、4608时间分解、9216对照，共152064个view单元；另576个辅助f00对历史Q全部通过。没有新前向或GPU查询。见[结果重放](../../../outputs/itm_rework/method/training_sufficiency_native_implementation_20261008_v1/result_verification.json)。

**唯一下一研究任务：先只读分解已保存F/U轨迹的f00/f10/f01/f11响应及时间能量，追踪动态缩减在去噪过程何处出现，再冻结最小轨迹机制对照。** 不直接追加训练、不扫损失系数、不重新采样中间checkpoint。GPU低负载共享使用权限继续有效，无需逐次手动批准。无新模型效果/泛化/真实设备/相容编辑/严格IMU闭环或论文主结果认证。
