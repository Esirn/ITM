# 扩展开发面板：20 epoch完整训练与生成比较

本轮完成了128训练来源＋32验证来源的新数据、扩展U与两个简单时序基线的完整训练、38条DDPM动作和CPU比较。双腕相对MDM仅文本有部分误差收益；简单融合的相对姿态误差仍低于扩展U。当前不是论文最终方法认证。

## 数据与模型

- [数据计划](expanded_canonical_pilot_20261009_v1/plan.json) `95d7ed06…`：保留原6开发锚点，按seed20261009的source-group顺序扩展，一源一完整片段，train128/val32，来自13个AMASS数据集；原已知不确定6例不重新准入，无正式test。
- 复用原SMPL配对映射，160条/320次CPU FK完成。[数值核验](../../../outputs/itm_rework/expanded_canonical_pilot_20261009_v1/verification.json)160/160通过，公共10Hz关节差0，RIC解码最大9.536743e−7m，小于原1e−5m界。
- [共同输入](../../../outputs/itm_rework/method/expanded_canonical_inputs_20261009_v1/dataset.json) `e6dd7c4e…`；train14876真实20Hz帧拟合加速度统计，每源只计一次，val与两种sensor配置不进入重复统计；[1603项输入检查](../../../outputs/itm_rework/method/expanded_canonical_inputs_20261009_v1/verification.json)通过。
- 新数据使用原始全文数据集标注并核对时间支持；新增154条没有人工语义认证。扩展开发准入与原B0的逐条GT语义准入分开，新记录均 `benchmark_admissible=false`；不冒充原B0已审阅population或正式test。
- 新U保留ordered-concat、2层时序encoder、physical-sinusoidal、static-delta、strict-null及8层additive adapters；冻结官方MDM/CLIP，均匀263D normalized-x0 loss，从原共同initial重新训练，无U/F续训。16个独立真实长度图累积一次更新，不填充MDM attention。
- 两个基线是2层/256维temporal Transformer，分别native CLIP文本＋IMU和仅IMU；采用同数据/20epochs/每序列等权，AdamW lr3e−4，对比U lr5e−5。它们不是官方IMUPoser。使用观测IMU和sensor mask，无GT关节轨迹作为输入。

## 实际执行

[训练计划](expanded_canonical_training_20261009_v1/plan.json) `24d1ba4b…`、[训练状态](../../../outputs/itm_rework/method/expanded_canonical_training_authorized_20261009_v1/queue_status.json)：cuda:1，整队134.810436秒，peak695681024bytes。U320updates/5120exposures/5120MDMcalls/10240encodercalls/160captioncalls完成；两基线合计640updates/10240examples/128个完整验证输出完成。冻结backbone参数不变。

[采样计划](expanded_canonical_sampling_20261009_v1/plan.json) `eb697e6d…`、[采样状态](../../../outputs/itm_rework/method/expanded_canonical_sampling_authorized_20261009_v1/queue_status.json)：cuda:1，269.142275秒，peak531856896bytes，38samples/38000MDMcalls/44encodercalls/11captioncalls/38decodes完成。8个预选val来源×head/wrists×新U paired/MDM text g2.5＝32；3个train参考×2＝6。每对同长度、1000DDPM、同寻址噪声seed20261010、K=1。新U factorized(1,1,1)；MDM text g2.5严格bypass adapters。这是不同固定guidance设置的整体方法比较，**不能把差值单独归因于IMU模块**。

两个队列均按长期GPU委托、UUID锁、3次free≥4096MiB预检、高利用率不拒绝、3600秒全队期限/24GiB allocator cap执行，无重试/恢复或其他进程干预。

## 相同8个验证来源的主要结果

每来源等权，head/wrists分开，不合并帧；单位m。

| 方法 | head相对姿态误差 | wrists相对姿态误差 | head全局误差 | wrists全局误差 |
| --- | ---: | ---: | ---: | ---: |
| MDM text g2.5 | 0.161450 | 0.155373 | 0.671622 | 1.008493 |
| 扩展U20epoch | 0.154729 | 0.135555 | 0.754577 | 0.853106 |
| 简单文本＋IMU20epoch | 0.110945 | 0.101734 | 0.794612 | 0.817041 |
| 简单仅IMU20epoch | 0.122068 | 0.107985 | 0.769193 | 0.805537 |

新U对MDM：head相对姿态−4.16%/全局+12.35%；wrists相对姿态−12.75%/全局−15.41%。新U jerk约56.795/52.369m/s³，简单文本融合86.088/78.379；jerk低不能单独认证自然性或排除运动收缩。

两个基线在全部32val的相对姿态误差：head文本0.131042/仅IMU0.136543（约改善4.03%）；wrists文本0.122145/仅IMU0.119512（约变差2.20%）。文本在不同配置和指标上有收益与代价，单种子无显著性主张。不能把all32基线均值与val8新U均值直接比较。

新U的train参考相对姿态误差：000027 head/wrists0.1806/0.1240，000506 0.2358/0.2098，004728 0.3248/0.2761。出拳/后举仍未稳定保留旧U/F的好观感。旧3来源模型与新模型在训练数据、归一化、训练轮数和采样seed上不同，不作为单变量模块消融。弓步仍有明显动作问题；没有给新模型分配行为complete或修复成功标签。

## 核验、可视化与下一轮

[完整评价v2](../../../outputs/itm_rework/protocol/expanded_canonical_evaluation_20261009_v2/summary.json)：38生成＋128基线输出有效，无缺失；166个输出CPU独立解码检查通过，996项主要指标第二公式通过，全部320更新/5120exposure/3checkpoint有限状态核验通过，591运行文件索引保存。评价v1在首次读取NPZ后离开with块导致错误，发生在指标计算前；[失败与当时入口](../../../outputs/itm_rework/protocol/expanded_canonical_evaluation_20261009_v1/failure.json)保存，仅CPU修复后写新v2目录，没有重训/重采。

前端 `/static/expanded-review.html`，22组/104面板；[目录](../../../outputs/itm_rework/protocol/expanded_canonical_review_20261009_v1/catalog.json)。真实HTTP逐坐标一致，Node DOM/canvas 2568帧/12044面板帧通过，非真实浏览器截图。原68面板不覆盖。3个train参考接触图仅为粗查看，不是全帧盲审。

首轮20epoch不能直接代表充分训练后的方法上限。下一轮固定100epoch，各模型均同数据/初始化，原20epoch曝光、噪声、t、dropout计划保持完整前缀，不搜损失权重。已冻结[100epoch计划](expanded_canonical_training_20261009_v2/plan.json)，完成后用相同38个采样case/相同seed/noise比较，再决定方法结构取舍。新文本语义指标、官方IMUPoser/Ego4o、正式测试及核心结构消融仍未完成；论文未改。
