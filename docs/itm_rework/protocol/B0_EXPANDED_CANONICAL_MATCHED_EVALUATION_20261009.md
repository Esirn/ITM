# 扩展U完整开发实验：100epoch与统一文本引导对比

**完成128训练来源/32验证来源的数据扩展、U与两个简单基线的20/100epoch训练，以及98条实际完整DDPM生成。统一text CFG2.5后，新U在head全局/根轨迹和双腕相对姿态上有收益；双腕全局轨迹和jerk仍明显较差，不能作为整体优越或任务已修复的认证。** 本轮把工作推进到真实完整方法比较，没有运行旧控制均值探针或修改论文。

## 完成范围

数据与20epoch结果见[首轮报告](B0_EXPANDED_CANONICAL_20EPOCH_EVALUATION_20261009.md)。新数据128train/32val、一源一片段、13个AMASS数据集，160配对数值检查及1603项共同输入检查通过，14876个唯一train帧计算加速度统计；新154条caption只验证全文时间支持，没有人工语义认证。开发准入与原B0逐条审阅协议分开，正式test未使用。

[100epoch训练计划](expanded_canonical_training_20261009_v2/plan.json)、[完成状态](../../../outputs/itm_rework/method/expanded_canonical_training_authorized_20261009_v2/queue_status.json)：新U1600updates/25600exposures/25600MDMcalls/51200encodercalls/160captioncalls完成；两简单基线3200updates/51200examples/128个完整val输出完成。整队607.339691秒，peak693599744bytes，物理cuda:1。新U最终checkpoint SHA `be0d7fe7fb92e792ba36bdfe016e4a31f390583b987c5d4148cf5547a3518e11`。原共同initial、数据、optimizer、框架、真实长度累积保持；100epoch前20epoch的320条receipt/噪声描述/loss完全相同，冻结backbone参数不变。各模型只用最终checkpoint，无val择优。

[100epoch unit采样](expanded_canonical_sampling_20261009_v2/plan.json)38条（8val×2配置×paired/MDM text g2.5＋3train参考×2），38000MDMcalls/44encoder/11caption/38decode；267.999031秒，peak531856896bytes。同case/noise与20epoch一致，16个MDM输出normalized features/关节均bitwise相同。

[统一引导采样计划](expanded_canonical_sampling_matched_g2_5_20261009_v1/plan.json)、[完成状态](../../../outputs/itm_rework/method/expanded_canonical_sampling_matched_g2_5_authorized_20261009_v1/queue_status.json)：同100epoch权重，仅text_scale1→2.5，IMU/joint均1；新增22paired完整采样，22000MDMcalls/44encoder/11caption/22decode，164.553396秒，peak531856896bytes。原16个MDM g2.5与128简单基线输出直接复用，未重复GPU或CPU旧指标数学。8val/3train来源、22noise groups、seed20261010、1000DDPM/真实长度/K1保持。

本轮累计5个独立有界队列均completed，无GPU重试/恢复/预算失败；按长期自主授权与高利用率政策，首选cuda:1，UUID项目锁、free≥4096MiB、3600秒/24GiB allocator cap，不修改其他任务。累计队列墙钟约24分钟，不是GPU kernel活跃时间。

## 同8个验证来源、同噪声与text CFG2.5

每来源等权，配置分开；误差单位m。

| 方法 | head相对姿态 | wrists相对姿态 | head全局 | wrists全局 |
| --- | ---: | ---: | ---: | ---: |
| MDM text g2.5 | 0.161450 | 0.155373 | 0.671622 | 1.008493 |
| U100 text g1 | 0.154134 | 0.115560 | 0.690077 | 1.568088 |
| **U100 text g2.5＋IMU** | **0.150016** | **0.113100** | **0.520770** | **1.431350** |
| 简单文本＋IMU100epoch | 0.127223 | 0.112079 | 1.003387 | 1.031183 |
| 简单仅IMU100epoch | 0.116678 | 0.103687 | 0.817521 | 1.096190 |

相对同text CFG2.5的MDM：
- head：global−22.46%、root-relative−7.08%、root-path−25.54%、head proxy−22.80%、velocity proxy−1.85%，但jerk+51.36%。
- wrists：root-relative−27.21%、velocity−11.53%，但global+41.93%、root-path+43.81%、wrist position proxy+32.83%、jerk+102.05%。

即有实际IMU条件增量，但不能仅选姿态列宣称整体方法更优；根轨迹、动作自然性和文本语义仍需同时改善。head和wrists的noise groups不同，不把跨配置均值差解读为传感器数量的单变量因果。

20→100epoch在同噪声下：wrists相对姿态0.135555→0.115560，而global0.853106→1.568088；head相对姿态变化小、global0.754577→0.690077。增加训练不是所有指标的统一修复，既不继续盲加epoch，也不由两端loss断言唯一过拟合原因。

两个简单基线在全部32val的100epoch相对姿态：head text0.147411/IMU-only0.133088；wrists text0.134610/IMU-only0.119418。文本版本的velocity和jerk更低，姿态误差更高；保留代价，不能以单个正面量证明融合全面有效。8val主表与32val基线全表分别保留，不互相池化。

## 可查看结果与验证

最新前端 `/static/expanded-100ep-g2_5-review.html`，22组/148面板：[目录](../../../outputs/itm_rework/protocol/expanded_canonical_review_matched_g2_5_20261009_v1/catalog.json)。同屏有GT、U20、U100 g1、U100 g2.5、MDM/简单基线；train参考另有旧U/F。原用户反馈[交替弓步差、出拳转身/后举较好](USER_MOTION_REVIEW_20261009.json)保留为非盲观感，不改旧行为标签。原68面板、20epoch104面板、unit100epoch126面板均未覆盖。

[最终比较](../../../outputs/itm_rework/protocol/expanded_canonical_evaluation_matched_g2_5_20261009_v1/summary.json)：22新输出独立CPU解码/132项主要指标第二公式通过，复用144个已有输出的绑定指标（16MDM＋128基线），无缺失。100epoch1600更新/25600exposures/3个finite checkpoints核验通过，1951运行文件索引保存。最新22组148面板真实HTTP和原坐标一致，Node DOM/canvas2568帧/17180面板帧检查通过；不冒充浏览器截图或独立02/盲评。

CPU评价首次v1在NPZ上下文关闭后读取时报错，发生在指标前，旧失败入口/错误保存；CPU修复后的20epoch评价v2、100epochv3及matched评价独立目录完整。没有为了CPU入口错误重跑模型。

## 当前决策与下一阶段

保留当前架构的实际正面证据和所有负面结果，尚不冻结为论文最终方法。下一轮准备**一个解码后根轨迹/速度监督候选**，与U100保持同数据、共同初始化、曝光及噪声/验证人口，先验证有限梯度、真实长度、train-only统计和不把GT作为模型输入，再一次完整训练/采样/对照；不搜索大量系数，不回到单弓步保存状态的无限细分。目的是优先减少全局路径代价，仍要防止以收缩降低误差。该候选尚未实现/执行，不预先认定其有效；根据结果再确定静态参照差分等模块的完整训练消融。

官方基线核查见[接口报告](../BASELINE_INTERFACE_AUDIT_20261009.md)：IMUPoser原代码支持h/lw_rw，需canonical root rotation及公平姿态解码；Ego4o本机配置依赖尚未到位的作者预训练资产，不阻塞最低矩阵。官方IMUPoser/Ego4o、标准文本匹配/FID、正式test、多种子与核心结构消融仍未完成。新文本没有人工全审、严格generated-IMU闭环/真实设备校准未认证，body/root/velocity仍joint proxies。论文未修改，当前无活动实验队列，uvicorn本机服务保持。
