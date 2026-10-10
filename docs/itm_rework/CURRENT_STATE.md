# ITM 实验主会话当前状态

最新阶段（2026-10-10）：头部主线的一次局部监督候选训练/评价，以及原U100全部32val的四方法比较已完整关闭。实验交付时HEAD30d1f96、method/itm-rework；本轮按[Git边界](VERSION_CONTROL.md)分阶段整理源码/协议/报告，原始tasks原样纳入版本控制。主会话承担01＋02，无代理/跨会话消息，论文LaTeX与前端未改，当前无活动GPU队列。

**主任务**
文本＋单个头部IMU→完整全身动作。双腕作为扩展，保留全部已知结果，不作为主线推进前置条件。头部局部位置/速度为主要控制代理，还需全身动作、文本语义和质量；根/global作辅助报告，不把GT轨迹作为输入。

**本轮实际完成**
- 新head监督worker保留原U100初始化/optimizer/seed/100epoch25600曝光、head/wrists各12800及真实长度累积；仅head且IMU存在的11528曝光增加0.1局部位置＋0.1速度辅助损失。新增5训练＋2采样CPU tests通过，旧尺度/梯度验收复用。
- 训练1600updates/25600MDM/51200encoder/160caption完成，401.902秒，骨干冻结校验不变；checkpoint77e97455…；11条head完整DDPM完成（8val＋3train），83.737秒；全部receipt/噪声/辅助启用、11解码/242标量核验通过，无缺失。
- 候选对原U头部位置−5.21%，速度+1.78%（仅2/8改善）、全身global+31.87%、根轨迹+39.62%；局部jerk−1.70%。**不替换原U100，不将辅助损失认定为最终模块，不扫该系数。** 候选8val表单独保留。
- 原U100/MDM全部32head val补齐：复用8来源的16生成与两个简单基线64输出，只新增24来源×两方法48完整DDPM，317.884秒；48解码/1056标量检查/32配对噪声组通过。四方法各32来源，共128个完整输出，无缺失、无输出挑样。
- 三成功队列合计约803.52秒，59新完整DDPM/1298新指标检查。首次沙箱预检nvidia-smi退出9且没有设备节点，0模型调用；外部只读查询同UUID成功后新目录运行同计划，原失败保留，无恢复/拼接或科学参数修改。

**全部32val头部结论**（单seed/K1，textCFG2.5，开发集）
- 原U头部局部位置0.132428 vs MDM0.163536m（−19.02%，21/32改善），速度0.368437 vs0.413664m/s（−10.93%，20/32）；全身相对姿态0.152754 vs0.174872（−12.65%），global0.787783 vs0.914183（−13.83%）。局部jerk76.683 vs44.250（+73.29%）、全身jerk+47.58%，不认证整体优越。
- 对简单text＋IMU：头部位置+1.63%/速度−1.76%、global−20.07%、局部jerk−35.41%；全身相对姿态+3.62%、全身速度+15.35%、全身jerk+3.24%。保留全部代价，位置接近不是显著等价。
- 对简单IMU-only：头部位置+34.09%、全身相对姿态+14.78%；头部速度−7.37%、global−18.93%、局部jerk−45.54%。文本语义和融合价值仍需独立评价。
- 无新行为成功/严格generated-IMU闭环、FID/R-precision或显著性认证；正式test未用。头部结果支持主线推进，阶段工作模型仍为原U100 be0d7fe7…，不是论文最终模型冻结。

**数据与历史保留**
128train＋32val、13个AMASS数据集，160配对/320CPU FK；160/160数值与1603共同输入检查通过，14876唯一train帧用于加速度统计。新154条文本沿用原标注，未人工语义认证。原20/100epoch共同训练、98完整DDPM、传感器补充354已有输出/594行/5664checks保持不变。原U/F弓步not_supported、双腕全局/jerk代价及用户对旧出拳转身/后举的非盲正面观感保留。

**交付入口**
- [32val完整头部主表与决策](protocol/HEAD_VALIDATION32_EVALUATION_20261010.md)、[候选完整评价](protocol/HEAD_PRIMARY_CANDIDATE_EVALUATION_20261010.md)、[官方head基线接口补充](HEAD_IMUPOSER_INTERFACE_ADDENDUM_20261010.md)、[语义评测器资产准备](HEAD_SEMANTIC_EVALUATOR_READINESS_20261010.md)。
- 32val结果：outputs/itm_rework/protocol/head_validation32_evaluation_20261010_v1/；候选结果：outputs/itm_rework/protocol/head_primary_evaluation_20261010_v1/；静态候选对比PNG/PDF：outputs/itm_rework/protocol/head_primary_review_20261010_v1/，已目视核对。
- [原扩展完整报告](protocol/B0_EXPANDED_CANONICAL_MATCHED_EVALUATION_20261009.md)、[局部补充评价](protocol/SENSOR_RELATIVE_EVALUATION_20261010.md)、[头部范围决策](HEAD_PRIMARY_DELIVERY_PLAN_20261010.md)。上述较早文档的待办由当前状态覆盖，冻结科学结果不重写。
- 原前端缓存68/104/126/148面板与历史HTTP/Node检查保留。用户反馈无法访问，按要求暂不处理，不承诺历史uvicorn仍在线。

**下一完整交付**
优先实现官方IMUPoser的head12维/加速度÷30/24关节6D旋转适配：先核验canonical rootQ转换、官方源码入口12维与数据加载60维不一致的修正、真实20Hz/来源隔离及公平SMPL FK，再一次有界训练并进入共同32val头部表。不能把现有自建IMU-only称为官方IMUPoser，不能给没有平移输出的方法填GT根轨迹。

标准HumanML语义评测器的finest.tar与GloVe资产在已检查目录缺失，按官方来源独立下载/绑定后准备共同语义评价；随后同一工作模型的条件贡献与核心结构训练消融。多种子、来源隔离正式test、文本编辑/语义与严格传感器闭环仍待做。不要回到损失系数、epoch或单弓步的无边界诊断，旧均值探针继续延期。

**权限与环境**
长期GPU自主、首选物理cuda:1 UUID GPU-43c2f925-db05-c655-bb46-25196d0627d6，3次free≥4096MiB预检，利用率仅记录、UUID项目锁/native allocator；新队列冻结3600秒整队期限/24GiB allocated cap，首故障停止，不干预其他进程。当前沙箱看不到GPU设备，已验证沙箱外可用，可依据授权申请工具执行访问，无需向用户重复要GPU批准。Python /home/a200/miniconda3/envs/itm/bin/python，OMP_NUM_THREADS=2 PYTHONPATH=src:scripts。
