# 01 → 02：单一 root＋RIC 损失配重修正提案

接受 `85859c9` 的独立评价：V 诊断交付通过，**B0 有效性仍未认证**。本轮提出一个候选 **F：将 normalized x0 MSE 的 root＋RIC 与剩余特征各赋予一半权重**。只改变训练目标的固定通道权重；从原共同初始化训练 200 updates，以既有 R final200 的 60 条 S1 完整输出作历史对照，新增 F 的同 60 条完整采样。

本轮完成代码阅读、既有数组重算、训练计划与历史对照身份核验，以及可重建的提案包。**未实现新训练入口、未加载 checkpoint 张量、未运行模型、训练、采样或 GPU 查询。** 状态为提案，尚未冻结、软件验收或授权。下一责任方是 02 冻结设计与评价，之后 01 实现并完成 CPU 验证，再交 02 增量验收和申请新授权。

## 要检验的一个问题

主问题选择 **“训练来源局部有利响应未形成完整动作”**，特别是 train `000506` 交替弓步。当前训练对全部 263 个标准化特征等权平均；可见关节的 RIC 解码直接使用 root 4 维和 RIC 63 维，而 rotation6d/local_velocity/contact 共 196 维不直接进入这个解码函数。代码依据为 `run_mdm_correction_worker.py::train` 的 `(prediction-x0).square().mean()`，以及绑定 `motion_process.py::recover_from_ric` / `recover_root_rot_pos`。

这不意味着后 196 维无用：它们参与模型输入及后续状态演化，也承载旋转、速度和接触约束。不能据此删除这些通道、改输出、只 mask 下肢或声称 root＋RIC 的标准化 MSE 就是几何误差。拟检验的窄假说是：**固定训练预算下，等通道配重允许其他块的收益掩盖位置表示的退化；增加位置表示在目标中的份额可能改善完整训练动作。** 当前没有梯度冲突、根因或修复成功证据。

只读重算全部 192 个 V 分支和 192 项比较，结果见 [saved_contrast_loss_views.json](b0_feature_balance_proposal_20261003_v1/package/saved_contrast_loss_views.json)。以下 t50 差值均为同状态前项减后项的标准化特征 MSE，不是米；负值有利：

| 来源 / 配置 / 比较 | 原 263 维均值 | root＋RIC 67 维 | 其余 196 维 | 拟配重下的既有误差 |
| --- | ---: | ---: | ---: | ---: |
| train 000506 / head / P−S | −0.048299 | +0.012019 | −0.068918 | −0.028449 |
| train 000506 / head / P−R | −0.002529 | +0.016298 | −0.008964 | +0.003667 |
| val 000382 / wrists / P−S | +0.185063 | +0.273264 | +0.154912 | +0.214088 |
| val 000382 / wrists / P−R | +0.002564 | +0.004887 | +0.001770 | +0.003329 |

全部 train 的 48 项主要比较（3 来源×2 配置×4t×P−S/P−R）中，5 项整体下降但 root＋RIC 上升。表末列只是对**不变的旧预测**重新汇总，绝不是新模型预测或通过标准；不会按是否翻转符号挑选权重。即使位置组各占一半，000506 P−S 仍为负，目标再配重也不能保证姿态不再被补偿。

验证集 head 全来源劣于倒序，以及 wrists 拍手 P/R 均劣于静止，是必须保留的外部检查。**本候选不直接修正顺序学习，也不预言拍手必然改善。** 若只有 train 改善，报告训练拟合证据，不能认证验证集动态 IMU 利用。

## 单一干预的精确定义

设每条真实长度片段的标准化 x0 误差为 `e[c,t]`，有效帧集合为 M，半开通道区间 D=`[0,67)`、A=`[67,263)`：

```text
原 R: L_U = sum_{t∈M,c∈[0,263)} e[c,t]^2 / (263 |M|)
候选 F: L_F = 0.5 sum_{t∈M,c∈D} e[c,t]^2 / (67 |M|)
              + 0.5 sum_{t∈M,c∈A} e[c,t]^2 / (196 |M|)
```

若写成原 masked mean 的逐通道权重，D 为 **263/134**，A 为 **263/392**，所有权重严格为正且总和 263。原位置组占 67/263≈25.48%，候选占 50%。1/2 来自两个功能分组等权的预设，不从来源/t/条件/GT 指标拟合；不试其他系数。归一化权重和避免统一放大标量 loss，但不保证梯度范数或 AdamW 更新量相等。

- 数据、完整 x0 目标、398 帧 train-only 归一化、采样器、模型结构、static-delta 和 strict null 均保持原定义。所有关节用相同规则，head/wrists 使用同一模型，不按来源设权重。
- 所有存在/缺席组合统一应用 L_F，保持原 dropout 条目和 effective-batch 分母。每次更新三条真实长度 microbatch，各 loss/3 累加；不按帧长改变来源权重。
- 不新增姿态解码 loss、teacher、时序对比 loss、rollout 监督、gain、guidance 搜索或额外前向。所有 263 维仍被监督，不在推理时 mask/替换通道；模型全部输出照常进入 sampler。
- 固定 200 updates 后的唯一 checkpoint 做完整评价；中间 50/100/150 快照只按旧运行方式保留，不采样或选模。若未满 200 updates 则停止，不用中间快照顶替 final。

这是新的候选实验，不改写旧模型、冻结协议或旧评价结果。
新 checkpoint 和逐样本元数据必须绑定新包/实现、`loss_version`、完整263维权重、原初始化和训练计划hash、final200状态。旧 R 的只读对照身份单列，不能借用旧 checkpoint schema 将候选标成原 R；训练恢复或采样入口对来源/损失身份不符应拒绝。

## 匹配训练与历史对照

候选 F 从原 **initial.pt** 开始，文件 SHA256 `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401`。**不继续 R final200 训练**。沿用 R 的 static-delta 配置、bias unused/frozen、共享 encoder 参照梯度和匹配 dropout，原 frozen_eval backbone 和原生 CLIP 精度路径。

原 [training_schedule.json](../protocol/b0_correction_20260926_v1/training_schedule.json) 的 600 条记录逐项复用：样本顺序、caption、timestep、完整 noise 地址/hash、text/IMU presence、encoder dropout seed。optimizer 为同设置的全新 AdamW，lr=5e−5、weight_decay=0.01，其余参数继承绑定 common；不是加载 final200 optimizer。仍为 200 updates／600 exposures，各 source/config 100 次；不改变训练覆盖或时刻分布。

历史对照 U 是已经按相同初始化/计划完成的 **R final200**，SHA256 `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`，完整动作采用 S1 的 unit guidance 输出。已把原 200 个 update JSON 中全部 600 条实际 exposure、噪声 hash 与冻结计划核对一致。初始化仅做文件 hash，未载入权重张量。

本提案选择复用现成对照，避免为不变训练再增加 200 updates。**这是历史配对对照，不是新同期双臂或独立训练重复。** 01 实现准备时必须核对全部初始 tensor 身份、环境、训练计划、参数集合和采样路径；CPU 验证统一权重退化为原损失。若无法保持这些身份或发现其他计算变化，停止交 02 修订，不自动加跑新控制臂，也不把差异全归因于权重。

## 完整动作验证：60 新输出配对 60 既有 S1 输出

[sampling_pairs.json](b0_feature_balance_proposal_20261003_v1/package/sampling_pairs.json)已经逐条绑定历史 NPZ、sidecar、random_plan_id、caption、presence、长度和 guidance；不是泛指未来再选例子。

| split | 来源 | 条件 | 新条数 |
| --- | --- | --- | ---: |
| train | 000027、000506、004728，各 head/wrists | paired、text_only、physical_static | 18 |
| development val | 000139、000382、000433，各 head/wrists | 原六条件：paired、text_only、physical_static、numeric_zero、imu_only、backbone_bypass；另 tail_reversed | 42 |

全部 K=1、DDPM 1000 steps、factorized batched CFG `(1,1,1)`，完整重放 S1 同组初始噪声与每步噪声，原真实长度。保留原 240 个 t900/500/250/50 trace，来自实际已执行前向，无额外调用；它们仅用于后续解释，不能替代完整动作评价。没有新 Q/V 探针队列，不按训练中间 loss 决定是否采样某个 case。

02 冻结前明确评价与阈值，建议如下：

1. **主负例：train 000506 的 head 与 wrists 分开判断。** 与同噪声历史 U paired、同臂 text/static 比较，观察是否形成交替前弓步及起身/回位；保留全帧脚高/左右前后位移、pelvis 高度/轨迹曲线和完整固定序列审阅。低坐/低蹲变浅但没有交替与回位，仍是行为失败。不能仅凭 global MPJPE 或 weighted loss 下降成功。
2. 沿用 global、root-relative、root、joint proxy、velocity、jerk、motion extent 和逐来源差值；报告 F−U 以及 `(F paired−F static)−(U paired−U static)`，验证集同时保留相对倒序的差值之差。位置与速度/幅度一起报告，不用运动收缩冒充改善。阈值、uncertain 类和缺失分母由 02 在结果前固定。
3. **回归负例：**000382 wrists 拍手的下肢退化与双手动作、000139 步行对倒序的排序、000433 踢腿后回位、004728 方向/保持/幅度、000027 出拳/转身，及 numeric-zero/IMU-only 全部保留。训练 `004728` wrists 的有限粗正例也保留，不能为修一个动作牺牲其他来源而只报均值。
4. 检查 F static/text-only/backbone 的原软件等式及 F 与 U 的 null 输出，沿用既有数值容差。若 F 只靠整体压小控制恢复 text/static、paired 无目标一致的动态收益，则不支持 IMU 修正有效；若原始263误差改善而完整动作仍失败，本轮否定“这次固定配重足够修复”，不继续搜权重。
5. 新旧各 60 行纳入固定审阅，模型/条件标签先隐藏，已知旧结果和单 AI 审阅局限须如实保留。3 train＋3 已暴露 development-val、单训练 seed、K=1 不支持总体成功率、显著性、泛化、严格 IMU 闭环或相容编辑认证。

“支持候选”至少需要主负例完整行为改善且配对指标没有超出预先冻结界的姿态/速度退化，并排除仅恢复 text-only 的解释；其他来源分别报告。任何局部 train 改善都不自动升级为 B0 认证。若仅一个配置好转，明确限制为单配置结果。

## 拟预算与实施门槛

[budget.json](b0_feature_balance_proposal_20261003_v1/package/budget.json)为**未授权预算提案**：

| 阶段 | 工作上限 | 时间上限 |
| --- | --- | --- |
| 训练 F | 200 updates；600 MDM 前向/examples，1200 encoder examples | 10 分钟，含预检、加载、快照、核验 |
| 完整采样 | 60×1000 steps；60000 次 batch4 MDM／240000 examples，encoder 最多120 examples；240 trace额外调用0 | 60 条共享 50 分钟 |
| 整队 | 单 GPU；allocated≤24 GiB | **预检前起最多1小时** |

caption 调用上限训练600＋采样60=660；全队 MDM examples 240600、encoder examples1320。此前 R 训练阶段约33.80秒，S1 整队60采样含Q共34.06分钟仅为实耗参考，不是本次速度承诺。训练/采样阶段期限均受绝对总期限约束，不转移或重置超时额度；首个 timeout/OOM/nonfinite/身份/软件锚点/计数/写入失败即停，保留完整200更新/60样本/240trace分母与已完成产物，不重试、warmup、换卡或 CPU fallback。有限但动作错误不触发筛除或补采。

流程为 **02 冻结 → 01 实现与 CPU 验证 → 02 差异验收 → 01 绑定新 manifest/目录/GPU 申请授权 → 执行 → 02 独立评价**。原 V 软件验收不覆盖尚未实现的新训练损失；历史所有运行授权均不可复用。本轮没有向用户申请立即运行。

CPU 验证应覆盖：两组通道半开索引/权重和、有效帧及样本分母、统一权重退化、梯度参考与未训练参数不变、600条计划与初始化绑定、只改变损失公式、final200唯一入口、完整noise重放、调用上限和首错保留；不以镜像实现的断言替代独立数值参考。冻结包由02负责，不直接编辑公共协议。

## 本轮证据与复现

[提案 design](b0_feature_balance_proposal_20261003_v1/package/design.json)、[CPU 核验摘要](b0_feature_balance_proposal_20261003_v1/package/validation.json)、[573 份来源身份](b0_feature_balance_proposal_20261003_v1/package/sources.json)、[192 分支的两组误差](b0_feature_balance_proposal_20261003_v1/package/saved_branch_loss_views.json)和60条映射已交付。使用标准库和 NumPy；不导入 torch，不构造模型、加载 checkpoint tensor 或查询 GPU。包可重建到一个不存在的新目录：

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python \
  docs/itm_rework/method/b0_feature_balance_proposal_20261003_v1/build_proposal.py \
  --output /tmp/itm01_feature_balance_proposal_rebuild
```

本轮已在独立临时目录重建，全部7份提案JSON逐字节一致，见 [重建核验](b0_feature_balance_proposal_20261003_v1/rebuild_verification.json)。这是提案数据复现，不是尚未实现的候选训练软件验收。

现有 V/Q 与动作输出保持原样；重配重视图仅作设计证据，不改变 02 的原指标或结论。
