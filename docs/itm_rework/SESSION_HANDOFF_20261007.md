最新完整交付（2026-10-10）：头部主线本轮已关闭：U_head_relative_pv_v1一次100epoch训练/11条head完整采样，及原U100全部32val的四方法比较（另48新DDPM、复用80输出）完成；59新解码/1298指标核验、1600receipt/25600曝光/11528辅助启用检查通过。候选头部位置−5.21%但速度+1.78%、global+31.87%，不替换原U100，不扫系数。原U100在全部32head val对MDM局部位置−19.02%/速度−10.93%、全身相对姿态−12.65%，但jerk仍高、对简单基线存在代价，未认证整体优越。见[32val完整报告](protocol/HEAD_VALIDATION32_EVALUATION_20261010.md)、[候选报告](protocol/HEAD_PRIMARY_CANDIDATE_EVALUATION_20261010.md)、[当前状态](CURRENT_STATE.md)。下一完整交付改为官方IMUPoser head适配/CPU公平FK核验/一次有界训练与32val比较；标准语义评测资产准备和核心消融后续推进。双腕扩展不阻挡主线，论文/前端未改、无活动队列。一次沙箱预检无模型调用失败保留，沙箱外同UUID已验证可用；GPU长期自主/首选cuda:1有效。下文候选待整合或活动队列文字由本段覆盖。

最新用户方向（2026-10-10）：头部IMU作为论文与实验主线，任务为文本＋单个头部IMU→全身动作；双腕作为扩展配置，保留全部已知结果，不作为头部进度的前置条件。见[头部主线交付范围](HEAD_PRIMARY_DELIVERY_PLAN_20261010.md)与[新候选设计](protocol/head_primary_supervision_20261010_v1/plan.json)。下一任务是整合U_head_relative_pv_v1新worker：沿用U100全部25600曝光/共同初始化，只在head且IMU存在的11528曝光增加局部位置/速度辅助损失；一次完整训练后新增11条head采样，复用head旧U/MDM/简单基线。128train头部尺度/128真实长度梯度检查已验收，无需重复软件测试；worker尚未整合、GPU未启动，正式test/语义指标/官方IMUPoser与核心消融仍待做。论文LaTeX未改，前端按用户要求暂不处理，长期GPU自主/首选cuda:1有效。下文双配置候选和22条新采样待办由本段覆盖。

最新进展（2026-10-10）：根据用户“用哪个部位的IMU，就关注哪个部位的轨迹”，已完成传感器部位相对骨盆的位置/速度补充CPU评价：354个已有完整输出/594行/5664项检查，无新增GPU/模型调用。统一textCFG2.5的同8val，U相对MDM的head局部位置−10.40%/速度−9.50%，wrists−41.79%/−36.90%；wrists速度仍比简单文本＋IMU差9.88%，局部jerk仍高，不认证整体优越。已实现局部位置＋速度辅助损失，13项CPU tests（指标8/损失5）及128train/256配置真实长度梯度检查通过；候选尚未整合worker/训练/采样。唯一下一任务改为整合新worker、冻结有界计划并做一次U_sensor_relative_pv_v1完整训练/采样比较，暂缓根轨迹优先候选；原全局/根指标和负例保留。见[补充报告](protocol/SENSOR_RELATIVE_EVALUATION_20261010.md)与[当前状态](CURRENT_STATE.md)。前端用户反馈无法访问，按用户要求暂不处理，不承诺历史服务仍在线；论文未改、无活动实验队列，GPU长期自主/首选cuda:1有效。下文旧待办由本段覆盖。

最新完整交付：128train/32val的扩展开发实验已经关闭：U与两个简单基线20/100epoch、98条真实完整DDPM、统一text CFG2.5对照、CPU核验、22组148前端面板与归档全部完成，当前无活动实验队列。head全局误差相对MDM改善约22%，wrists相对姿态约27%，但wrists全局/根轨迹和jerk明显退化，未认证整体优越/动作修复或论文主结果。见[当前状态](CURRENT_STATE.md)与[最终报告](protocol/B0_EXPANDED_CANONICAL_MATCHED_EVALUATION_20261009.md)。唯一下一任务是准备一个解码后根轨迹/速度监督候选并做完整训练/采样比较；旧均值探针不执行，不盲加epoch或大规模扫系数，论文未改。GPU长期自主/首选cuda:1有效。下文旧待办与“当前活动队列”由本段覆盖。

最新用户方向与交付（2026-10-09）：用户要求先整体实现文本＋IMU任务、基线对比与消融，再抠细节；当前进行整体复盘，原控制均值探针保留为可选辅助设计，不再是所有后续工作的唯一前置任务。见[投稿推进方案](PAPER_DELIVERY_PLAN_20261009.md)与[当前状态](CURRENT_STATE.md)。已有六动作×head/wrists共12组/68面板已接入原uvicorn前端，缓存outputs/demo_runs/itm-rework-20261009-v1-*，入口/static/rework-review.html。12组真实HTTP及68面板全坐标核对通过、8原demo tests通过、Node逐帧绘制smoke通过；模型/GPU查询/采样新增0，论文未改。uvicorn127.0.0.1:8001已启动；0.0.0.0因网络暴露被自动审批拒绝后采用本机绑定。下文旧“唯一下一任务”由本段最新方向覆盖，旧科学结果与不可变实验文件保持。

最新进展（2026-10-09）：跨来源组件/误差CPU检查完整完成，计算implementation v1 `70ccba55…`/17911来源；480行/240F-U配对/七条件/120旧完整参考全部valid，9新增CPU tests通过。首个verifier的Python3.10索引语法在核验前失败，v1代码/失败/结果保持；兼容verification-only v2 `86bdfd46…`/17922来源复用原结果，602400标量含1008 null核验及8128汇总字段全部通过，原64eps尺度界不变，无重复旧数学或GPU/模型/噪声/解码。F paired训练来源24/24主项MSE改善，验证来源17/24变差、20/24平均姿态误差增加，能量不是统一下降，不能据单片收缩全局删主项。原F/knockout not_supported及论文不变。**唯一下一任务：实现/CPU验收[已冻结控制时间均值同状态探针](protocol/b0_control_mean_probe_20261009_v1/plan.json) `b3fe884e…`/17967来源/12case/48state/240checks，3冻结tests通过，然后自主一次新600秒GPU队列并完整CPU评价。** 固定F200与原F paired t900/500/250/50，C=原static-delta latent control，唯一改C−有效帧时间均值，保留变化分量/真实IMU/text/(1,1,1)；MDM48calls/192examples、E24/caption12，无noise/采样/解码/训练，尚未实现或执行。当前无活动队列，优先物理cuda:1 UUID `GPU-43c2f925-db05-c655-bb46-25196d0627d6`，长期自主/高利用率GPU授权继续有效，无需再手动批准。[当前状态](CURRENT_STATE.md)、[完整CPU报告](protocol/B0_CROSS_SOURCE_COMPONENT_EVALUATION_20261009.md)、[归档](method/cross_source_component_run_20261009_v1/artifact_index.json)覆盖下文旧待办/固定cuda:0文字。

# ITM 实验主会话迁移入口

整理日期：2026-10-07（Asia/Shanghai）。本文件为接管快照，不是运行授权。
准备时仓库 HEAD：`971d101bbacbbc914a276b474f6d878a93962826`，分支 `method/itm-rework`。
本迁移提交本身会使 HEAD 前移；接管时检查增量，不要求回到上述提交。

最新权限（2026-10-08）：用户明确要求“以后使用GPU不用我手动授权，你自行决定”，详见 [GPU 委托记录](GPU_EXECUTION_POLICY_20261008.json)。这一新指令覆盖本文件及旧任务书中的逐次手动授权要求；主会话自行维护具体 GPU/目录/预算记录，不再反复询问批准。科学冻结、每次队列预算与首故障停止、失败保留及不干预其他进程的边界不变。

最新结果：2026-10-08 F队列已完整完成，200/600/60/240/48全部completed，耗时2493.12秒。F/U评价已完成，主000506 head/wrists均不支持局部行为修复：数值门槛通过，行为仍不完整，其他10个paired配置中9个有数值回归。当前无活动队列。见[简短当前状态](CURRENT_STATE.md)与[完整评价/下一研究决策](protocol/B0_FEATURE_BALANCE_EVALUATION_20261008.md)。旧CPU准备和失败保留原位；以下为原迁移快照，不覆盖新权限和已完成结果。

最新进展补充（2026-10-08）：诊断v2已冻结；有界worker/supervisor、评价入口、implementation `82df1581…`完成，26增量CPU检查/2556依赖通过。首次GPU执行因compute-process预检停止，1.91964秒、worker未启动、模型调用0，全部计划分母与失败评价保留。下一任务是在GPU满足既有预检时自主创建新执行记录/目录并沿用验收实现运行，不需再次手动授权或重做无变化的软件验收。见[CURRENT_STATE.md](CURRENT_STATE.md)及[诊断入口/预检停止](method/B0_TRAINING_SUFFICIENCY_RUN_20261008.md)。下文历史任务不要求重做F、冻结设计或实现入口。

## 1. 当前目标和工作方式

用户希望新会话统一承担原 01 方法实现和 02 数据/协议/评价的日常工作，减少手动传话。使用 [启动提示词](NEW_SESSION_PROMPT.md)后，由主会话连续推进设计、实现、测试、获批后的运行和评价；03 论文仍单独负责。旧文档里的“下一步交 01/02”是当时分工，不要求恢复旧往返流程。

当前一个阶段的目标是**完成 F 损失配重候选的完整动作评价并作出下一步决策**。不是重新开始两周审计，不是直接扩大训练，也不是立即完成整篇论文。例行检查由主会话自检；主要研究结论在适当里程碑另行复核，同一会话自检不能标为新的“独立 02 验收”。没有默认多代理或跨会话消息授权。

科学目标仍是离线完整片段、head 或 wrists 的文本＋稀疏 IMU 全身动作生成。最终论文需要同一模型支持实例控制、相容文本编辑和优于简单替代方案。当前主要处于 G2 基线修复，尚未认证上述联合能力。

## 2. 最小阅读顺序

首次接管先读本文件和 [身份快照](session_handoff_20261007_v1/snapshot.json)，然后按任务读取：

| 顺序 | 文档 | 用途 |
| --- | --- | --- |
| 1 | [F 冻结协议](protocol/B0_FEATURE_BALANCE_FREEZE_20261003.md) | 唯一候选、固定对照、行为与指标门槛、预算 |
| 2 | [F 软件验收](protocol/B0_FEATURE_BALANCE_ACCEPTANCE_20261004.md) | 已通过事项、实际接受的是 outputs v2 |
| 3 | [最新待批准申请](method/B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005_V2.json) | 唯一目录/GPU/预算，authorized=false |
| 4 | [最新失败核对](protocol/B0_FEATURE_BALANCE_FAILURE_ACCEPTANCE_20261005.md) | GPU 占用停止、无 worker、旧授权已消耗 |
| 5 | [F 评价机器协议](protocol/b0_feature_balance_20261003_v1/evaluation.json)及同目录相关表 | 开发 F 评价脚本前必读；包括 target_review/review_order/within_contrasts |

需要实现细节再读 [F 实现报告](method/B0_FEATURE_BALANCE_IMPLEMENTATION_20261003.md)及代码。需要数据解释再读 [B0 数据交付](protocol/B0_DATA_HANDOFF_20260925.md)。两份大型 STATUS/HANDOFF、完整 evidence_index 和旧任务书均为按需索引，不是首次启动必读全文。

## 3. 关键身份、权限和当前状态

以下路径除特别说明外均相对仓库根 `/home/a200/0proj/ITM`。精确 SHA256 见快照，接管前以实际文件核验。

| 项目 | 当前值 |
| --- | --- |
| 方法实现 / 软件验收 | `963a132` / `e0999b7`，155 项 CPU 回归通过 |
| F package | `docs/itm_rework/protocol/b0_feature_balance_20261003_v1/package.json`；SHA `a4189b89…` |
| **实际接受的 manifest** | `outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json`；SHA `db323ef5…` |
| manifest 归档 | `docs/itm_rework/method/feature_balance_implementation_20261003_v1/implementation.json`；与上述 v2 字节相同，归档目录名 v1 不代表旧 outputs v1 |
| 原共同初始化 | `outputs/itm_rework/method/correction_implementation_20260927_v1/initial.pt`；SHA `3ad3c1bc…` |
| 历史 U 控制权重 | `outputs/itm_rework/method/correction_authorized_20260927_v2/R_static_reference_delta/model.pt`；R final200，SHA `91744ac5…` |
| 历史 U 完整动作 | `outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1/samples/`；60 条 unit-guidance S1，不能换成更早 guidance 2.5 输出 |
| 最新申请提交 | `971d101`，待批准，当前无真实 F 结果 |
| 拟新目录 | `outputs/itm_rework/method/feature_balance_authorized_20261005_v2`；迁移时不存在 |
| 拟实际授权记录 | `docs/itm_rework/method/B0_FEATURE_BALANCE_AUTHORIZATION_20261005_V2.json`；迁移时不存在 |
| 指定 GPU | `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`；当前是否空闲未知 |

两次 F 失败分别为 10/04 首项 nvidia-smi 退出码9（原因未确定）和 10/05 第二项查询发现指定 GPU 计算进程。两次 worker 均未启动、模型调用均为0，账本及所有计划槽位保留，授权均已消耗。后一次不能解释前一次的具体原因。最新申请目录名含 authorized 只是路径，不能据此执行。

接管不等于批准预算。需要用户后续明确批准这一次身份/目录/GPU/限额；没有自动轮询、重试、换卡或干预已有任务的授权。若用户改变限额或预检政策，应先形成具体方案/实现及相应验证，不能暗改本轮冻结入口。

## 4. 已确立的数据与模型边界

- 数据包：`docs/itm_rework/protocol/canonical_b0_20260925_v3/package.json`，schema `canonical_smpl_paired_v1`。同源重建的限定合成面板，不能冒充旧 `standard_imu_v1` 或真实设备认证。
- train 三源组：000027（选定 caption index1）、000506、004728，20 Hz 目标长度104/136/158，共398唯一帧；val：000433/000382/000139，长度41/94/163。head 槽位4，wrists 槽位0/1。
- IMU 为世界系运动学线加速度，不加重力；30 Hz合成→20 Hz输入，加速度插值/姿态SLERP。归一化仅来自上述398个train帧，不重复计传感器配置。目标 motion 使用已绑定的 MDM Mean/Std。
- 只有限定离线合成映射得到准入；生成动作→安装顶点加速度/段姿态闭环仍未认证，评价是关节代理，不称严格 IMU 一致性。
- 6源组是开发面板，不能改名当独立测试集；预训练暴露未排除。原诊断缓存继续隔离。相容编辑 witness、正式扩大验证和真实硬件实验均未完成。
- `contract.yaml` 顶部 status/read_commit 反映当时数据交付阶段，不是当前运行状态。公共契约仍 v0.3-draft；本轮状态以新申请、运行账本和验收记录为准，不能为了消除旧字段而改动已绑定契约。

## 5. F 冻结内容速查（完整协议优先）

唯一干预：有效帧上的 `L_F=0.5*mean(e[0:67]^2)+0.5*mean(e[67:263]^2)`。root4＋RIC63与其余196维各半；完整263维正权重保留，loss_version=`normalized-x0-root-ric-half-v1`。

从共同 initial 开始，fresh AdamW，原600 exposures/200 updates计划、真实长度、每更新三样本各除3、static-delta、dropout和参数顺序不变。不是从R final200续训；只有F final200可采样，50/100/150只保留不选择。600条记录的uniform/D/A/balanced来自同次预测。

新60完整采样=val42＋train18；单位 factorized batched CFG `(1,1,1)`，DDPM1000步，12组历史完整噪声身份不变。240 trace来自原四分支调用，额外调用0。48完整motion软件等式界1e−5m；**F/U IMU-only不要求相等**。无新Q/V探针。

主负例是train000506，head/wrists分别判断：GT起点已是左侧前跨屈膝，须评价初始恢复、右侧前跨下沉、中段回收和末端下沉，不要求片外起始迈步或末尾站直。F paired行为完整而U不完整，且相对U paired和自身text/static的五项指标均不退化、root-relative严格改善，才支持该配置的局部修复。浮点松弛为 `1e−6+1e−6*abs(reference)`，不能放宽或替代行为审阅。其余10个source/config逐项检查退化。

评价人口为F/U120行固定隐藏标签顺序；60跨臂、72臂内对照和36差值之差，全部负例、失败/缺失分母保留。旧U已看过，单AI审阅不能宣称完全盲评或独立人工验证。contact仅val000433/000382适用，其余为NA；无空间/时间拟合。不同臂trace的x_t不同，不是同状态因果对照。

拟限额：训练200 updates/600 exposures，t0在预检前，训练≤600秒；60采样共享≤3000秒且总≤3600秒，单GPU/allocated≤24GiB。MDM≤60600调用/240600 examples，encoder≤1320 examples，caption≤660。首次规定故障即停；有限原始特征、已完成输出、失败与未执行槽位均保存，无重试/补采/恢复。

## 6. 接管后马上可推进的工作

1. **核对增量和身份。** `git status --short`、`git log -5 --oneline`，读取快照；若仓库更新，按实际证据修正摘要。未跟踪 `docs/itm_rework/tasks/` 为用户原有文件，保留，不自动加入提交或清理。其他旧会话停止写入后再开始修改；发现并行写入先协调。
2. **补齐 F 评价准备。** 迁移时 `scripts/`、`tests/` 没有专用 F 效果评价入口，只有F实现、冻结及软件验收代码。先搜索确认没有新交付，再按冻结 evaluation.json 复用旧指标实现，建立F输入解析、配对、完整性/失败分母、判据与审阅输出。历史U及合成fixture可CPU测试，不伪造真实F数据或语义判定。旧评价脚本有硬编码历史路径，不能只替换run名后盲跑。
3. **依赖只读历史产物的工作先完成。** 不因待GPU授权就停止CPU评价准备；也不借此无边界增加诊断。新增评价脚本不修改已绑定方法/协议文件；若运行依赖变更则按实际影响验证并更新manifest，旧验收不能自动覆盖变化。
4. **真实运行后闭环。** 获新授权后使用已验收入口执行一次；成功则完整F/U评价，失败则保留账本并精简核对。产出可重建指标/表、全量审阅记录、局部支持/不支持/证据不足和下一决策。不能仅交付“运行成功”。

CPU Python：`/home/a200/miniconda3/envs/itm/bin/python`。涉及已绑定 PyTorch 环境时使用 `OMP_NUM_THREADS=2 PYTHONPATH=src`。本文件不附可误触发的GPU运行命令；获批后从已验收CLI及新批准记录构造。

| 代码/证据 | 用途 |
| --- | --- |
| `src/itm/models/mdm_feature_balance.py` | 损失、来源和计数 |
| `scripts/run_mdm_feature_balance_bounded.py`、`run_mdm_feature_balance_worker.py` | F父入口/worker；无新授权不能execute |
| `scripts/accept_b0_feature_balance.py` | 已完成的CPU输入/manifest验收；不是效果评价 |
| `scripts/evaluate_b0_sampling_alignment.py`、`evaluate_b0_correction.py` | 复用指标、配对和绘图逻辑，先检查硬编码 |
| `scripts/summarize_b0_sampling_alignment.py` | 旧汇总参考，不能直接标成F报告 |
| `tests/test_mdm_feature_balance.py`、`test_b0_feature_balance_protocol.py` | 方法49项/协议9项，既有联合共155项已通过 |
| `docs/itm_rework/protocol/evidence_index.jsonl` | 历史证据账本，按evidence_id检索，追加事实不改旧结论 |

## 7. 历史研究结论，按需追溯

- [首次B0评价](protocol/B0_EVALUATION_20260926.md)：200updates/36完整输出；有步行变高抬腿及接近静止条件的偏置。
- [D/L/R评价](protocol/B0_CORRECTION_EVALUATION_20260927.md)：480探针、两臂各200updates、120输出。真实长度和静止参照差分有局部改善，不是全面有效或纯padding因果证明。
- [单位guidance评价](protocol/B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md)：固定R、480探针/60输出/240trace；有收益也有退化，额外文本引导不是统一根因。
- [V保存状态评价](protocol/B0_STATE_DIAGNOSTIC_EVALUATION_20261003.md)：48状态/192分支/144锚点；训练来源局部预测改善未转成完整动作，部分root/RIC被其他块改善掩盖；val顺序/拍手问题仍在。因此提出F，不代表F已有效。

无需重复旧探针、噪声重建或全量回归来“接管”。新失败或实质变更才扩大相关检查。F若无效，应报告候选不足并重新判断训练充分性/表示/机制，不自动搜索权重或追加预算。论文不能把多个checkpoint各自优点拼成同一模型能力。

## 8. 后续摘要和关键复核

每完成一个研究阶段，更新一个简短当前状态：目标、HEAD/关键身份、已确认结论、主要负例、预算是否消耗、下一任务与必要路径。历史大报告保留原位，用链接追溯，避免再次在两个STATUS反复堆叠相同内容。

如需另开短复核会话，可使用：

> 请只读复核 ITM 的指定提交/产物（主会话填入准确路径和SHA），按该次冻结协议从原始结果重算关键表，检查分母、负例、来源和主张。不要只接受主会话摘要；不修改方法、阈值或旧结果，不运行模型/GPU，不代替用户批准预算。输出发现、证据和适用范围；同模型复核也不称统计独立或人工验证。

迁移本身未发消息给旧会话、未停止进程、未启动代理或创建新聊天。用户在同一项目/文件系统中打开新会话，粘贴启动提示词即可。让旧01/02保留为历史查阅，避免同时修改同一工作树。
