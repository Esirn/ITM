# 02 → 01 / 03 交接

**最新加载器修复差异验收（2026-09-27）：**[CORRECTION_LOADER_ACCEPTANCE_20260927.md](CORRECTION_LOADER_ACCEPTANCE_20260927.md) 接受 `03e6d95`，无新增返工项。4项CPU针对性测试通过：旧函数在内存中复现同一NoneType/eval异常，修复后的L/R加载接口通过；01的204项联合证据已核对，未重复运行。四份失败runtime副本与原目录相同，480/120未运行分母保留，模型前向/更新/采样计数均为0。

**下一责任方为01：绑定新manifest `a802ef3b…` 申请另一次原上限队列授权。** 126份绑定文件仅worker改变，新旧初始权重、60份输入metadata及环境一致；127份文件核对通过，原冻结包/输入/限额验收沿用。本次旧授权已用于约4.05秒后停止的失败队列，不可重用。02本轮无真实模型加载/前向、训练、采样、GPU查询或重试；收到新授权后才由01预检并单次执行。

**最新 D/L/R 实现独立软件验收（2026-09-26）：**[B0_CORRECTION_ACCEPTANCE_20260926.md](B0_CORRECTION_ACCEPTANCE_20260926.md) 通过 `8e4df1e`，无新增返工项。最终代码联合回归 **203 passed，0 failed/error/skipped**（含25项修正实验测试）；另行独立核对全部120槽位、398帧train-only统计，两臂60份输入逐值一致。新准备在禁止模块前向/GPU调用下完成，初始权重逐张量及文件整体相同。

**下一责任方为01：对已实现整队申请新授权，之后预检并执行一次 D/L/R。** 绑定冻结 package `e24c8cdd…`、01最终v3 implementation `449b8d27…`、唯一新目录和GPU；预算为480探针＋两臂各200 updates/60样本，单GPU/24GiB/合计最多2小时。此次仅软件验收，不是授权；未运行真实训练/采样或GPU预检，不复用旧B0授权。相同实现无需再次冻结/往返确认，真实输出交02按已冻结协议评价。

**最新 B0 修正设计冻结（2026-09-26）：**读取 `cd1079e`，已交付 [B0_CORRECTION_FREEZE_20260926.md](B0_CORRECTION_FREEZE_20260926.md) 和 `b0_correction_20260926_v1/package.json`：D=480 探针，两臂共用 600-exposure/200-update 训练随机计划，每臂 60 条完整采样（val42/train18），共 120；单 GPU/24 GiB/总 2 小时上限保持不变。全部比较、数值容差、输出/失败分母与审阅顺序预先冻结，数据准入不扩充。

**下一责任方为 01：按冻结包实现并做 CPU 软件验证；之后由 02 独立验收，再申请新的整队预算授权。** 当前只完成协议构造和 6 项计划测试，11 个包文件独立目录重建逐字节一致；没有模型前向、训练、采样或 GPU 查询。冻结包不是授权，不能复用旧 B0 授权/运行器，也不把 R 静止恢复当作动态 IMU 收益。

**最新 B0 结果独立评价（2026-09-26）：**[B0_EVALUATION_20260926.md](B0_EVALUATION_20260926.md) 已完成对 `4ae11d0` 的全部输出评价：36/36 身份/数值有效、398 帧训练统计及实际输入通过、6,000 个唯一噪声 draw hash 重放、6/6 strict-null/backbone 输出逐值相等；新增评价解析测试 6 passed。本轮仅 CPU 评价，无新训练/采样/GPU。

**运行交付通过，联合控制能力未获认证。** paired 的 global MPJPE 虽下降，root-relative MPJPE 却上升；000139 从步行变成持续抬腿，paired 与 physical-static 输出很接近。全 36 条条件隐藏审阅、逐条数值、失败分母和负例均已保存，未调阈值、未挑输出。

**下一责任方为 01：**基于本报告诊断 B0 条件分支的动作偏置及真实时变输入增量，交付有针对性的修正/诊断方案及预算；无需继续防护或重复接口验收。02 本批评价已完成，待具体方案后冻结后续评价。原一次授权已用完，不能自行追加训练/采样或复用剩余时长；下一计算试验需对应授权。

以下为各批次交付时的历史状态，旧“待授权/尚未生成”不覆盖上方本次已授权执行及评价结果。

**最新运行限额独立验收（2026-09-26）：**[B0_RUNTIME_ACCEPTANCE_20260926.md](B0_RUNTIME_ACCEPTANCE_20260926.md) 通过 `1d44adc` 的 CPU 软件验收，无新增返工项。166 项联合回归通过（含 23 项限额故障测试），补充三种顶层队列场景及实际微型 CPU 1000-step 循环通过。真实 GPU 显存上限仍未实测；本轮无 GPU 预检、真实训练或完整动作生成。

下一步只待原预算授权与执行前 GPU 预检，由 01 在原 200-update／36-sample／最多 4 GPU 小时范围执行一次 B0；02 随后独立评价全部输出与失败。旧输入/统计验收继续有效，无需再为相同接口往返确认。授权须另存记录绑定原预算/计划、单个运行目录及 GPU UUID，不改写已绑定数据包。

**最新方法接入独立验收：**[CANONICAL_METHOD_ACCEPTANCE_20260925.md](CANONICAL_METHOD_ACCEPTANCE_20260925.md) 通过 `1d7a32d` 的限定合成 B0 输入接入：398 帧 train-only 归一化逐值相等，6 个训练配置与 36 个验证槽位通过独立参考，143 项联合回归通过，两个旧输入报告逐字段不变。无输入接入返工项，无需重复确认。

已准备一条训练与 36 条单 case 采样命令模板，未启动。当前剩余工作是 01 落实时间/显存/更新次数硬限制及失败保留，02 核对具体运行方案；用户预算授权和 GPU 预检仍待完成。不要把输入验收当成限额验收或计算授权。公共契约及原数据包保持不变；未训练或扩散采样。

**最新实质数据交付（2026-09-25）：**见 [B0_DATA_HANDOFF_20260925.md](B0_DATA_HANDOFF_20260925.md)。固定 12 条重建候选，12/12 时间/坐标/根与 RIC 检查通过；首批限定离线合成 B0 准入 **3 train + 3 val**，5 uncertain、1 左右冲突全部保留。26 项相关测试通过。公共契约提升 v0.3-draft，旧 v0.2 原文归档。

新数据清单、实际缓存、逐条变换、GT 审核和 200-update / 36-full-sample / 最多 4 GPU 小时提案已交付。尚未训练或生成；预算等待用户授权，01 需要接入新 `canonical_smpl_paired_v1` 并绑定新 train-only IMU 归一化。数据准入仅限预先统一坐标的合成基准，不是任意真实 IMU 部署标定。**B0 不再等待相容 witness；witness 属于后续 B 分支。**

[ACCELERATION_NUMERICS_20260925.md](ACCELERATION_NUMERICS_20260925.md) 保留原四项 acceleration allclose 失败并分解精度因素；新诊断缓存可逐值重放，旧缓存仍有缺少历史后端/mesh 记录的未决重放差。新 B0 数据单独同源构建，没有把旧失败改记通过。

以下为各历史批次在交付时的记录；其中“全部未准入/等待物理对齐”等状态针对当时旧资产，不覆盖上方新限定数据包。

**最新缓存/恢复独立验收：**见 `CACHE_MIGRATION_ACCEPTANCE_20260925.md`，`b4f70fc` 的五文件双 manifest 拒绝和 caption 重放均复现，126 项联合回归通过。
旧 checkpoint 的未绑定恢复已拒绝，仍不代表真实采样已重跑或数据已获训练许可；无需再次确认本轮已验收接口，后续推进实质数据/对齐依赖。

**最新 caption 方法验收：**`CAPTION_METHOD_ACCEPTANCE_20260925.md` 接纳 `1a7e887` 的输入实现，联合回归 109 项通过；18 输入成功、2 阻塞与 13 SSM 拒绝全部重放一致。
01 无需再重复确认这些已验收入口；后续需实质数据/对齐证据。真实非零裁片仍未认证，所有训练准入阻塞保持，不能升级为新模型效果。

**最新前缀实测：**`PREFIX_ALIGNMENT_20260925.md` 提供十条预先固定来源的三时钟比较和五份独立 CPU 重建。前缀补偿在 10/10 公共帧比较中降低残差；5/5 重建支持完整特征时间，但 acceleration 重叠 allclose 有 4 份失败。
新缓存只在 `prefix_rebuild_20260925_v1/diagnostic_manifest.jsonl` 声明，全部未准入训练；旧 cache、默认映射和公共契约不变。不要把 caption 入口接纳或本诊断升级为全数据/物理对齐认证。

**最新时钟证据（2026-09-25）：**见 `HUMANML_CLOCK_20260925.md`。13 条 SSM 的整数降采样时钟解释获帧数和固定轨迹诊断支持，但不得直接作为部署变换；2177 条涉及 notebook 数据集前缀的记录仅为待查范围。
运行时源映射和旧缓存保持不变；caption 接口接纳不等于这些时钟差异已解决。03 可引用诊断证据，不能写成物理同步/生成收益。

**最新 caption 接口：**详见 `CAPTION_ALIGNMENT_20260924.md`。严格解析、半开区间和双时间轴计划已提供；`caption_plan_20260924_v1/cases.jsonl` 显式记录 caption 索引、源秒轴与模型秒轴。
01 接入时特别保留 val `000133` 的索引 1，不能对 `000004` 的全段 caption/witness 静默截成 196 帧。`000006` 可规划为 `[0,100)` 帧的有标签片段，但仍未认证语义/世界系。13 条 SSM 来源短 IMU 输入继续拒绝；02 已补新缓存构建的源区间越界检查，历史时钟成因待查。
7,443 条审计和最终 31 项相关测试已完成；全部候选保持 training_admissible=false，不是训练/生成收益。

**最新输入追验：**`INPUT_FOLLOWUP_ACCEPTANCE_20260924.md` 接纳 `0574491` 的 seconds-v2 输入更新；83 项联合回归通过。
已登记方法交接的 train ID 描述差异，并按明确 ID 补查。公共契约保持 v0.2-draft；世界系/骨架/caption、compatible witness 和合格 B0 仍未认证。
01 已在 `0b7ed28` 正确勘误，02 已核对其仅改三份方法文档；样本记录差异闭环，无需再次回执。后续交接以实质依赖的新证据为准。
以下是前批记录，不将旧 checkpoint 或旧诊断追认为新管线训练/研究结果。

**第二批更新：**当前交接见 `METHOD_ACCEPTANCE_20260924.md`，读取方法提交 `5519365`，公共契约为 `itm-rework-v0.2-draft`。
已独立验收 strict null/随机流/同源解码、停止全段拉伸，补充静止 fixture、9 条时序诊断及 3 条 uncertain 张臂候选。
请勿按下文第一批历史条目重复修复已完成的 crop_seconds；世界坐标/骨架、真正 compatible witness 和合格 B0 checkpoint 仍待完成。
以下保留为第一批交接记录。

日期 2026-09-24；读取 commit `663414f1d9bffd6a70189846739bc9d9b1d0d873`；改动未提交。
契约 `itm-rework-v0.1-draft`，未冻结；代码快照 hash 见 `code_snapshot.json`。
本批只新增数据/协议/评价代码与本目录文件，未修改方法代码、论文或任务书。

## 给 01：可直接用的接口与阻塞

1. 读取 `../contract.yaml`。继续使用现有 world linear acceleration + global orientation 分布，不把缓存无声改为 raw body specific force。
2. `itm.data.observation.load_standard_cache` 提供显式 schema/FPS/单位检查。旧 `_PairedDataset` 尚未接入；由 01 选择集成点，不强制改变模型函数签名。
3. `validation_20260924_v1/panel.jsonl` 提供 6 个 val-only 来源组的候选 ID、文字、原始区间与路径；同步验收前不是正式 frozen panel。
4. 修复 `sample_mdm_imu_control.py::_load_case` 的 `_fit_length_*` 全段时间拉伸；应在明确秒时间轴上匹配、裁切，并记录有效区间。不要给 shuffled 片段任意 time warp。训练入口 `min(length)` 也不能代替同步证明。
5. 核验 HumanML canonicalization 与原始 IMU 朝向；不得用目标 GT pelvis 做部署预处理。缓存 GT joints 与 HumanML263 存在一帧及坐标定义差异。
6. 输出 B0 的 null/mask/CFG/dropout/采样正确性证据、checkpoint/hash、模型卡、完整配置。现有 Stage-2b 仅是历史资产，未命名为修复 B0。
7. `itm.data.random_plan.sampling_noise` 给出与调用顺序无关的逐步噪声；在 sampler 中显式消费并记初始/后续随机流，不能只设置全局 seed。
8. 新采样逐样本字段见 `result.schema.json`；失败也写记录。训练 seed 与 sample seed 分开。所有最终联合能力使用同一 checkpoint。
9. `observe_mesh` 必须读取最终渲染同一 mesh/rotation；它不解决 HumanML→SMPL 旋转丢失。输出 IMU 认证通道当前为空。方法 B 严格观测约束和方法 C 的物理梯度分支仍受此阻塞。
10. 新 GPU 作业在执行前登记 GPU、显存、样本/K、sampler/steps/forward 次数、seed、guidance 和停止条件。当前建议 P=84 输出/模型，仅建议，未启动。

腕部 orientation 18/19 与 proxy joint 20/21 不同已有安装权重依据；不建议直接把 orientation 段换成 wrist 段。
若要改变输出表示，请先给最小接口变化、迁移成本和验证计划；02 尚未启动大型人体解码器项目。

## 给 03：可写与不可写

- 可以写本轮核查了 8776 条缓存，以及旧控制输出的可复算性；不能把缓存合法性称为生成动作的真实 IMU 一致性。
- 可以写在历史开发诊断上，Stage-2b 双腕 paired-vs-shuffled 的 joint-proxy 均值改善跨五个 sampling seeds 保留；必须同时披露来源重叠、时间拉伸和未覆盖训练方差。
- 单头实例控制证据仍弱；30-motion source-group CI 跨零。不能与双腕作相同强度主张。
- 可以写旧文本编辑中摆臂/转向属性发生变化，快慢/大步不稳定；不能说这些编辑已确认物理相容，不能报告本轮 C 成功率。
- 100-motion 的 74/100 是样本比例，不证明总体超过 70%。文本编辑 head 差值 CI 改为每 motion 平均后 bootstrap，见新统计。
- 557 对跨 split 时间区间重叠必须披露；本轮尚无封存的独立测试池。
- 旧 5-case round-trip 加速度/姿态不通过；保留负结果，不能通过更大阈值包装为成功。
- 当前没有“同一最终模型同时解决语义编辑与传感器控制”的新证据；不得拼接 Stage-1/2b 的不同能力。

唯一事实索引为 `evidence_index.jsonl`。raw 表在 `recomputed_20260924_v1/`，
来源组统计在 `statistics_20260924_v3/`。历史评估脚本的 Markdown 标题或旧文字“正式”不改变新证据状态。
失败过的两次身份检查记录保留在 statistics v1/v2；v3 才是完整核验结果。

## 真实硬件后续导入准备

设备未知时不假定信号类型。每次采集记录 participant/session、设备/固件、sensor_config、原始时间戳与时钟、
量纲、重力状态、姿态方向、坐标轴、安装段/固定旋转、静止标定区间、同步方式、丢包掩码和重新佩戴记录。
机体系比力用已确认 R_wb 与 g_w 转为 a_w；机体系线加速度只旋转，不补重力。未知 quantity 拒绝导入。
保留原始设备数据与转换配置 hash；推理预处理不能读取完整人体 GT。
参与者/会话隔离；校准片段不包含目标完整动作真值；零样本与微调分开。
无独立全身 GT 不报告 MPJPE；没有通过同一输出链的观测算子也不报告真实传感器成功率。
Nymeria 3-clip 仍是 pilot。本批没有上传人体数据、假定设备身份、确定采集人数或开展新采集。
