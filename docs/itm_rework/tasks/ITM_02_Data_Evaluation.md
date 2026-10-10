# ITM 会话 02｜数据契约、评价协议与实验协调任务书

**整理日期：2026-09-24。角色：数据/观测/评价负责人兼协议协调者。**

把本文件交给第二个独立的服务器 Codex 会话。你的核心职责不是多跑几张表，而是让不同方法接受同一物理定义、相容性标准与成功判据，并将可复核证据交给论文会话。模型实现由 `ITM_01_Method_Implementation.md` 管理，写作由 `ITM_03_Paper_Rewrite.md` 管理。

## 给 Codex 的启动指令

请先盘点现有数据 manifest、缓存、checkpoint、逐样本输出和已用测试样本，建立最小公共契约与证据索引。优先复用旧资产，不默认重做整个数据集或 full-set×20。然后实现数据/观测审计、三类评价协议和分阶段运行器；先让方法会话在固定验证面板上得到可信反馈，再对最终冻结候选进行评价。没有独立真值或未通过观测闭环的量必须明确标为 proxy，不能用结果看起来好来放宽定义。

## 0. 三个会话共同遵守的协作契约

### 0.1 范围、目标与证据状态

本轮只研究**完整片段输入的离线全身动作生成**，分别使用 `head`（槽位 4）或 `wrists`（槽位 0、1）两种配置。不默认增加 head+wrists 联合配置，不扩展任意佩戴组合，不做在线跟踪、实时系统或未来动作预测。

主场景是**局部表演驱动的虚拟角色动作创作**：用户录制局部运动，并用文本指定最终目标动作或编辑意图；输出不必等于录制时真实发生的完整动作，但应保留输入中实际可观测的惯性特征。文本和 IMU 不预设为严格独立的“语义/物理”变量。

三条主证据是：① IMU 带来文本未确定的实例信息；② 在已确认相容的组合上，文本实现目标编辑且不破坏观测；③ 所提机制在同一 backbone 和明确预算下优于简单替代实现。必须由**同一个最终模型**支撑联合能力，不能拼接不同 checkpoint 的优点。

本任务书依据既有评审和用户反馈，参考提交为 `663414f1d9bffd6a70189846739bc9d9b1d0d873`，**不声称它是服务器当前 HEAD，也不声称新方案已经有效**。开始时读取当前工作树、提交和有关文件，核对历史问题是否仍存在；已修复的问题只补证据和回归测试，不重复重写。下文区分：历史稿件报告、代码可推导性质、待实测风险、待验证研究方案。

### 0.2 文件所有权与交接

| 会话 | 主责 | 可修改范围；路径不存在时按现有工程风格创建 |
|---|---|---|
| 01 方法实现 | 编码器、条件关闭、CFG、训练和采样、新方法 | `src/itm/models/` 中相关代码；训练/采样脚本；方法测试与配置；`docs/itm_rework/method/` |
| 02 数据与实验 | 数据契约、观测算子、划分、评价、实验调度和证据汇总 | `src/itm/data/`、相关 metrics/evaluation；数据/评估脚本和测试；`docs/itm_rework/protocol/`；公共契约与证据表 |
| 03 论文 | 场景、任务、论证结构、LaTeX、结论账本与编译 | `paper/latex-ITM/`；必要的同源草稿同步；`docs/itm_rework/paper/` |

上述是责任分界，不是要求移动现有目录。实际文件交叉时，先记录具体 owner；非 owner 提交补丁建议或接口请求，不直接同时编辑同一文件。每个会话只写自己的 `STATUS.md` 和交接说明。**02 是协议与证据协调者，不代替 01 修改模型，也不代替 03 改论文。**

02 维护唯一公共契约 `docs/itm_rework/contract.yaml`、协议版本与 schema；维护只记录事实的 `docs/itm_rework/protocol/evidence_index.jsonl`。01 提供 checkpoint、模型卡、配置及逐样本生成产物；02 计算与校验评价；03 读取证据索引，维护自己的 claim 账本。不能让三个会话各自维护一份含义不同的 IMU 定义或“主结果”。

建议各用独立 git worktree/分支：`method/itm-rework`、`protocol/itm-rework`、`paper/itm-rework`。公共契约冻结后以明确 commit 交接，后续变更提升版本并记录影响，禁止无声改字段。使用共享机器时由 02 汇总运行队列；01 的调试任务也登记显存和设备，默认同一 GPU 仅启动一个重训练/采样任务。

### 0.3 非破坏性与实验诚实性

先检查 `git status`，保留用户未提交修改。不得强制 reset、删除原始数据、覆盖旧 checkpoint/结果、清理他人任务、擅自 push/合并主分支或扩大硬件采购/数据上传范围。新增产物用独立 run ID 和版本目录；已完成旧实验不因换名字成为新证据。

服务器数据或算力暂缺时，完成可执行代码、测试、dry-run 与清晰阻塞记录；不得伪造成功日志或数字。只运行已经明确的预算，不擅自无限扫描。完整采样的负结果要保留，不以单步 loss、检索率或精选动画替代生成能力。

采样种子与训练种子分开记录。测试集不用于选模块、选 epoch、选 guidance、定阈值或判定继续训练。历史用于选择的 test-20/可视化面板标为开发诊断；不能通过重命名、重采样或只换 seed 恢复独立性。

### 0.4 公共交付与阶段门槛

每份交付都说明：读取的 commit、改动 commit、契约版本、改动文件、已执行命令与结果、未执行项、影响哪些旧结论、下一会话可直接使用的路径。运行产物至少关联数据/协议版本、配置和 checkpoint hash、采样器、所有 guidance 权重、训练 seed、采样随机流与评价器版本。

阶段顺序：**G0 核查当前工程与旧证据 → G1 公共契约和正确性测试 → G2 修复基线 B0 → G3 小验证集方法筛选 → G4 最终候选冻结评价 → G5 更新论文证据。** 03 可以从 G0 开始重写场景与结构，但不能提前写成已验证结果。B 分支依赖相容编辑协议；C 分支还依赖通过验证的输出观测闭环。某依赖未满足时记录阻塞，不私自降低“传感器一致性”的定义。


## 1. 本会话的判断框架

分别回答三个问题，不把它们合并：

1. **存在性：**是否有自然动作同时满足给定文本与 IMU？
2. **增量性：**文本是否对 IMU 尚未确定的内容提供额外约束，或 IMU 是否选择了文本未指定的表演实例？
3. **模型能力：**当前方法是否真正实现上述联合满足，而非忽略一个条件或只对输入敏感？

相容不等于有信息增益；模型失败不等于不存在可行解；当前模型失败也不能一律解释成 bug。用观测见证、单模态能力、简单基线和代码审计区分来源。

## 2. P0：工程资产和历史证据盘点

### 2.1 首轮读取与交付

读取当前数据加载与合成代码、evaluation scripts、历史审计文档、现有实验输出和 `paper/latex-ITM/sec-cn/4_experiments.tex`。记录当前 HEAD 与参考提交差异；以实际入口为准。

建立 `docs/itm_rework/protocol/inventory.md`，每个资产记录路径、存在性、hash 或版本、生成命令、使用过的 split、是否被用于开发选择、可否直接复评。把证据分成：

- `reported_only`：稿件写了，但原始产物尚未定位。
- `artifact_verified`：配置、checkpoint、数据和输出可追溯。
- `recomputed`：已经用指定评价器从输出重新计算。
- `invalidated_or_changed`：确认存在协议/实现变化，需要补跑或降级。

历史稿件已有约 7009/434/1333 对齐记录、672 条×5 的同子集生成质量比较、100-motion 控制统计、30-motion×5 采样种子、文本属性实验和 3 条 Nymeria pilot。这些是**盘点线索**，不是新协议必须复用的样本数，更不是已在本轮复现的结果。

### 2.2 最低成本复用策略

区分“重新计算指标”和“重新生成动作”：只有评价逻辑改变而输出足够时，优先离线复评；checkpoint/sampler/null 修复改变生成机制时，受影响比较必须重新生成并使用新模型 ID。缺乏随机计划或关键配置的历史结果可以作为探索性记录，不强行纳入正式统计。

不要一概废弃旧结果，也不将不可复核的表格升级为新证据。先完成一张**旧表—原始资产—已知风险—需要重跑范围**映射表。

## 3. P1：公共数据与接口契约

### 3.1 唯一契约文件

你维护 `docs/itm_rework/contract.yaml`。以下是要求覆盖的逻辑字段，不要求机械照搬命名，也不假装现有工程已经具有这些接口：

```yaml
contract_version: itm-rework-v1
scope:
  mode: offline_full_clip
  sensor_configs:
    head: [4]
    wrists: [0, 1]
imu:
  schema_version: <explicit_version>
  quantity: <world_linear_acceleration_and_global_orientation_or_other>
  acceleration_unit: <m_per_s2_or_declared_legacy_unit>
  rotation_convention: <source_to_target_frame_and_matrix_layout>
  gravity_convention: <included_removed_or_not_applicable>
  world_frame: <axis_handedness_and_initial_alignment>
  sensor_mounts: <versioned_vertices_body_segments_and_transforms>
  timestamps: <seconds_and_clock_reference>
  normalization: <train_only_statistics_hash>
motion:
  representation: <humanml263_and_decoder_version>
  world_alignment: <explicit>
  observation_operator: <version_and_certified_channels>
protocol:
  split_manifest: <path_and_hash>
  development_exposure: <path_and_hash>
  compatibility_manifest: <path_and_hash>
  threshold_spec: <path_and_hash>
  sampling_spec: <path_and_hash>
```

在与 01 对齐前，不先强制改其函数签名。01 可以请求契约变化；你更新版本、迁移规则及影响说明，03 读取同一版本。

### 3.2 IMU 物理约定与旧数据隔离

参考实现 `src/itm/data/standard_imu.py` 的标准合成为：安装顶点世界位置二阶差分乘 FPS²，经平滑后作为 acceleration；orientation 取身体段全局旋转。它更接近**世界系运动学线加速度 + 全局姿态**。

先核对服务器真实 cache 是否由此生成，再决定是否保持该定义。不要自动改成原始机体系比力，导致旧 checkpoint 输入分布完全改变。

旧的方向单位向量与 `position/frame²` 使用独立 schema。加载时检查维度、单位、坐标、FPS、旋转含义与版本，混用报错。一个 `12D` 形状并不能证明物理定义相同。

必须记录：槽位和身体部位；安装顶点/旋转段；矩阵左右乘、local-to-world 方向；安装变换；重力处理；坐标轴和初始朝向；平滑窗口的秒数与帧数；差分边界；缓存 FPS 与模型 FPS；镜像；裁剪；normalization 数据来源。

参考代码中腕部 orientation 关联的身体段编号和 evaluation 对应的 wrist joint 编号不同，可能是合理安装选择，也可能需要对齐；必须检查实际 SMPL 语义和顶点附着，不凭编号不等就断定 bug。

### 3.3 真实设备到训练约定的桥接

在明确设备输出后实现转换。例如，若设备给出机体系比力 `f_b` 和 body-to-world 姿态 `R_wb`，理想情况下可按已确认符号约定得到 `a_w = R_wb f_b + g_w`。若设备已经输出去重力线加速度，不能再次补重力。安装旋转、磁航向/姿态漂移与时间同步应独立记录。

这是条件转换，不是用 GT 修正输出。head 或 wrists 推理预处理不能暗中读取真值 pelvis 朝向/位置、完整人体或目标 caption 对应动作来做只有 GT 才能完成的 canonicalization。GT 可用于训练监督和独立评价，但不进入被宣称可部署的输入路径。

### 3.4 必做单元与集成测试

| 测试 | 预期与注意事项 |
|---|---|
| 静止、匀速、恒加速度 | 按实际 quantity 验证；静止世界线加速度应为零、姿态为合法常量，原始比力的预期不同；边界帧单独处理。 |
| 刚体单轴旋转 | 安装点有偏心时检查切向/向心项，不能只检查关节中心；旋转正交、det≈1、单位一致。 |
| 重采样与差分 | 同一物理时间轴；平滑、差分、下采样次序明确；高频与 aliasing 风险可测，不用强拉长度掩盖错位。 |
| 镜像、全局旋转和平移 | 将真实运动作变换后重合成，与声明的信号变换一致；反射不是旋转，不能直接把 det=-1 矩阵当姿态。 |
| 时间裁剪与同步 | caption 区间、动作和 IMU 对应；仅掐短到 min(length) 不能证明同步正确。 |
| mask 与无效值 | inactive 传感器数据变化不影响模型输入；NaN/丢包不能以零值和静止混为一谈；差分要求整个 stencil 有效。 |
| 归一化 | 统计量只来自训练数据，逐传感器含义与左右交换测试一致；train/val/test 不各自重新拟合。 |

与 01 共同审计完整输入管线，给出通过、失败、尚未测试三种状态，不只写“已检查”。

## 4. P2：输出观测闭环——它决定论文能声称什么

### 4.1 统一观察接口

建议逻辑接口：

```text
observe(renderable_motion, timestamps, calibration, sensor_mask)
    -> acceleration, orientation, valid_mask, operator_metadata
```

该算子从**实际用于渲染/评价的同一个运动结果**计算观测。独立预测 IMU 的网络头、直接复制条件 IMU、每次用目标信号重新拟合安装角，都不能作为输出一致性证据。

### 4.2 先测真值 round-trip，再测生成结果

执行：AMASS/SMPL 真值 → 模型动作表示 → 实际解码/重建运动 → 相同传感器定义的虚拟 IMU。分别报告位置、加速度、姿态 geodesic residual 的均值、中位数、尾部分位数与失败率。

稿件已经有 22-joint→SMPL 拟合时姿态/加速度不可靠的诊断。不能假设增加拟合迭代即可恢复已丢失的轴向 twist；也不能假设 263D 包含旋转通道就意味着原始 AMASS 旋转被无损保存。检查表示生成来源、局部旋转约定、骨架/身体形状、安装点和全局对齐。

### 4.3 三条明确分流

- **已验证通道：**进入 measurement-space residual、方法 B 的严格观测约束，以及 C 分支。
- **只具备可信 proxy：**命名为 joint-proxy，并将联合成功率写成 proxy 版本；可以继续小论文方法诊断，但不能声称真实加速度/姿态均满足。
- **需要改输出表示：**给 01 提出最小有理由的变更和迁移成本；未获明确预算前不启动新的大型人体解码器项目。

不要用过大阈值“让 round-trip 通过”。测量容差应有应用与误差基线依据，且仍能区分合理输出和明显错误输出。即便真值 round-trip 通过，也要测扰动/生成分布下的数值稳定性，不能把它当万能正确性证明。

若 C 使用自动微分，增加 JVP/VJP 与有限差分一致性测试；确认梯度经过与真实输出同一链条，而不是只对方便的 proxy 反传。

### 4.4 可观测性与对齐边界

输入没有直接给出完整位置轨迹时，不得在编辑协议中把所有 root/全身 GT 轨迹都作为硬性观测。理想加速度与姿态可能对常量平移、某些初始速度等不敏感；允许的 nuisance alignment 必须事先定义，并对所有模型相同。

同样，不允许用任意逐帧旋转/时间拉伸把错误生成“对齐到正确”。已声明固定时间的控制任务必须评价相位与时序；若研究时间重参数化，作为另外协议，不与原始时序跟随混排。

## 5. P3：数据划分、开发暴露与前瞻性评估

### 5.1 分组溯源

按原始 AMASS sequence、时间区间、镜像来源和多 caption 关系审计 train/val/test。相邻片段或同一动作的不同文字不应被当完全独立样本来扩大统计量。

分组审计不意味着必须重建所有 cache。先列出真实重叠、可疑重叠及官方原始划分继承情况；重新组合 manifest 往往比重新合成信号成本更低。

若采用新 split，检查 backbone 预训练曾使用的数据。把旧训练样本改名放到“新 test”不会制造对预训练模型独立的测试。保留官方合法测试池中的未开发分组，或使用明确独立的数据来源，并披露预训练可见范围；不能假称绝对无泄露。

### 5.2 开发暴露登记

历史 `test-20` 若用于停止 Direct-V3/选择模块，即已经参与开发；100-motion/文本编辑面板是否多次用于选择也需核查。对每组记录 `exposure_role`、决策用途、日期和 run。

优先从合法测试池中封存尚未用于开发决策的来源组。若没有足够未暴露数据，明确旧结果是探索性，并建立后续独立采集或其他前瞻性评估；不能通过不看标签、换 seed、改名称恢复独立性。

### 5.3 样本不足与 1333→682→672 的说明

解释每个过滤阶段的原因、丢弃 ID、动作类别/长度分布变化。保留现有同子集直接比较的价值；不是所有实验都要使用最大集合。

“为凑 batch size 去掉末尾样本”要可追溯。如果修改尾 batch 实现，所有同表方法按相同新集合复评；不能一半用 672、一半用 682。小 batch 下的检索候选数也可能改变 R-precision 含义，要检查评价器而非只改 `drop_last`。

## 6. P4：相容性标注与有动作见证的编辑集

### 6.1 定义与标签

\[
\mathcal F_m(s;\tau)=\{x\in\mathcal X_{natural}:d(\mathcal O_m(x),s)\le\tau\}.
\]

存在 `x ∈ F_m(s;tau) ∩ X(c)` 才称在声明容差下相容。标签用 `compatible / incompatible / uncertain`；额外记录文本是否提供新的属性约束，不能把冗余文本当互补成功案例。

不要让待评估模型决定标签，不按输出质量事后挑相容样本。LLM 可生成/解析文本，不可仅凭词语或 IMU 数组宣布物理相容。有限动作库找不到 witness 不证明不相容。

### 6.2 三种可执行构造方式

**A. 运动见证构造，优先。** 从真值或经过质量检查的运动出发，在保持有关运动链/已认证观测的前提下改变目标属性，重新 FK、合成传感器，再验证观测误差与自然性。固定 head 时从可控的上肢修改起步，但仍实际验证，不能把“不同部位”当独立性证明。双腕时选择经观测检查确实可改变的其他属性，不先验保证腿部任意可动。

**B. 真实动作库近观测匹配。** 找出在相同配置、时长与已声明对齐规则下观测相近，但语义属性不同的动作。后一个动作是近似 witness；阈值由验证确定。不能为了配对任意 DTW 或加旋转，从而消除需要控制的差异。

**C. 人工复核的相容/冲突诊断。** 采用明确的运动学边界、自然性检查和标注规则；无法确认存在性时标 uncertain。对冲突组合说明冲突的是哪项观测与属性，不能仅按动作名分类。

每种构造记录覆盖率、排除原因、来源组和质量检查。手工构造的 witness 不能冒充真实硬件观测；泛化结论与真实数据分开。

### 6.3 不预先把属性分成固定名单

固定 head 观测下的摆臂是合理起点但需逐段检查。转向可能影响头部全局朝向，也可能有颈部补偿；快慢可能涉及不可观测初始速度，也可能改变步频/加速度；大步同样视具体观测。不能写成“摆臂一定可行、转向一定冲突”或相反。

### 6.4 相容编辑 manifest

至少包含：

```text
case_id, source_group_id, source_motion_id, sensor_config,
base_caption_id/text, edit_caption_id/text, edited_attribute,
compatibility_label, evidence_type, witness_motion_id/path,
witness_observation_residuals, quality_checks, tolerances,
text_information_role, timestamps, calibration_id,
split_role, construction_version, reviewer_or_rule_version
```

训练/验证/测试的来源动作组隔离；文本模板拆分和同义改写可作为泛化诊断，不能以同义改写掩盖同一动作泄露。teacher 置信筛选只用于 01 的训练规则；测试 manifest 在观察待评估输出前冻结。

## 7. P5：三个主协议与必要对照

### P：原始配对与实例控制

文本和 IMU 来自同一原始动作；比较 Text-only、真正 IMU-only、Text+IMU、同类近长度 shuffled、完全 bypass、数值零和物理静止诊断。区分“开启控制通道的平均偏移”和“利用具体实例读数”。

固定宽泛文本，换同类的不同实例 IMU，检查观测 residual、相位和目标属性方向，不只检查两个生成结果距离。加入时间移位/局部打乱/仅统计量控制作为选做诊断；对移位产生的边界断点单独处理，不能把 OOD 损坏当唯一证据。

配对恢复可以报告 MPJPE，不要求先关闭生成随机性。明确对齐、每条件采样数、平均误差与 best-of-K 分开；各方法不能用不同样本预算竞赛。

### C：相容组合编辑，核心新协议

固定一段观测，比较基础文本与编辑文本；反向比较固定文本下的实例控制。每个 case 同时判断目标语义、传感器观测与自然度，所有组件来自同一输出、同一 checkpoint。

输入换成 `c_A + s_B` 时评价的条件是这个新组合；不能仍用原始 A 或 B 的全身姿态作为唯一正确解，惩罚合法自由度。可以使用 witness 验证条件存在，但不要求模型复制唯一 witness。

主比较为同 backbone 的 B0、简单融合、绝对 anchor、分支 B、普通观测梯度、分支 C 及必要组合。外部方法需输入与协议可比；对无法复现的算法说明缺口，不捏造数字。Flexible IMUPoser 作为重建边界保留，不用其结构差异替代主要方法消融。

### D：冲突与不确定边界

单列 incompatible 与 uncertain；不把冲突编辑计入相容成功率分母，不静默丢弃。描述不同 guidance 下各条件违约和动作质量，不要求系统同时满足不可能条件，也不把偶然折中称“解决冲突”。

若用户未来需要传感器优先或文本优先政策，那是额外任务规范；本轮不自动新增冲突分类器或在线询问系统。

## 8. P6：主指标、条件内多样性与防作弊

### 8.1 联合成功率与完整分解

\[
JSR=\frac1N\sum_i\mathbf1[S_i\ge\tau_c\land E_i\le\tau_s\land Q_i\ge\tau_q].
\]

语义 success 至少包含预期方向和足够幅度，不把任意非零变化当成功。观测在已认证通道分别报告 acceleration、orientation 与同步误差；混合单位归一化规则明确。质量阈值只排除明确无效输出，同时辅以连续质量指标和盲评，不能声称一个门槛穷尽自然性。

只有 joint proxy 时名称用 `JSR_joint_proxy`，不是 `JSR_IMU`。同时报告各组成成功率、误差分布、阈值敏感性和语义—观测 Pareto 曲线。不能只报告“通过样本中的准确率”。

所有阈值在验证阶段根据 round-trip、设备重复误差和应用容差确定，测试时冻结；同一表中各方法相同。阈值既不能小于表示误差到无法解释，也不能大到无意义。

生成崩溃、NaN、超预算等预先定义为失败或明确计入 failure_rate；不能从分母删除。输入本身在模型运行前就无效的传感器样本按统一规则单列，并报告排除比例，不按某模型表现选择。

### 8.2 语义评价必须有独立性

配对协议用同一文本—动作 evaluator；编辑协议增加属性级运动学和盲评。若方法训练用某个 Phi，评价不能只用同一个 Phi 宣布成功，至少提供独立判据、人工盲评或未参与训练的属性验证。

Text-only 是否理解编辑作为诊断报告，不是测试集删样理由。语义保持也不等于 retrieval 分数接近就证明每个编辑属性保持。

### 8.3 多样性：先有效率，再有效集合内差异

固定 `(c,s,m)` 多次采样，报告联合有效样本比例，再报告有效样本对之间的条件内多样性和可解释属性覆盖。少于两个有效样本记 `not_estimable`，不填零或挑好看的数字。

同时公开无条件筛选前的输出差异和失败分布，避免“只剩两个优质样本”的多样性掩盖低成功率。传感器约束越强，合理多样性可能降低；不使用“一律越大越好”的结论。

未安装传感器的关节未必完全自由；部位差异叫部位差异，不叫严格零空间多样性，除非另有验证。

### 8.4 自然度与生成质量

jerk ratio 的小值不能自动作为优点；运动幅度收缩、静止都可能降低 jerk。报告绝对 jerk/其分布、足滑、运动幅度及质量盲评；GT jerk 接近零时按冻结规则处理，不用任意 epsilon 制造漂亮比值。

FID、R-precision、Matching 在同一原始配对子集、同一 evaluator、caption 和长度协议下比较。刻意修改语义的组合编辑不机械对比原始 GT 分布并将偏离判为失败；可以单列参考质量，但不能替代联合条件评价。

## 9. P7：统计、随机计划与结果账本

### 9.1 层级统计

同一 motion 的多个 seed/caption 不视为独立动作。先按 case/motion 汇总，再按预先定义的源组 bootstrap；按来源组重采样时保留各模型成对关系。若报告不同层级 bootstrap，说明估计对象不同。

保留采样种子稳定性分解，最终少数训练方案尽量有独立训练重复。训练重复不足时明确不确定性范围只覆盖采样/动作，不覆盖训练方差。小样本类别分层是探索性，不凭若干显著结果作全面属性结论。

方向测试、均值差、JSR 不是同一假设。历史 `74/100, CI 65%—82%` 不能证明总体超过 70%；非显著也不证明等价或固定偏置。需要等价结论时预设等价界限和相应设计。

### 9.2 可比随机计划

同一 backbone 条件比较共享长度、sampler、initial noise 和后续随机噪声计划；记录 eval 状态、dropout、dtype、采样步数。仅设置全局 seed，而不同分支消耗 RNG 次数不同，不保证控制变量成立。

不同 backbone 的噪声维度/空间不同，不声称逐元素共享同一 noise；共享可比的条件、物理时长和重复预算即可。

### 9.3 逐样本结果字段

至少保存：

```text
run_id, case_id, source_group_id, split_role, protocol_version,
data_schema_hash, compatibility_manifest_hash, condition_ids,
model_id, code_commit, config_hash, checkpoint_hash, train_seed,
sensor_config, caption_id, sampler, steps, guidance_mode,
text_scale, imu_scale, joint_scale, sample_seed, random_plan_id,
motion_representation_version, observation_operator_version,
evaluator_version, thresholds_version, output_path,
semantic_metrics, observation_metrics, quality_metrics,
joint_success, failure_reason, runtime_and_memory
```

数值结果和失败原因均保存，聚合脚本可从这些记录重建表。不得只交付一张 Excel/Markdown 平均数表或输出动画。

`evidence_index.jsonl` 逐项登记 `evidence_id, status, config_scope, protocol, artifacts, model_commit, evaluator_commit, sample_groups, limitations, paper_claim_candidates`。模型实现事实和实证结果分开；论文作者的修辞不能改变 evidence 状态。

## 10. P8：资源控制与运行阶梯

先用 CPU/少量 GPU 做 schema、单步、短采样测试；再在小型、冻结的验证面板做完整生成。推荐验证面板覆盖两配置、非对称双腕、相容属性、短中长时长，不机械继承任意旧抽样的偏好。

预算登记包含 GPU、显存上限、预计 forward/观测算子次数、样本数、每条件 K、训练/采样 seeds、停止条件。这里不规定机器未知时的固定显存或耗时；先读取机器信息和现有占用，不结束他人进程。

不要让每个分支都运行 672×5 或 full-set×20。旧可追溯结果先复评；只有最终候选及必要对照进入较大同协议比较。将“采样次数相同”和“总计算相同”分开，C 的求解和 B 的 teacher 成本要计入。

最终测试前冻结 checkpoint、阈值、guidance、评价器和 protocol commit；测试结果若失败，保留失败。后续调整使用新版本/开发数据，不能反复在同一封存集上优化却继续叫首次测试。

## 11. P9：真实 IMU 小规模验证准备

保留 Nymeria 3-clip pilot 为可行性诊断，不承担泛化主张。用户自有硬件后续测试只准备采集/校准/导入协议，不假定设备型号、输出量或时间同步已经已知。

建议覆盖两种配置、不同参与者或不同录制条件、重新佩戴后的静态安装标定、重复片段和时间同步；样本规模在设备与人力明确后制定。零样本和只微调 encoder/adapters 分开；参与者/会话级隔离，校准片段不夹带目标全身真值。

没有独立全身 GT 时，不报告全身 MPJPE。可以报告由同一生成运动重合成的观测一致性、语义编辑、盲评和失败率，但要说明输出骨架和标定误差。禁止对每个测试动作拟合隐藏参数以最小化最终评价误差。

不需要在线 demo，不扩展传感器数。硬件身份、参与同意和采集隐私由实际研究团队确认，工具不擅自上传原始人体记录。

## 12. 本会话最终交付

在 `docs/itm_rework/protocol/` 下交付库存/影响表、公共契约说明、数据审计、观测认证、相容性 manifest 与规则、划分/暴露账本、指标/阈值、运行命令、逐样本结果、可重建总表、失败案例及 `STATUS.md`。

给 01：版本化数据读取接口、观测算子、验证面板、相容训练样本与验收规则。给 03：evidence 索引、合法可写的逐条结论、样本数量/限制和可插图产物。给所有会话：契约 commit 和之后的变更记录。

**最低完成版本：**清楚的数据契约、可复用的旧证据盘点、修复 B0 可运行的 P/C 两协议、诚实命名的联合指标和测试隔离。完整物理闭环失败时报告失败并限制主张，不以代理指标偷换后宣布全部完成。

## 13. 历史依据定位

参考提交中的 `src/itm/data/standard_imu.py` 给出合成定义；`scripts/train_mdm_imu_control.py` 给出预处理和损失；`paper/latex-ITM/sec-cn/4_experiments.tex` 给出旧规模、统计、条件交换和 Nymeria pilot；`5_discussion.tex` 记载 round-trip 与评价限制。对应历史任务书为 `ITM_Codex_Review_Tasks_20260923.md`。所有数值须与实际服务器产物核查后才成为本轮 evidence。
