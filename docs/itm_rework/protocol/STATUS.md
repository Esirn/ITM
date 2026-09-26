# 会话 02 状态（更新于 2026-09-25）

**最新运行限额独立验收（2026-09-26）：**[B0_RUNTIME_ACCEPTANCE_20260926.md](B0_RUNTIME_ACCEPTANCE_20260926.md) 通过 `1d44adc` 的 CPU 软件验收，无新增返工项。166 项联合回归通过（含 23 项限额故障测试），补充三种顶层队列场景及实际微型 CPU 1000-step 循环通过。真实 GPU 显存上限仍未实测；本轮无 GPU 预检、真实训练或完整动作生成。

下一步只待原预算授权与执行前 GPU 预检，由 01 在原 200-update／36-sample／最多 4 GPU 小时范围执行一次 B0；02 随后独立评价全部输出与失败。旧输入/统计验收继续有效，无需再为相同接口往返确认。授权须另存记录绑定原预算/计划、单个运行目录及 GPU UUID，不改写已绑定数据包。

**最新方法接入独立验收：**[CANONICAL_METHOD_ACCEPTANCE_20260925.md](CANONICAL_METHOD_ACCEPTANCE_20260925.md) 通过 `1d7a32d` 的限定合成 B0 输入接入：398 帧 train-only 归一化逐值相等，6 个训练配置与 36 个验证槽位通过独立参考，143 项联合回归通过，两个旧输入报告逐字段不变。无输入接入返工项，无需重复确认。

已准备一条训练与 36 条单 case 采样命令模板，未启动。当前剩余工作是 01 落实时间/显存/更新次数硬限制及失败保留，02 核对具体运行方案；用户预算授权和 GPU 预检仍待完成。不要把输入验收当成限额验收或计算授权。公共契约及原数据包保持不变；未训练或扩散采样。

**最新实质数据交付（2026-09-25）：**见 [B0_DATA_HANDOFF_20260925.md](B0_DATA_HANDOFF_20260925.md)。固定 12 条重建候选，12/12 时间/坐标/根与 RIC 检查通过；首批限定离线合成 B0 准入 **3 train + 3 val**，5 uncertain、1 左右冲突全部保留。26 项相关测试通过。公共契约提升 v0.3-draft，旧 v0.2 原文归档。

新数据清单、实际缓存、逐条变换、GT 审核和 200-update / 36-full-sample / 最多 4 GPU 小时提案已交付。尚未训练或生成；预算等待用户授权，01 需要接入新 `canonical_smpl_paired_v1` 并绑定新 train-only IMU 归一化。数据准入仅限预先统一坐标的合成基准，不是任意真实 IMU 部署标定。**B0 不再等待相容 witness；witness 属于后续 B 分支。**

[ACCELERATION_NUMERICS_20260925.md](ACCELERATION_NUMERICS_20260925.md) 保留原四项 acceleration allclose 失败并分解精度因素；新诊断缓存可逐值重放，旧缓存仍有缺少历史后端/mesh 记录的未决重放差。新 B0 数据单独同源构建，没有把旧失败改记通过。

以下为各历史批次在交付时的记录；其中“全部未准入/等待物理对齐”等状态针对当时旧资产，不覆盖上方新限定数据包。

**最新缓存/恢复验收：**`CACHE_MIGRATION_ACCEPTANCE_20260925.md` 接纳方法提交 `b4f70fc`：五份诊断文件的原 manifest/重标 manifest 共十次拒绝全部重现；caption 报告不变；真实旧 checkpoint 缺少来源时恢复验证拒绝。独立联合回归 **126 passed**。
检查限于方法侧缓存入口和明确绑定的恢复来源；无真实训练统计重算、新训练或生成，旧 checkpoint 采样未重新执行。物理/语义及训练准入仍未认证。

**最新方法验收：**`CAPTION_METHOD_ACCEPTANCE_20260925.md` 独立接纳方法提交 `1a7e887` 的 opt-in caption 输入接口：18 条输入通过、2 条按计划阻塞、13 条 SSM 拒绝均完整复现，模型卡统计与旧 checkpoint 一致；联合回归 **109 passed**。
真实非零起点、物理时钟、坐标/骨架、语义和训练准入仍未认证；没有新 B0/witness/生成，公共契约保持 v0.2-draft。

**最新前缀核查：**`PREFIX_ALIGNMENT_20260925.md` 固定每类两个 train-only 来源组，10/10 在相同公共帧上的前缀补偿 oracle yaw 残差下降，均值 0.1580 → 0.0173 m。每类首条另做 CPU 重建，5/5 诊断 cache 支持完整片段，原 cache 不变；重叠 acceleration 的既定 allclose 4/5 未通过，负结果保留。
最终相关 13 测试通过，完成 5 次有预算的 SMPL FK；无扩散模型生成/训练/GPU。所有新 cache 为独立 diagnostic schema、training_admissible=false，未修改运行时源映射或生产 manifest，不能外推 2177 条均已认证。

**最新时钟诊断：**`HUMANML_CLOCK_20260925.md` 记录 13 条 SSM 样本与 notebook 整数降采样规则的吻合：13/13 特征帧数匹配，13/13 在相同公共帧上的 oracle yaw 残差下降，逐片段均值 0.1030 → 0.0338 m。历史原始处理产物缺失，仍未认证部署时钟或世界系。
另有 2177 条 train/val 来自 notebook 存在前缀裁剪、当前桥接未补偿的数据集；仅列为代码推导待查范围，非已确认错位数。新增诊断相关 17 测试通过；未改运行时映射/缓存，未训练生成。

**最新数据推进：**`CAPTION_ALIGNMENT_20260924.md` 完成 7,443 条 train/val 的 caption/特征区间审计，新增严格 `caption-interval-v1` 和 10 条固定案例计划（9 区间通过、1 超模型长度阻塞）；相关测试最终 31 passed。
独立于 caption 分类检查发现 13 条 SSM 来源 train cache 对当前前缀支持不足（其中 10 条 caption 区间匹配）；源裁片全部越界，13/13 现有 cache 帧数吻合静默截断后的重采样预测。已补缓存构建越界拒绝，旧 cache 不覆盖；真实源时钟成因仍待查。
6 条固定 val caption 的区间均通过，但 `000133` 必须保留索引 1。完整记录见 `caption_audit_20260924_v1/` 与 `caption_plan_20260924_v1/`。
这些是时间支持/标签范围检查，尚非物理/语义同步认证；方法入口尚未接入新 caption 接口，没有新训练或生成。

**最新：**已独立验收方法提交 `0574491` 的 seconds-v2 输入更新，联合回归 **83 passed**；详见 `INPUT_FOLLOWUP_ACCEPTANCE_20260924.md`。
方法输入报告完全复现；文档与产物的 train ID 差异已登记，另按 `000004/000008/000010` 显式补查通过。
**勘误闭环：**已核对 `0b7ed28e31dccaa9748ec91f08be8e04839f6891`，仅修改 01 的三份交接/状态文档，正确区分原 train ID、val 仅 head 与 02 补查。该记录问题已关闭；代码、旧产物及验收结论不变，无需重跑测试或继续往返确认。
公共契约保持 v0.2-draft。本批输入审计没有模型前向、训练、生成或 GPU 作业；数据/观测联合门槛仍未通过。
以下第二批与第一批条目保留各自提交和执行范围。

**第二批方法交接验收已完成；G1 接口与诊断可用，但数据/观测联合门槛未通过。G2 只有方法侧单次 update smoke；没有验证合格的 B0，G3–G4 未执行。**

当前读取方法提交 `55193655b1a8e7835de7e6bd96ed21122957641e`；公共契约升级 `itm-rework-v0.2-draft`。
新验收详见 `METHOD_ACCEPTANCE_20260924.md`、`commands_acceptance_20260924.md`。
本批不修改 01 或 03 文件。

- 方法/原协议联合 65 测试、新增 12 测试，共 77 个不同测试通过。
- 2 条保存 smoke 样本同源解码差 0；3,000 次逐步噪声 hash 复现。仍拒绝正式研究结果 schema 准入。
- 7 次 CPU forward 完成 head/wrists 物理静止对照；数值零与物理静止不同，strict null 还原原始 MDM。
- 9 条 HumanML/source 诊断不支持统一起始一帧偏移修复；坐标/骨架差仍存在，GT oracle yaw 禁止用于部署。
- 3 条 train-only 张臂候选均 uncertain，1 条目标方向失败保留；训练入口拒绝它们。compatible manifest 仍为空。

下列第一批结果/路径继续有效；其旧读取 commit 用来区分历史产物，不能覆盖成新实验。

- 读取 commit：`663414f1d9bffd6a70189846739bc9d9b1d0d873`，等于参考提交。
- 协议改动提交：本文件所在提交，可用 `git log -1 -- docs/itm_rework/protocol/STATUS.md` 定位；执行时的代码快照分别见 `code_snapshot.json`、`code_snapshot_acceptance.json`。
- 契约：`../contract.yaml` 当前 v0.2-draft；第一批 v0.1 已归档，均未冻结研究协议。
- 所有新增文件的责任方为 02。保留原任务书及旧资产，没有改 01 模型/采样实现或 03 论文。

## 已交付

| 项目 | 状态与产物 |
| --- | --- |
| 资产与旧表影响 | `inventory.md`；627 项重点资产 hash；可复用/重算/重生成边界已列 |
| 全量缓存 | 8776 条完整数组/覆盖/旋转检查通过，`full_cache_audit.json`；三个源片段重合成比较保留微小 acceleration 不一致 |
| 划分与暴露 | 749 跨 split 来源组、557 对重叠时间区间；逐 ID/来源组/区间/暴露记录齐全 |
| 过滤链 | 当前文件复现 1333→682→672，651 个排除和 10 个尾批 ID 已记录 |
| 旧输出复评 | 100-motion、30×5、30×6 属性；136 NPZ / 1880 输出身份检查；组 bootstrap、负结果/round-trip 分布 |
| 数据/观测代码 | strict cache reader、有效 stencil、mesh observer、条件重力桥接；不宣称 HumanML 输出 IMU 已认证 |
| 评价/调度代码 | P/C/D 结果核算、失败分母、分解成功率、组 bootstrap、有效集合多样性、逐步随机流、串行 CPU runner |
| 验证候选 | 6 个 val 来源组（短/中/长各 2），没有与 train/test 共享来源；待时序/属性覆盖验收 |
| 测试 | 33 passed；包含物理与无效输入、协议错误拒绝、噪声顺序不变、CPU runner 集成 |
| 证据与交接 | `evidence_index.jsonl`、`HANDOFF.md`、`commands.md`、schema 和 draft thresholds |

## 当前阻塞与责任

1. **01：**正确性及 smoke checkpoint/模型卡已交付并验收；验证合格的 B0 checkpoint 尚无。02 不越权改模型。
2. **01+02：**全段拉伸已由方法默认 crop_seconds 修复；HumanML/source-world 对齐、caption 同步仍需认证，不机械平移一帧。
3. **02+01：**HumanML→最终运动→同定义传感器闭环未通过；当前无认证生成 IMU 通道。
4. **02：**三个几何候选均 uncertain，尚无已复核 compatible witness 和独立语义/自然度阈值；C manifest 为空，阈值 draft。CPU evaluator 可运行，但真实新 P/C 成功率未产生。
5. **共同：**新生成预算、sampler、guidance、steps、显存与停止条件未冻结；GPU 作业为 0。
6. **02：**映射 test 池均在旧评价记录中出现，且存在源区间重叠；未建立合法独立封存池，backbone 预训练可见范围未完成认证。

上述是实质依赖，不是要求用户再次批准已经授权的 CPU 工作。
任务书的“最低完成版本”还要求修复 B0 可运行的真实 P/C 协议；本批**尚未达到该整体门槛**，不能称整个任务完成。
03 可根据已确认限制改写论证，不能将草案或测试 fixture 写成新实验成功。

下一步：秒轴入口已接入并复核，不再重复要求修复；01+02 仍需认证世界坐标/骨架与 caption 对齐，01 交付合格 B0；02 验收相容 witness、校准独立阈值、登记明确的小验证预算，随后才进入 G3。
