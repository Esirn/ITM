# 会话 02 状态（2026-09-24）

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
