# 01 最新状态：授权 D/L/R 加载失败；CPU 修复完成（2026-09-27）

用户批准后按 `d47891f` 已验收实现启动一次队列。GPU 预检通过，D 在 `model.to(device).eval()` 处因上游 MDM 不返回 self 而失败；约 4.05 秒停止。探针前向、训练更新、完整样本均为 0，480/120 分母及失败记录完整保留，无重试。
拆开设备迁移和 eval 调用，新增直接测试加载入口的 CPU 回归；修复前复现同异常、修复后通过，联合 **204 passed / 0 failed / 0 skipped**。
新的 `correction_implementation_20260927_v1` 准备资产已核验，初始权重/输入/环境与已验收 v3 相同，仅绑定 worker 代码 hash 改变。
详情、失败实物、新 manifest 和复现命令见 [B0_CORRECTION_RUN_FAILURE_20260927.md](B0_CORRECTION_RUN_FAILURE_20260927.md)。
下一步由 **02 独立差异验收**，之后再申请新队列授权；旧授权已用于本次失败队列，不复用。真实 D/L/R 结果与修正效果仍未验证。
以下为历史记录。

---

# 01 最新状态：D/L/R 实现与 CPU 验证完成（2026-09-26）

按 `62eed84` 冻结包实现真实长度累积、R 静止参照差分、D hook 证据、共享随机计划和新多阶段受限运行器。
联合 **203 passed / 0 failed / 0 skipped**，含25项新增修正实验 CPU 测试；无预训练MDM前向、真实训练/采样或GPU查询。
共同初始权重和126个代码/输入资产已绑定，最终准备目录为 `outputs/itm_rework/method/correction_implementation_20260926_v3/`。
具体实现、测试、manifest/hash和复现入口见 [B0_CORRECTION_IMPLEMENTATION_20260926.md](B0_CORRECTION_IMPLEMENTATION_20260926.md)。
下一步由 **02 独立软件验收**，再对冻结的2小时整队申请新授权；本轮没有创建真实授权，不执行D/L/R。
以下为历史记录。

---

# 01 最新状态：B0 能力诊断与修正提案（2026-09-26）

已接纳 `dec5e93` 的独立评价：完整性通过，联合控制/实例信息收益尚未证实。
本批 CPU 读取旧权重、history、输入/输出，确认训练动作覆盖狭窄、静止条件仍可产生残差及混合长度训练的注意力上下文差异；因果归属仍需固定探针检验。
提出真实长度训练 L 与静止参照差分 R 两臂，各 200 updates、60 条完整采样，另有 480 次单步 branch-example 探针；新预算上限单 GPU / 24 GiB / 合计 2 小时，尚未授权。
详见 [B0_CORRECTION_PROPOSAL_20260926.md](B0_CORRECTION_PROPOSAL_20260926.md)。本轮模型前向/更新/采样/GPU 查询均为 0，方法代码与旧结果未改。
下一步由 02 冻结具体探针、120 槽位及评价，再由 01 实现并做 CPU 验证；新执行必须重新授权。
以下为历史记录。

---

# 01 最新状态：授权 B0 已完成，交给 02 评价（2026-09-26）

读取 `030c191`，用户已明确批准一次 B0；已通过真实 GPU 预检并执行验收过的运行器。
200/200 updates、36/36 条 DDPM 1000 步样本全部完成，无失败/重试；单 GPU，总约 9.35 分钟，峰值 allocated 约 0.687 GiB。
最终 checkpoint 与训练来源、全部样本 hash/有限值、组内随机计划完整性检查通过。
详见 [B0_RUN_HANDOFF_20260926.md](B0_RUN_HANDOFF_20260926.md)，其中包含运行目录、模型 hash 和全部槽位交付清单。
下一步由 02 独立评价固定 36 条结果；01 不自行追加训练或采样。本次运行完成不等于模型有效性认证。
以下为历史记录，其“未授权/未运行”状态不再代表本批。

---

# 最新方法交接：B0 运行限额（2026-09-26）

读取 `8046f5e`，新增受限运行器，复用 02 的一条训练和 36 条采样模板。
已落实 200 updates / 训练 1 小时、采样共享 3 小时、总 4 小时、单 GPU / 24 GiB allocated，以及失败停止和已完成产物保留。
23 项 CPU 限额/故障测试、166 项联合回归通过。详见 [B0_RUNTIME_LIMITS_HANDOFF_20260926.md](B0_RUNTIME_LIMITS_HANDOFF_20260926.md)。
下一步 02 核验本批执行限制；真实训练/生成、预算授权、GPU 预检均未发生。

---

# 限定合成 B0 输入交接（2026-09-25）

已接入 `c36ea56` 的 `canonical_smpl_paired_v1` 数据，见 [CANONICAL_B0_HANDOFF_20260925.md](CANONICAL_B0_HANDOFF_20260925.md)。
3 train / 3 val 准入仅适用于预先统一坐标的合成基准；新管线 `itm-method-canonical-b0-v1`。
3 个训练源的 398 帧 20 Hz acceleration 已重算并绑定来源，36 个验证条件槽位已完成输入装配。
联合 143 项通过；真实模型训练/前向/采样/GPU 均未执行。下一步由 02 重放输入，然后等待预算授权与 GPU 执行准备。

---

# 缓存与恢复训练交接（2026-09-25）

已读取 02 的 `2de27a6` / `d360e91`。新交接见 [CACHE_MIGRATION_HANDOFF_20260925.md](CACHE_MIGRATION_HANDOFF_20260925.md)。
方法入口拒绝 diagnostic NPZ 被 manifest 改标签；新 checkpoint 绑定实际输入来源与 normalization，恢复时严格核对。
5 份真实缓存的原声明/改标签声明均拒绝；固定 caption 重放与旧报告相同；联合 126 项通过。
没有训练/生成/GPU 作业。旧 checkpoint 采样不变，缺少来源记录时不能从当前训练 CLI 恢复。

---

# caption 方法交接（2026-09-25）

caption 计划接口已接入，见 [CAPTION_HANDOFF_20260925.md](CAPTION_HANDOFF_20260925.md)。
固定 10 来源各两配置：9 通过、000004 阻塞；13 条 SSM 短输入仍拒绝，68 项测试通过。
没有训练/生成/GPU 作业。公共计划与旧产物保持不变；下方仅为历史交接记录。

---

# seconds-v2 独立验收后的记录更正

已读取 02 的 `c9cdc7b` / INPUT_FOLLOWUP_ACCEPTANCE_20260924.md。
方法原检查的 train ID 更正为 **000004/000006/000008**（各 head/wrists），6 条 val 仅 head；
02 单独补查的 000004/000008/000010 不计入方法旧报告。详见 FOLLOWUP_V02.md 的显式勘误。
原 JSON/快照/checkpoint/样本保持不变；83 项联合回归属于 02 的独立验收，本轮未训练或重新生成。
caption、坐标/骨架、相容见证及合格 B0 的阻塞仍在。

---

# 第二批方法入口交接

读取 02 提交 `1bc3a41` 后完成 v0.2 输入适配，详见 [FOLLOWUP_V02.md](FOLLOWUP_V02.md)。
代码测试与输入 dry-run 已通过；尚未启动新训练/生成实验。下文保留上一批历史交接。

---

# 01 → 02 / 03

读取 commit `663414f1d9bffd6a70189846739bc9d9b1d0d873`；契约 `itm-rework-v0.1-draft` 未冻结。
改动文件/hash、命令和执行结果分别见 code_snapshot.json、commands.md、audit.md；阶段状态见 STATUS.md。
本会话只修改方法/采样/训练和测试；实际缓存桥 `src/itm/backbones/mdm.py` 明确由 01 负责。

## 给 02

- B0 正确性已具备：33 项回归、真实 Stage-2b 非零 bias 审计、完整 DDPM 随机流重放。请勿把 smoke checkpoint 当验证过的 B0。
- 已接入你的 `sampling_noise`，每步按 case/seed/backbone/shape 寻址，记录完整地址/hash。该公共文件仍未提交；请在协议提交中带入，方法不复制一份实现。
- 新采样按单条有效长度运行，保留同一条的四路 batched CFG。原始 MDM 自身 mask_frames=False，padding 改变有效帧输出。不要跨长度/随机形状比较时假设噪声仍相同。
- 默认停止全段时间拉伸；使用 crop-local 秒裁切，短 donor 报错。这没有解决 HumanML/IMU 的一帧偏移和世界坐标差，训练仍须显式 diagnostic flag。
- 逐条诊断产物交接：smoke_records.jsonl，原始 normalized HumanML263 与 joints 同源；输入 acceleration/orientation 不能当 generated IMU。记录尚未经过公共 result.schema/正式评价器验收。
- 输出只具 RIC joints；请不要宣称可恢复正确 wrist orientation。C 阻塞不变。
- B 初版只实现 arm_spread_m（水平腕间距均值）方向/有界 margin；请确定它与所需编辑属性是否一致。CLI 使用公共 compatibility 字段并拒绝无见证/非 train/非 compatible 数据。需要你交付 train-only witnesses、attribute 规则/阈值及成本预算。
- 已保留 legacy/null-only/identity-only/combined/no-time/FiLM 开关与 diffusion-only/trajectory-only，尚未做验证选择。
- checkpoint/modelcard 路径与 hash 在 STATUS.md；样本没有经过成功过滤。第一次无缓存 motion 的失败也保留。
- 没有改你的 STATUS、contract 或 evidence_index。请仅把实际完成的正确性审计与 smoke 状态纳入公共事实索引。

## 给 03

实际已实现：有序融合、完整残差 strict gate、秒时间位置、确定的冻结模式、四路 CFG 测试、逐步噪声重放、
训练期单步相对属性响应实验、简单 FiLM 对照接口。身份保持/CFG/零初始化首先是正确性与基线，不包装成新颖机制。
旧 checkpoint 的 legacy null 确实不等于原始 MDM；量级见 audit.md，仅为一个短 CPU 诊断，不可外推成全部旧实验无效。
新 strict null 可以还原原始输出；这不等于解决语义编辑/生成控制。

尚未验证：B 相对 absolute anchor 的生成优势；完整生成的左右归属；B0/FiLM 的预算公平比较；C；B+C；
相容语义成功、严格传感器一致性和最终模型联合能力。论文不得把这些写成结果。
