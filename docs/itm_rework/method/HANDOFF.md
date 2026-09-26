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
