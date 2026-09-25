# 01 STATUS — 限定合成 B0 数据接入（2026-09-25）

读取 `c36ea56`，公共契约 v0.3-draft。首批 3 train / 3 val 的限定合成准入已接入，旧的全面“对齐未认证”阻塞不套用到该 scope。

- 显式 canonical package/schema/mapping/hash 消费，完整片段、审核 caption index 和直接 20 Hz GT。
- 实际训练输入 398 帧重新计算 IMU normalization，唯一源只计一次；val 不参与统计，MDM Mean/Std 不重拟合。
- 新统计已绑定输入 provenance 预检报告；尚未生成新研究 checkpoint。
- 六种采样条件含显式文本关闭、原 backbone bypass 与首次观测姿态静止 fixture；36 槽位输入已装配。
- 最终联合 143 项通过；旧两份输入报告完整重放相等。真实模型前向/训练/采样/GPU 为 0。

新交接：[CANONICAL_B0_HANDOFF_20260925.md](CANONICAL_B0_HANDOFF_20260925.md)。
下一步 02 验收实际输入/统计；预算授权、GPU 检查与受限运行调度准备完成后再由 01 执行 B0。
相容编辑 witness 不阻塞该 B0；仍是 B 分支依赖。真实设备标定与生成 IMU 闭环未认证。
以下为历史记录。

---

# 01 STATUS — 缓存与恢复训练来源核验（2026-09-25）

读取 `2de27a6` / `d360e91`，接纳 02 caption 独立验收及前缀负结果边界。公共契约仍为 v0.2-draft。

- 方法 cache wrapper 检查内嵌 schema，拒绝诊断缓存仅通过 manifest 改标签进入标准输入。
- 新 checkpoint 记录实际输入/配置/管线/MDM/normalization 来源哈希；`--resume` 在模型初始化前核验，不再静默覆盖统计。
- 旧 checkpoint 采样兼容；缺少来源记录时拒绝训练恢复。来源一致的恢复仍是 fresh optimizer warm start。
- 5 条真实诊断声明和 5 条内存改标签声明均拒绝；caption 重放 JSON 与旧报告相等。
- 定向 29 项、联合 126 项通过；无真实训练更新/生成/GPU，无新增研究 checkpoint。

详情、命令和兼容边界见 [CACHE_MIGRATION_HANDOFF_20260925.md](CACHE_MIGRATION_HANDOFF_20260925.md)。
没有迁移新缓存或重算真实训练统计。物理/语义准入、witness、预算等 G2/G3 阻塞保持不变。
下面为历史状态记录。

---

# 01 STATUS — caption 计划接入（2026-09-25）

读取 `9439818`；公共契约 v0.2-draft 未冻结。已完成显式 caption sidecar 消费，opt-in 管线
`itm-method-caption-seconds-v3`；原 seconds-v2 路径/历史产物保留。

- 严格绑定 artifact hash、caption index、半开区间与双时间轴；IMU 使用源秒轴，encoder 使用归零模型秒轴。
- 固定 10 个来源各 head/wrists：9 个通过、000004 的 199 帧全段保持阻塞；000006 实际 100 帧；000133 保留索引 1 / 59 帧。
- 已锁定 13 条 SSM 短输入全部继续拒绝。时钟 notebook 诊断没有进入部署/训练映射。
- 未准入 caption 计划在训练 CLI 的模型/设备初始化前拒绝；非零起点不能复用未重审的 B witness。
- 68 项相关测试通过；真实输入检查无模型前向、训练更新、生成或 GPU 作业。

详情/命令：CAPTION_HANDOFF_20260925.md；逐条结果：caption_input_20260925_v1.json；
代码快照：code_snapshot_caption_v3.json。没有修改 02 文件或旧 checkpoint/结果。
G2 仍只有旧单 update smoke；物理时钟/坐标/骨架/根参考与语义认证、相容 witness、阈值及预算仍阻塞正式 B0/B/C。
下面保留之前各批状态，不能将历史未完成项/运行范围套用于本批。

---

# 01 STATUS — v0.2 输入验收与记录勘误（2026-09-24）

最新读取协议提交 `c9cdc7b`，独立验收对象为方法提交 `0574491`。本次仅更正交接记录与验收状态，未改方法代码或旧产物。改动提交可用
`git log -1 -- docs/itm_rework/method/STATUS.md` 定位。公共契约 `itm-rework-v0.2-draft`，仍未冻结。

已接纳 02 独立验收：旧方法 smoke 同源解码、strict null、3000 个噪声 hash 已核对；静止 fixture 已提供，
公共 random_plan 也已随协议提交，上一批的这两项依赖已解除。

本次完成的方法入口工作：

- 训练、归一化和采样调用公共 `load_standard_cache`，拒绝 schema/物理量不符。
- 输入管线版本 `itm-method-seconds-v2`。按显式 20Hz 秒时间轴对加速度和旋转共同采样，越界拒绝，无端点夹紧；不做统一一帧平移。
- 训练不再以 IMU 更短为由悄悄缩短 motion；支持不足时拒绝。记录原始/有效帧、裁片区间、caption 区间和未认证对齐状态。
- sampling 支持 `imu_mode=static`，调用 02 fixture 并使用当前 checkpoint 的 normalization，保持 presence=true。
- B 入口要求明确 training_admissible、无未解决 blockers；metadata 与 witness 文件的实际 timestamps 都必须匹配训练片段。
- 当前 arm_spread_m + 双编辑 active-joint-proxy 原型仅接纳 head。该约束不推导成 wrists IMU 的一般物理不相容结论。

验证：48 项相关回归通过，日志 `logs/input_followup_v2.txt`。
真实输入 dry-run：train 000004/000006/000008 × head/wrists、6 条固定 val 候选的 head 输入、2 种静止输入均通过，
`input_handoff_20260924_v2.json`。模型前向 0，训练更新 0，GPU 作业 0，未修改旧 checkpoint/产物或重新标注其契约。
代码快照 `code_snapshot_v2.json`；旧 `code_snapshot.json` 保留为 02 首次验收对应的历史快照。

02 在 `c9cdc7b` 独立重现原输入报告，并完成 83 项联合回归（是 02 的验收结果，本次未重跑）。
02 另查的 train 000004/000008/000010 与本方法原报告分开记录；原文 ID 错误已在 FOLLOWUP_V02.md 显式勘误。
000006 的 caption [0,5] 秒与输入 [0,9.75] 秒不匹配，仍不能用于正式配对语义评价。

阶段：G1 输入软件正确性获独立验收，公共物理/评价契约仍未冻结；G2 仍只有单次 update 的旧 smoke；G3 无相容编辑训练/完整采样收益；C/G4/G5 未启动。
剩余实质阻塞仍是世界坐标/骨架与 caption 对齐认证、真正 compatible witness、独立评价阈值、冻结训练/采样预算和可验证输出观测链。
不使用目标 GT 做部署 canonicalization，不把新 uncertain 候选当作训练许可。

当前详细交接见 `FOLLOWUP_V02.md`。以下为第一批状态快照，历史未提交依赖/静止 fixture 缺失已由上文更新。

---

# 01 STATUS — 2026-09-24

读取 commit：`663414f1d9bffd6a70189846739bc9d9b1d0d873`。
方法提交：本文件随方法 commit 交付，可用 `git log -1 -- docs/itm_rework/method/STATUS.md` 定位。
契约：`itm-rework-v0.1-draft`（未冻结）；未修改 02 公共契约/事实索引或 03 论文。

| 阶段 | 状态 |
|---|---|
| G0 当前代码、依赖、旧 checkpoint 核查 | 已完成所列方法侧审计；静止物理 fixture 未提供 |
| G1 正确性代码/测试 | 33 项通过；strict null 真实 checkpoint 单步/1000-step 完全还原；公共契约仍未冻结 |
| G2 B0 | 编码/null/time/冻结/dropout/CFG/随机流与 CLI 已实现；真实单次 update + 保存/重载/完整采样 smoke 成功；尚无验证过的 B0 |
| G3 B 分支 | 单步相对属性响应 + paired diffusion + 双编辑 joint-proxy 训练入口已实现，梯度/冻结/模式测试通过；真实相容编辑训练与完整采样收益未验证 |
| C 分支 | 阻塞：generated-IMU certified channels 为空，HumanML→最终输出的旋转闭环未认证；不实现伪物理投影 |
| G4/G5 | 未进入最终训练、测试或论文结论阶段 |

已交付：audit.md、interface_model_card.md、commands.md、configs.json、cpu_budget.json、
logs/regression.txt、真实 checkpoint 审计 JSON/NPZ、合成左右身份过拟合记录、smoke 模型卡和逐条交接记录。
代码快照：code_snapshot.json（交付代码 hash，非声称每条早先诊断都在同一未变更工作树执行）。

smoke checkpoint：`outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/model.pt`，SHA256
`623b16e6e22c09a2e6f8aaceaa635496b8afdde932ba0b3a3326eb7fbcbb00d9`。
完整样本：同目录 `samples.npz` / `samples.json`，paired/null 共 2 条 × 12 帧，采样 26.2545s CPU。
每条完整 DDPM 1000 steps，factorized batched；两个条件随机计划 ID 相同，输出 finite。
只使用 train 来源 000004 的诊断，不作为 val 或独立性能证据。

不能写成已完成：B0 泛化收益、左右完整生成控制改善、相容编辑成功率、严格传感器一致性、B/C 优于基线、同一最终模型联合能力。

继续执行前的实质依赖（不是要求用户重复授权）：

1. 02 冻结物理秒/坐标/文本区间同步接口，并提供 source-group 隔离的 train/validation 与认证面板。
2. 02 提供 train-only compatible arm_spread_m 见证、独立评价/阈值与数值预算；当前公共 compatibility_manifest 为空。
3. 02 登记 GPU 与正式 B0/消融预算；本轮只运行文件内列出的有限 CPU 正确性 smoke，没有 GPU 作业。
4. strict B/C 需要通过输出观测闭环；在此之前只能明确命名 joint-proxy。
5. 本提交依赖 02 未提交的 `src/itm/data/random_plan.py`（hash 见快照）。没有替 02 提交、复制或重新定义它；合并时须先带入协调者对应提交。

用户已有未提交文件均保留。没有修改原始数据、旧 checkpoint、旧实验结果，没有 push 或合并主分支。
