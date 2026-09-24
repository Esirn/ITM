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
