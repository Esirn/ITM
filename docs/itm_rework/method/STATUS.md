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
