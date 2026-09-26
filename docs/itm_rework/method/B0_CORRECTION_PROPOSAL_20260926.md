# 01 → 02：B0 能力诊断与下一轮修正提案

读取 **`dec5e93`**。接纳 02 的结论：完整性通过，但联合控制有效性与真实 IMU 实例增益未证实。本轮只读取代码、已有权重、history 和保存的输入/输出，执行 CPU 数组统计；**模型前向、训练更新、扩散采样、GPU 查询均为 0**。没有修改运行代码、原 36 输出或公共评价口径。

建议先验证两个具体机制：混合长度训练与单条采样的注意力上下文差异，以及条件分支在静止参照下仍能产生的共同动作残差。随后用两臂小实验检验一个明确的修正：按真实长度训练，并将新增条件残差定义为相对同一片段物理静止参照的增量。该修正尚未实现或证明有效；不把再训练更久作为首选。

## 已定位的事实及边界

证据为 [existing_artifact_diagnosis.json](b0_diagnosis_20260926_v1/existing_artifact_diagnosis.json)，来源 hash 在 `sources.json`。可复现命令：

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python \
  docs/itm_rework/method/b0_diagnosis_20260926_v1/inspect_existing.py
```

### 1. 训练覆盖和固定动作偏置

| train ID | 帧数 | 已审核 caption 的粗动作 |
| --- | ---: | --- |
| 000506 | 136 | 交替腿前跨/蹲起 |
| 004728 | 158 | 右腿向后抬起并保持，身体前倾后放下 |
| 000027 | 104 | 出拳、转身 |

两个腿部来源共 294/398 帧（73.87%）；训练按样本等权，故这是帧覆盖描述，不是 loss 权重。没有正常行走、拍手或踢腿后站回的同类训练覆盖。100 epochs 对同 6 个 source/config 条目共使用 600 examples。控制 encoder 参数 6,607,872，8 层 adapter 参数 2,101,248，共 **8,709,120**；不是 600 个独立动作。

实际条件计数为 `00=2, 10=51, 01=56, 11=491`（首位 text，次位 IMU），IMU-present 共 547。strict-null 时控制分支被完整屏蔽，text-only 的 51 个样本不能为这条分支提供文本保持梯度；它们仍计入原 batch loss/更新分母。原训练只有 diffusion x0 MSE，未显式约束 paired 对 static 的增量，也没有开启文本保持辅助目标。

首 10 / 末 10 epoch 平均 diffusion loss 为 0.08860 / 0.04818，但 timestep、noise、dropout 都在随机变化，且包含冻结 null 路径。这不是同一探针下的收敛曲线，也不能区分欠训练与过拟合。encoder 从 epoch 1 到最终权重 L2 变化约 2.263，所有 adapter 非零；不存在“整个控制器根本没更新”的证据。

训练覆盖与 000139 抬腿负例相符，支持将共同动作偏置列为首要假说，**尚不能断言训练样本 004728 导致该输出**。没有对 epoch/guidance 进行试采样或挑选。

### 2. 非零静止参照可通过条件分支产生残差

现有 additive adapter 每层为 `r_l = W_l E(u, mask, seconds) + b_l`。encoder 包含传感器存在位、线性偏置、时间编码及 Transformer；物理静止时 acceleration 为 0、orientation 保留第一帧，归一化后也不是零向量。`IMU present=true` 就会保留整个残差。strict-null 只保证关闭分支时还原骨干，不约束开启分支但缺少运动信息时的行为。

最终 adapter 权重 Frobenius 范数约 0.244–0.383，bias L2 约 0.0105–0.0166。没有 hidden activation 的尺度就不能据此说“bias 主导”；单纯删 bias 也不能消除 `W_l E(static)`。零初始化只能保证初始化时不影响骨干，不能保证训练后语义保持。

已有保存数组还排除了“000139 两种输入本来完全一样”的解释：

| 000139 配置 | active 归一化输入 paired−static RMS | 其中 acceleration RMS | orientation RMS | 输出平均关节间距 |
| --- | ---: | ---: | ---: | ---: |
| head | 0.28377 | 0.48602 | 0.16920 | 0.00410 m |
| wrists | 0.23268 | 0.36807 | 0.16441 | 0.00437 m |

输入 RMS 为无量纲分量统计，不能与米数做单位混合的“衰减率”。拍手 head 的对应输入 RMS 仅 0.03798，确实比 wrists 的 0.59654 弱，不能要求 head 对手部动作提供同样信息。上述差异没有证明 encoder 丢信号、adapter 忽略信号或扩散过程抹平差异中的哪一种；需分层探针定位。

### 3. 明确的训练/采样长度差异

`_collate` 将 batch 中不同长度补到最大长度；`_predict_xstart_and_diffusion_loss` 对整个补齐后的 motion 加噪，再送骨干。IMU encoder 与 loss 有 frame mask，**加载的 MDM `mask_frames` 缺省为 false，骨干注意力不屏蔽补齐帧**。因此有效帧可以关注补齐后的带噪 motion token；给 loss 加 mask 并不能消除这个影响。采样则按单条真实长度执行。

6 条训练项的长度为 `[104,104,136,136,158,158]`；batch=3、不重复抽取，所以全部 20 种三项组合都含混合长度。均匀 shuffle 下预计 13.58% token 为 padding（组合期望，未冒充实际 batch 重放）。这是代码可确认的上下文差异；是否足以造成抬腿偏差仍未测量。不能直接把 backbone 改成 `mask_frames=true`，这会改变既有骨干语义；首选消除训练端多余 token。

### 4. 暂不支持归因于 CFG 实现错误

当前 factorized 公式为 `f00 + st(f10-f00) + si(f01-f00) + sj(f11-f10-f01+f00)`。原配置 `si=sj=1` 时代数化简为 `f11 + (st-1)(f10-f00)`，IMU-only 项抵消。故 paired 劣化不能仅由“IMU-only 只有 56 个训练例”直接解释；也不是 IMU 被重复加了两次。02 已验证 strict-null 与 backbone 完全相等，传感器张量和时间轴正确。文本 token 未直接加残差，不代表后续 attention 不受运动 token 改变影响。

这里没有认定全部实现无误；只排除了几种缺乏支持的捷径解释。没有重新跑已经通过的软件验收。

## 下一轮固定设计：D 探针 + L/R 两臂

这是新开发诊断实验，旧数据/结果仍保留。现有 3 val 已参与方案设计，继续标为 development validation，不能包装成独立最终测试。训练仍限原 3 train 来源；不将静止/扰乱输入伪装成新增配对 GT 或扩充物理准入。

### D：只读最终模型的定点探针，不更新、不生成完整样本

固定最终 `44d2c330…`，不搜索 epoch。6 source × head/wrists，共 12 项；timestep `{50,250,500,900}`，固定两个 noise replicate（seed 20260927/20260928）。每项四种输入：text-only、paired、physical-static、tail-reversed。共 **384 个 MDM branch-example forwards**。全部探针（含下述 padding 对照）已列在 `b0_diagnosis_20260926_v1/proposed_probe_slots.json`。

`tail-reversed` 固定第 0 帧，联合反转 acceleration/orientation 的第 1 至 T−1 帧；保留 timestamp/mask/length 和第 0 帧静止参照。它是故意破坏时序对应的敏感性输入，不是物理合法动作或同类 shuffled donor，不用于训练监督。使用同一个带噪 GT x_t（teacher-forced）比较输入影响；GT 仅用于开发诊断，不进入部署 canonicalization。

额外在 train 的 6 项 × 4 t × 2 noise 下，对 text-only / paired 做原 padding 上下文对照：**96 个 branch-example forwards**，总 D=**480**。按 head/wrists 各组 `[104,136,158]`；有效帧 noise 与单条探针完全相同，padding noise 使用另一个固定地址。所有模块 eval，排除 dropout 差异。只比较共同有效帧。

记录每层 encoder paired−static 差异、adapter 残差及其相对骨干 hidden RMS、带/不带 static 的输出差异、x0 masked MSE，以及同一条在 padded/unpadded 下的变化。按 t/source/config 全部保留，不能只报平均。若 encoder 差异已弱，定位到编码端；若编码差异明显而 adapter/输出差异弱，分别定位后级。teacher-forced 的观察不代替自由采样结论，也不预设“多大才有效”的新阈值。02 在执行前冻结描述性汇总与解释边界。

D 不用于挑 checkpoint、guidance、loss 权重；L/R 配置在下文预先固定。D 发现新的确定性实现错误时停止并交接，不能边试边换设计。

### L：真实长度训练对照

从同一预训练骨干和全新初始化控制器开始，200 updates；不继续旧模型。保留原 encoder/additive adapter、lr=5e-5、AdamW 默认、diffusion-only、text/IMU dropout=0.1、train-only normalization 与原 guidance。

每个原 effective batch=3 拆成 3 条真实长度 microbatch，每条按其原长度执行 MDM，loss/3 累加梯度，**每三条只 step 一次**。沿用预登记样本次序；两臂共享每条的 timestep/noise、文本/IMU dropout 地址。保持 600 example-exposures，不通过变 batch 把 200 updates 偷换成 600。encoder dropout=0.1 保留，按显式每条地址控制。骨干保持 `mask_frames=false` 和 frozen_eval。

L 相对历史运行还会改变执行/RNG组织；D 的 matched padding probe 才是隔离 padding 影响的对照。不能把 L 对旧输出的所有改善都归因于 padding。

### R：在 L 上只增加“相对静止参照的残差”

对每个 clip，从**观测**第一帧 orientation 和零 raw acceleration 构造 `s(u0)`，使用原 train-only normalization、相同 mask/timestamp，定义：

`r_l(u) = W_l [E(u, mask, seconds) - E(s(u0), mask, seconds)]`。

adapter 不增加独立 bias（实现可保留旧 schema 的 bias 张量但此新模式不使用且冻结），避免在差分后重新加入常量通道。两 encoder 分支共享权重；训练 dropout 使用同一组随机 mask，参考分支不 detach，避免对同输入引入随机/梯度偏差。随机状态局部保存/恢复，不能改变后续 timestep/noise/dropout 的已登记流。采样 eval 下分别预编码一次，缓存差分；每步 MDM 分支数不增加。

严格控制分支关闭仍由原 gate 决定。物理静止输入等于自己的参照时残差应为零，恢复 text-only 是**架构约束**，不能拿它作为新实验发现或传感器增益证据。`numeric_zero` 仍是归一化零张量，其参照使用该 case 原始观测第一帧 orientation，不将零矩阵当合法旋转；预计它不必等于 text-only。tail-reversed 保留第一帧以固定同一参照。

两臂初始化完全相同的 encoder 与 adapter W（显式克隆一次初始 state），除差分定义外其余设置相同。均只评最终 update=200，不做 epoch/guidance/学习率搜索，也不同时添加文字 anchor、jerk 或对比 loss。

**风险**：R 有意丢弃静态绝对姿态作为控制信号；静止输入本来可能携带有意义的朝向，因此此模式首先是检验共同残差假说的消融，不是可直接推广的完整 IMU 设计。共享非线性 encoder 中仍可能存在对所有动态输入相似的残差，差分不能保证实例识别。可能仅将条件分支整体压小、恢复文本而不读真实时序；必须用下列对照揭示，不能宣布问题已经修复。

### 每臂固定 60 条完整采样，共 120 条

- 原 val 36 槽位（3 source × 2 config × 原六条件），保留 seed=20260926、原长度和同组 noise plan。
- val 新增 tail-reversed 6 条，与各自 paired 同文本/噪声；只能评价时序敏感性，不能称同类实例优势。
- train 18 条（3 source × 2 config × paired/text-only/physical-static），只用于记忆/训练能力诊断，单独报表，不混入 val 平均。训练样本看过 GT 不等于模型在训练分布已能自由生成，需实际采样检验。

两臂相同 case/config 使用相同随机地址；val 保留旧地址；train 使用显式 train 前缀的唯一地址避免碰撞。所有 case K=1、1000 steps、factorized batched CFG 2.5/1/1，最终 checkpoint 唯一。L/R、train/val 都列全分母，不替换坏样本。

## 如何解释结果与下一责任方

请 02 在执行前冻结新 120 槽位、D 探针明细、各 split 的评价/审阅和负例保留规则。沿用原 continuous metrics；姿态误差、时序动作、运动幅度与全局位置同时报告，000139 继续不套平地接触假设。

- 若 L/R 的 train 配对自由生成也失败，不能单凭训练 loss 降低说“已学会，只是泛化差”；先报告训练能力未建立。
- 若 train 改善而 val 仍失败，优先请求 02 扩展 source-group 隔离、语义审核过的动作覆盖及同类近长度 donor。当前 3 条数据不能支持泛化结论；不由 01 自行采纳 uncertain 数据。
- 若 R 仅恢复 text-only，paired 对 static/时序扰乱没有目标一致的增益，判为未显示动态 IMU 利用，不能因静止对照恰好为零残差称成功。
- 若 paired 对两类输入对照有一致的目标位置/速度与粗语义收益，还需检查 root-relative 姿态没有再次劣化、行走未变抬腿；仅报告此开发面板上的机制证据，K=1、n=3 不做总体认证。
- 02 负责冻结具体比较/数值容差及完整审阅；01 不根据新输出追加通过阈值或删负例。B/C、真实设备标定和严格 generated-IMU 闭环仍不进入本次范围。

## 新预算提案（待冻结、实现和新授权）

120 条建议槽位已逐条列在 `b0_diagnosis_20260926_v1/proposed_sampling_slots.json`，无运行目录或 GPU 绑定。机器可读提案见 [proposed_budget.json](b0_diagnosis_20260926_v1/proposed_budget.json)，当前 `proposal_not_authorized`，不是运行器可执行授权。

| 阶段 | 固定计算量 | 时间硬上限 |
| --- | --- | --- |
| D | 480 branch-example forwards，0 update，0 完整采样 | 10 分钟 |
| L/R 训练合计 | 2 个 fresh run × 200 updates；共 1200 example-exposures | 每臂 10 分钟，合计 20 分钟 |
| L/R 完整采样合计 | 2 × 60=120 条，120,000 去噪调用 / 480,000 branch-example forwards | 两臂共享 90 分钟 |
| 总计 | 单 GPU、≤24 GiB allocated | **120 分钟 / 2 GPU 小时** |

执行顺序固定为 D → L 训练 → R 训练 → L 的 60 条采样 → R 的 60 条采样。新训练只在完成 update 50/100/150 及最终 200 时发布 checkpoint，故障保留最后完整文件，最终选择规则不变。各阶段加载、保存、核验计入总期限，采样时间不按条或按臂续期。训练主干 forward 上限共 1200 example-forwards；R 参考 encoder 的额外计算不冒充零成本，计入训练期限并单独计数。硬上限不是预测耗时；原 36 条约 8.8 分钟的采样记录只作为量级依据。编码差分与 microbatch 的新开销尚未实测。

沿用原停止原则：任一超时、OOM、非有限值、身份或计数异常立即停止整个实验队列，保存已有产物和未运行分母，不自动重试/换卡/转 CPU，也不为某臂挪用其他臂 update 额度。旧授权已经消耗，不能沿用旧目录/旧 plan 或修改原预算后重用其 hash。

**执行前顺序**：02 冻结这份具体实验的输入/评价与队列 → 01 实现真实长度累积、差分残差和新版本多臂运行限额，完成 CPU 软件验证 → 02 核验这些变更 → 用户对完整新队列授权 → GPU 预检 → 单次执行。当前受限运行器写死一次 200-update/36-slot，不能直接承载本提案；明确需要新版本，不能绕过其限制。CPU 验证重点是累积梯度归一化与 step 计数、共享 dropout 下静止残差的值/梯度为零、null/bypass 不变、参照来源、两臂随机地址相等及全队列总期限/故障停止。

本批交接已完成诊断事实、候选修正、对照和预算，**不在代码未实现/队列未冻结时提前请求执行授权**。下一步请 02 审阅并冻结实验设计；01 据此实现，不重复旧 36 输出的完整性验收。
