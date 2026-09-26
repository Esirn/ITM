# 02 → 01：D / L / R 修正实验设计与评价冻结

读取 **`cd1079e`**，承接 `dec5e93` 的完整 B0 负结果。**设计与评价已冻结，01 可直接进入实现和 CPU 软件验证。** 采纳“覆盖狭窄、共同静止残差、padding 上下文差异”为待检验线索，不将其升级为已证实因果。保留 D=480、每臂 200 updates＋60 条采样、合计 2 小时的提案额度；本文件不是计算授权。

本批仅构造 CPU 计划、生成有限的随机数组并核验哈希；**模型前向、训练、采样、GPU 查询均为 0**。原 36 条输出、原数据准入、旧契约、旧预算/授权及 01 文件均不改写。变更提交由 `git log -1 -- docs/itm_rework/protocol/B0_CORRECTION_FREEZE_20260926.md` 定位。

## 冻结入口与身份

入口：[package.json](b0_correction_20260926_v1/package.json)，SHA256：

`e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a`

这是 `canonical-B0-correction-package-v1`，状态 `frozen_for_implementation_not_authorized`；授权、运行目录及 GPU UUID 均为空。公共数据契约继续 `itm-rework-v0.3-draft`，只复用原已准入 3 train＋3 development-val。新实验包不把旧数据包中的历史 pending 字段改成授权。

| 文件 | 用途 |
| --- | --- |
| [design.json](b0_correction_20260926_v1/design.json) | L/R 唯一差异、初始化、训练/输入定义、D 上下文、数值容差、输出要求 |
| [training_schedule.json](b0_correction_20260926_v1/training_schedule.json) | 两臂共用的完整 600 次样本呈现；每个更新恰好三条 |
| [probe_slots.json](b0_correction_20260926_v1/probe_slots.json) / [probe_noise.json](b0_correction_20260926_v1/probe_noise.json) | 480 个探针和 96 组有效帧噪声；padding 有独立地址 |
| [sampling_slots.json](b0_correction_20260926_v1/sampling_slots.json) | 两臂全部 120 槽位、caption/GT/条件/随机地址 |
| [evaluation.json](b0_correction_20260926_v1/evaluation.json) | D 与自由采样的分层指标、全部对照、失败分母及解释限制 |
| [review_order.json](b0_correction_20260926_v1/review_order.json) | 全 120 输出的预登记条件/臂隐藏审阅顺序，seed=20260929 |
| [budget.json](b0_correction_20260926_v1/budget.json) | 阶段顺序、硬计数、共享期限及停止规则 |
| [sources.json](b0_correction_20260926_v1/sources.json) / [validation.json](b0_correction_20260926_v1/validation.json) | 提案/旧证据/数据/骨干/构造器 hash 与 CPU 计划核验结果 |

这些冻结文件不得在首次真实前向后改定义。实现暴露接口歧义时列出具体差异交给 02；若需改变实验，出新版本并说明与本包的关系，不能静默覆盖 hash。

## D：一次前向的配对探针

固定历史最终 checkpoint `44d2c330…`；6 个来源 × 2 配置 × 4 timestep（50/250/500/900）× 2 noise seed（20260927/20260928）× 4 条件，共 **384 个 exact-length branch-example forwards**。四条件为 text-only / paired / physical-static / tail-reversed；**均保留文本，分别是单个 f10 或 f11，不运行四分支 CFG**，不进入反向扩散循环。

每组用原始 GT 经固定 Mean/Std 归一化后构造同一个 x_t；噪声算法沿用 PCG64 addressed noise，组地址为 `canonical-B0-D-{split}-{motion_id}-{config}`，step 为探针 t。所有条件复用同一有效帧数组。此处 GT 仅是 teacher-forced 诊断目标，不进入传感器标定或生成时坐标拟合。

附加 **96 个 padded train branch-example forwards**：在相同 config/t/seed/condition 下，将 train `[000027,000506,004728]` 按 104/136/158 帧补成 158，text-only 与 paired 各一组。共 32 个 batch-size-3 前向，按 96 个 branch examples 计费，不能只计 32。归一化 motion 补零，然后对整个 tensor 加噪；有效区间噪声直接复用 exact 探针，追加的尾段用 `…-padding` 独立地址。IMU 和时间的补齐部分为零，encoder/loss mask 排除它；骨干 `mask_frames=false` 保持不变。

158 帧来源没有新增 padding，其差异只作为 batch shape/数值参照，不能混同于 104/136 帧的 padding 处理效应。所有模块 eval，排除 dropout 差异；只在共同有效帧统计。D 没有指定效果阈值，有限但不理想的诊断值不允许临时更改 L/R；确定性实现错误、身份/计数异常或非有限值才停止整队。

D 记录融合后/时间编码后、两层 temporal encoder 及最终 E，每层 MDM 加残差前的运动 hidden h_l 和 gated residual r_l（8 层），x0 masked MSE，以及配对差异。RMS 按有效帧×特征元素计算；残差/hidden 比值分母为零时记 null 并保留原量，不加任意 epsilon 掩盖问题。排除前置文本/timestep token 与 padding。按来源、配置、t、replicate、层保留全部值；先在来源内平均两次 replicate，再按三个来源等权，train/val 分开，不用跨 t 的单个均值作结论。

为了 02 能独立重算，保存这些前向内取得的有效帧 float32 激活与 x0、mask、x_t hash 和标量 reductions；相同张量允许按 hash 去重引用。**通过 hook 读取，不为记录多跑模型**，不以训练后的 L/R 再追加 D 探针。

## L / R：共同训练计划和唯一对照定义

两臂都从同一个新初始化控制器开始；共同初始化 seed=20260925，显式克隆 encoder、adapter W 和初始零 bias，保存初始 tensor hash。旧 checkpoint 只用于 D，不是两臂恢复训练来源。原骨干冻结 eval、`mask_frames=false`，原 398 帧 train-only IMU 统计和预训练 motion Mean/Std 均保持绑定。

- **L**：原 `W_l E(u)+b_l`，bias 可训练；每条按真实长度前向。
- **R**：`W_l [E(u)-E(s(u0))]`，adapter bias 不使用且冻结。s 用原始观测第一帧 orientation 与 raw acceleration=0 构成，不用 GT。参考 encoder 不 detach，与观测分支共享权重和逐层 dropout mask。

R 两次 encoder 不能简单拼 batch 后各自随机 dropout；在每条呈现的独立 RNG 上下文内重放同一随机状态，退出后恢复调用者 CPU/CUDA RNG，保证不会改变后续地址流。采样 eval 下预编码 observed/reference 并缓存差分，每步 MDM 分支数不增加。该消融取消纯静止信号的残差，但非线性 E 仍可使动态响应依赖初始姿态；不能据此宣称彻底消除了绝对方向信息。

每臂恰好 100 epoch × 6 个 source/config 条目＝600 exposures；每三条真实长度 microbatch 的 loss 各除以 3 累积，**只 step 一次，共 200 updates**。每条 loss 为有效 263×T 元素的 normalized x0 MSE，不按长度加权。完全冻结 null 项可跳过无梯度图的 backward，但 loss 分母、呈现数和前向计数仍保留；不能只用 IMU-present 条目重新归一化。

共同 AdamW lr=5e-5、betas=(.9,.999)、eps=1e-8、weight_decay=.01，其余保留旧记录默认（含 foreach/fused=null）；绑定相同 torch/device/backend，落盘实际配置。不增加辅助损失、学习率搜索、梯度裁剪、AMP 或 scheduler。

逐条计划已经把 shuffle、timestep、text/IMU dropout、训练噪声和 encoder dropout seed 分开寻址。随机地址为 SHA256(canonical JSON)，键包括 version、seed、namespace、epoch、position、motion_id、sensor_config；**没有 arm 字段**。顺序从排序后的 ID×head/wrists 人口中按 epoch 独立 PCG64 排列；每 epoch 六项恰好各一次。训练 noise 用完整 hash 的整数初始化 PCG64，生成 float64 standard_normal 后转 float32，保存 C-order hash。Torch encoder dropout 使用登记的 63-bit seed，不另抽条件 mask。

两臂相同的条件计数为 **00=4、10=58、01=43、11=495**（首位 text，次位 IMU）。这不是历史 B0 的随机流重放；**L 对历史输出的差异不能全部归因于 padding**。只保存完成 update 50/100/150/200 的训练 checkpoint，选择唯一 final200，不根据 D/损失/训练或验证结果选 epoch。

## 完整采样与评价

每臂固定 **60 条**，顺序按槽位文件：原 val 36、val tail-reversed 6、train diagnostic 18；两臂合计 120，K=1、1000-step DDPM、float32/eval、factorized batched CFG 2.5/1/1、condition cache 开启。原 val 地址和 seed=20260926 完整保留；train 使用提案的 `canonical-B0-probe-train-{ID}-{config}` 唯一地址。相同来源/配置/长度/条件在两臂使用同一随机计划，但各臂只能使用自己的 final200 checkpoint。

**tail-reversed 在 30→20 Hz 插值之后变换 raw 数据**：共同索引 `[0,T−1,…,1]` 反转 acceleration/orientation，保留第 0 帧、timestamps/masks/长度，然后用原统计归一化。不做速度变号、旋转取逆或重算加速度；它是故意破坏对应关系的输入，不能当物理有效逆动作。比较仍对原 paired GT，仅能说明时序扰乱对原目标的影响，不是同类 donor 优势。

R 的 numeric-zero 仍为全零归一化输入，但参考必须从**原观测 O[0]**构造；不能把零矩阵当旋转。物理静止输入等于自身参考时，恢复 text-only 是架构约束，**不作为实验发现或传感器增益**。

沿用原 global/root-relative MPJPE、root RMSE、head/wrists joint proxy、velocity RMSE、绝对 jerk 与 GT、各轴 extent 和 pelvis path length。无坐标/时间拟合，不更改旧效果定义。

- val 原六条件、tail-reversed 分表；train 三条件单列，不混入 val 平均。
- 接触速度仅用于已冻结的 val000433/000382；val000139 非平地不适用。train 三条尚未冻结平地资格，接触速度统一 N/A，不能猜测或记零；其它指标照算。
- 所有来源内 paired−text-only、paired−static，以及 val paired−reversed、paired−zero；各条件 R−L；另列两臂“paired−对照”的差值之差。保留逐条指标、差值和输出之间的平均关节距离，后者仅表示敏感性，不等于正确控制。
- 每格按来源等权；完整均值 n=3，缺失时标注 partial mean 与实际 n。总失败分母 120，另报每臂 60、val42/train18；配对比较只取双方都有输出的组，同时列缺失数和原因。不删负例或以成功条数重写计划分母。
- 全 120 条按预登记顺序隐藏 arm/config/condition 审阅，目标粗动作取原已审核 caption：train 为交替弓步/蹲起、右腿向后抬起保持后放下、出拳后转身；val 为原踢腿/拍手/步行。保存观察和 uncertain，不新增 semantic pass rate、JSR 或自动阈值。

若 R 只恢复 text-only，真实 paired 对 static/reversed 没有目标一致收益，记为未显示动态 IMU 利用；若 train 也失败，不能用 loss 降低声称已学会自由生成；若 train 好而 val 仍失败，下一步才考虑单独审核的覆盖扩充。任何正面迹象仍只是 n=3、单训练 seed 的开发面板证据；不作泛化、显著性、K=1 多样性、严格生成 IMU 闭环、相容编辑或真实设备认证。

## 数值容差与 01 实现验收项

本次预先冻结的是**软件数值一致性容差**，不是模型效果阈值。所有实际连续差值必须保存：

| 检查 | 定义 |
| --- | --- |
| 输入参考 | acceleration exact；orientation/normalized_imu atol=2e−7，rtol=0 |
| RIC 重放 | atol=2e−6 m，rtol=1e−6，保留 max_abs |
| R 静止差分 | 同输入、共享 dropout 时编码差与注入残差严格零；不能重新加 bias |
| R 静止梯度 | CPU fixture 的 encoder/adapter 梯度 max_abs≤1e−7；在 optimizer/weight-decay 前检查 |
| null / R-static 单步 x0 | 同设备 matched 输入，atol=1e−6，rtol=1e−6；目标为逐值相等 |
| 完整采样 null / R-static | 最大坐标差≤1e−5 m，同时报 bitwise 与连续值；不配准。超出为实现/重放差异，不是动作质量失败 |
| 累积梯度 | CPU 对显式三个独立目标之和，atol=1e−6、rtol=1e−5；更新和样本计数严格相等 |
| 哈希、mask、种子、计数 | 精确一致，无容差 |

01 的 CPU 验证应覆盖：真实长度 loss/3 累积与 step 计数（含 null）、两臂共同初始 state/随机计划、R 共享 dropout 的值/梯度、原始 O[0] 参考来源及 numeric-zero/tail-reversed、旧 null/bypass 与旧 checkpoint 路径不变、D 有效帧噪声复用和 padding、缓存失效身份，以及跨 D/L/R 的总期限、部分训练/采样失败保留。这里冻结的是要求，**本批没有执行或声称通过这些尚未实现的方法测试**。

## 队列、预算与下一步

| 顺序 | 硬计算量 | 阶段硬期限 |
| --- | --- | --- |
| D | 480 MDM branch examples；0 update / 0 full sample | 600 秒 |
| L train | 600 MDM example forwards，200 updates；encoder≤600 | 600 秒 |
| R train | 600 MDM example forwards，200 updates；encoder含参照≤1200 | 600 秒；L/R 训练还共享 1200 秒 |
| L sample → R sample | 120 full samples，120,000 denoising calls / 480,000 MDM branch examples | 两臂共享 5400 秒 |
| 总队列 | 单 GPU、allocated≤24 GiB | **7200 秒** |

总期限从队列启动（GPU preflight 前）计时，加载/保存/核验全计入；各阶段都取总 deadline 与本阶段 deadline 的更早值。训练共享期限从 L train 开始；采样共享期限从 L sample 开始，R 不续期。不转移未用的阶段上限，不跨臂挪用 update/slot。额外 encoder 计算单独计数且计入时间，不冒充零成本；提取 D hooks 不增加前向，不能插入未列明的真实 GPU warmup/试采样。

任何超时、OOM、非有限值、身份或计数异常停止整队；保留完成产物、最后完整 checkpoint、失败/未运行槽位及完整分母；不自动重试、换卡、CPU 回退或把 partial 模型称 final。

**下一责任方：01 实现上述方案及新版本多臂限额运行器，完成 CPU 验证并交付具体代码/计划/初始化来源。02 再做独立软件验收；随后才对完整新队列申请用户授权并绑定唯一目录/GPU，最后预检并单次执行。** 原 200-update/36-slot 运行器不能直接执行本包，不绕过原限制。现在无需重复询问 02 是否同意按本冻结设计开始实现，也不提前申请尚未实现的运行授权。

## 本批实际验证

新增 [构造器](/home/a200/0proj/ITM/scripts/freeze_b0_correction_protocol.py) 与 [计划测试](/home/a200/0proj/ITM/tests/test_b0_correction_protocol.py)。6 项测试通过，验证包/来源 hash、原 val 地址、120 槽位与两臂对应、480 探针/噪声重建、600 exposures/200 updates 及逐条噪声、预算边界和 120 条审阅排列。将最终构造器运行到新的 `/tmp` 目录，**全部 11 个包文件逐字节相同**；没有模型计算。

```bash
PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/freeze_b0_correction_protocol.py
PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_correction_protocol.py
PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/freeze_b0_correction_protocol.py --output /tmp/itm_b0_correction_final_replay_20260926_v2
```

再次复现应改用尚不存在的 `--output` 目录；构造器拒绝覆盖已有包。旧 166 项方法回归及 B0 输出评价没有重跑。本批证据追加到公共 evidence index，不能当成新模型结果。
