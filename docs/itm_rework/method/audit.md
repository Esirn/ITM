# 01 方法正确性审计（2026-09-24）

读取 commit：`663414f1d9bffd6a70189846739bc9d9b1d0d873`，启动时分支 master。
初始工作树已有 02 未提交的数据/协议代码，本会话保留它们，未编辑 contract、protocol、data、metrics 或论文。
契约：`itm-rework-v0.1-draft`，未冻结。本会话变更 commit 以方法提交和 STATUS 为准。
`src/itm/backbones/mdm.py` 的缓存交接属于实际采样调用链，本次明确由 01 修改。

## G0：历史问题与当前状态

| 项目 | 开始时核查 | 本次修复/证据 | 仍不能推断的结论 |
|---|---|---|---|
| 双腕身份 | `make_imu_control_encoder` 共享线性投影+身份加法后均值，对归一化输入互换不变 | 保留 legacy_mean；ordered_concat 在融合前清零无效槽位，再拼显式 mask 经 MLP；性质与学习测试 | 未见动作生成的左右归属提升未评估 |
| null bias | 每层 Linear bias 在 encoder 置零后仍生效 | strict 对完整残差门控；包含每样本 presence、frame mask；首个 text/time token 不受残差作用 | 不把一次 checkpoint 审计外推到全部旧结果 |
| 四分支 CFG | 参考版本已有 factorized 公式，非新贡献；null 实现不严格 | 顺序/批量/缓存/非缓存一致性与边界恒等式测试 | 四效应分解不证明统计独立或物理解耦 |
| 时间位置 | IMU encoder 无位置；MDM 有位置编码 | physical_sinusoidal 使用秒输入，支持 none 消融 | 条件对时长变化敏感不等于正确跟随 |
| padding | IMU encoder 有 key padding mask，但全空序列和非法 padding 不安全；MDM 默认 mask_frames=False | B0 清理无效输入/输出和全空 attention；完整采样每条实际长度执行 | 保留原始 MDM padding 行为，没有改变骨干来掩盖来源 |
| 临时缓存 | 异常可能残留 control | 消费前清除，顶层 always_call hook 清理；缓存携带 presence | 同一个模型实例仍不可用于并发重入 forward |
| 冻结状态 | requires_grad=False 但 model.train() 开启骨干 dropout | frozen_eval；只训练 encoder/adapters；teacher 保存/恢复所有模块 mode | legacy_train 明确保留用于诊断，不能与 frozen_eval 混写 |
| teacher | 历史无缓存 anchor 可 bypass；teacher 后 model.train() 改变混合模式；有缓存时未剔除 control | anchor 去除缓存、重编码完整文本，模式逐模块恢复 | absolute anchor 不证明保持语义编辑能力 |
| 采样噪声 | 仅初始噪声共享；DDPM 后续使用全局 RNG | 新 addressed sampler 每条/每步消费 02 PCG64 随机计划，存所有地址/hash | 改变 sampler/长度/骨干需新计划；不跨未定义 latent 共享 |
| 时间对齐 | `_load_case` 将 donor 全段拉伸到目标长度 | 新默认 crop_seconds，短 donor 拒绝；legacy_stretch 显式保留 | HumanML 与 IMU 坐标/一帧偏移仍待 02 认证；裁切不等于同步证明 |
| 辅助目标 | global joint trajectory 与 root-relative 指标不同；velocity/jerk 未除 dt | 显式 trajectory-space；seconds/legacy_frame；新默认 diffusion-only，velocity/jerk 权重 0 | 这些都是 paired joint proxy，不是测量一致性 |
| 训练续接 | 保存 optimizer 但恢复只读权重、归一化 | 显式记录 fresh_optimizer；配置不匹配拒绝；新输出不覆盖旧 checkpoint | 不是精确恢复 RNG/optimizer 的断点续训 |

## 真实 checkpoint 审计

脚本：`scripts/audit_mdm_method.py`；原始 MDM 的实际调用链为
`create_model_and_diffusion → model.mdm.MDM → seqTransEncoder`；采样后验使用已安装 runtime 的
`SpacedDiffusion.p_mean_variance`，不凭文档猜测函数。

Stage-2b：`outputs/mdm_control/stage2b_balanced_b.pt`，SHA256
`5d0800e61994c79f862378871e2c149fde052c7419553939a06a6a6632788a98`。
原始 MDM SHA256 `ce43c1d4232fc2b897999d7996c2d2b889791d99585d87e1289af773ab56ab93`。
PyTorch 2.5.1、NumPy 2.2.5、CPU float32、2 threads；种子 20260924；文本 a person walks。
固定 12 帧合成诊断输入、t=500。容差在断言前固定 atol=1e-6, rtol=1e-5。
以下数字处于**归一化 HumanML263 空间，不是米或任务指标**。

| 相对原始 MDM | 单步最大绝对差 | 单步相对 L2 | 完整 1000-step 最大绝对差 | 完整相对 L2 |
|---|---:|---:|---:|---:|
| 完全 bypass | 0 | 0 | 原始参照 | 原始参照 |
| legacy imu_uncond | 0.301335 | 0.088912 | 3.204223 | 0.551077 |
| strict imu_uncond（同一旧权重，显式改门控） | 0 | 0 | 0 | 0 |
| 有效 mask + 数值零 IMU | 0.633355 | 0.201520 | 未运行 | 未运行 |

8 层 adapter bias norm 为 0.1467–0.3386。归一化双腕交换 encoder 最大差 5.96e-7，符合浮点近似下不变。
原始 bypass 从 12 帧补到 16 帧时，有效帧输出最大差 1.442799，证明 padding 依赖来自骨干。
物理静止未运行：等待 02 的物理输入 fixture；绝不把零矩阵当静止旋转。
完整采样只有这一个短合成案例、三个流，不能据此给历史任务指标提供校正量。
原始预测、三组完整输出和全部噪声地址保存在 `checkpoint_audit_20260924_v1.{json,npz}`。

## 有限学习与真实入口 smoke

`identity_overfit_20260924_v1.json`：同一合成非对称双腕集，2 variants × 120 CPU updates，8 samples × 4 frames。
目标为有符号的左/右幅度，输入在归一化空间严格交换；legacy MSE=1.20000076，ordered_concat MSE=0.00040242。
这是可辨識性与短程学习证据，readout 是诊断头，不是最终生成器，也不是完整生成的对照结果。

真实 MDM 上额外做了一个 train-only record `000004` × head/wrists × 1 update 的 B0 diffusion-only smoke，
CPU、训练 seed 1234。总/diffusion loss=0.030115（日志六位小数）；checkpoint 标记 `uncertified_diagnostic`。
没有使用验证/测试集选参，没有运行 GPU 或扩大训练。
采样 smoke 的首次 spec 误用无标准 IMU 的 `000001`，在采样前失败，记录保留于 `b0_sample_failure_v1.json`；
随后改成训练 smoke 实际使用的 `000004`，不选择任何好看的输出。

## 历史结论影响

旧 “head-null=wrists-null” 只能说明两种 mask 的 null 一致，不能说明等于原始 MDM。
旧 test100、multiseed30、text-attribute、guidance sweep 中通过旧 CFG null 定义的 Text-only 对比需要明确旧 null 语义。
已存在的 paired-vs-shuffled 数字不因发现可达 bias 路径自动失效；其来源重叠、随机流和时间拉伸风险沿用 02 审计。
新 strict 门控会改变旧权重的生成，必须标成修复消融，不得声称旧方法原样复现。
旧“平滑”损失实际上是平方的 `relu(||Δ³pred|| - ||Δ³GT||)`，只惩罚超过 GT 的部分。
本轮没有证明 B/C 性能收益、物理一致性或同一最终模型联合能力。
