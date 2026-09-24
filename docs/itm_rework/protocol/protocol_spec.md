# P/C/D 与统计接口 v0.1（草案）

只覆盖完整片段离线生成，head=[4]、wrists=[0,1]。测试阈值、相容 manifest、B0 和采样预算未冻结。
`thresholds.json` 刻意保留 draft/null；正式 evaluator 会拒绝它，避免临时选阈值产出好看的成功率。

## 条件与准入

P：同一原始片段的文字与 IMU。必须比较 Text-only、真正 IMU-only、paired、同类近长度 shuffled、完全 bypass、数值零与物理静止。
Text-only、bypass、零值和静止不可互换。物理静止在线加速度定义下是零加速度及合法常量姿态，归一化后一般不等于数值零。
固定时间评价不允许 DTW、逐帧旋转或把长 IMU 拉伸成短片段。恢复 MPJPE 仅用于配对协议，并公开对齐。

C：固定观测，基础/编辑文字成对比较。manifest 必须在生成前给出 compatible 标签、独立 witness、观测 residual、质量复核、容差和信息增量角色。
“同一动作的原 caption”或“改不同部位”都不自动证明互补编辑。有限动作库找不到 witness 不判 incompatible。
旧 30-motion walking 编辑统一保留为探索性，不能导入 compatible 表。

D：incompatible 和 uncertain 分别记录，不进入 C 分母；输出各条件违约、连续质量和 guidance 关系。
评价器对 D 不输出联合成功率，逐样本 joint_success 为 null。

`compatibility.schema.json` 定义交接字段；`compatibility_manifest.jsonl` 当前为空。需要 01/02 验证的运动见证或独立标注后才能填入。
同一 witness 与 source 的来源组不能跨 train/val/test。没有完整物理闭环时即使构造 joint-proxy witness，也必须明确其定义与局限。

## 逐样本与联合指标

`result.schema.json` 是需要归档的字段清单。`evaluate_rework_records.py` 检查必填字段并聚合已经独立测得的 semantic、observation、quality scores。
该工具本身不是语义或自然度判定器；Phi 若参与训练，不得成为唯一评价依据。需要独立属性判据/盲评及明确 evaluator 版本。

每次调用限定一个 model/checkpoint、配置、条件和协议；expected manifest 在输出前建立。
缺文件、NaN、运行失败留在分母；unexpected ID、重复 ID、哈希/随机计划/阈值版本不匹配直接报错。
语义编辑既要满足 score，也必须在预期方向达到正的最小幅度。所有 observation_max 通道均须通过；质量门槛只排除明确无效输出。
只有 joint-proxy 获得声明，因此指标强制命名 `JSR_joint_proxy`；`JSR_IMU` 当前不被接受。
原始连续 metrics、失败原因、输入 hashes 和各分解成功率一起保存。模型失败不用于重标相容性。

`valid_pair_diversity` 同时报告有效比例、有效对差异、所有有限输出差异、非有限比例。少于两条有效输出为 not_estimable。
这是调用者声明的 feature-space 距离，不叫严格零空间多样性。尚未在真实新协议输出上估计。
jerk 必须同时给绝对值、幅度、足滑/盲评；旧 jerk ratio 的 epsilon 仅保留在历史复算，不能用低 jerk 自动判优。
阈值敏感性与语义—观测 Pareto 分析等待已校准阈值及新输出；当前不编造数值。

## 统计与随机数

先平均同一 case 的 seed/caption 重复，再平均同来源组的 case，最后 bootstrap 来源组。
该实现估计等权来源组均值，与旧 motion 等权均值不是完全相同的估计量。
成对比较先构造每样本 paired-reference 差值，再按相同来源组重采样；本轮 2000 次，seed 20260924。
100 motions 对应 99 个来源组；30-motion 多 seed 对应 30 个来源组。
文本编辑的 head error 先把每 motion 的五个编辑求平均，再按 30 个来源组 bootstrap，避免将 150 次编辑作为独立动作。
CI 仅覆盖动作/采样差异，不覆盖训练方差。历史 74/100 不能证明总体改善率超过 70%。

`src/itm/data/random_plan.py` 用 SHA-256 寻址 PCG64 的每一步噪声，地址包括 backbone、case、sample_seed、step、shape。
step=-1 为初始噪声，后续每步独立寻址；额外分支随机调用不会改变目标步的噪声。
同 backbone 同条件长度使用相同地址，不把 model/condition/sensor 配置塞入地址；不同 backbone 不宣称逐元素共享噪声。
返回数组 hash、NumPy 版本和 dtype；需要 01 把它接入 sampler 后才有实测同流保证。

## 运行阶梯

`plan_rework_validation.py` 已选 6 个 val 来源组：短、中、长各 2；每条计划 head/wrists。
ID 固定在 `validation_20260924_v1/panel.jsonl`，但输入同步、非对称腕部覆盖、相容属性仍待验收。
P 的 7 条件 × 2 配置 × 6 cases × K=1 = 每模型 84 个**建议输出**；并未授权或启动这批生成。
guidance、采样器、steps、forward 次数、显存/GPU、停止条件未定前不启动。

`run_rework_stage.py` 是串行 CPU 阶段运行器，默认 dry-run；只允许明确列出的协议脚本，使用新 run 目录、日志、退出码和 timeout。
它拒绝 GPU job。修复后模型的 GPU 训练/采样由 01 的入口执行，02 的队列统一登记，不能绕过预算或同时占用他人 GPU。
当前 `preflight_20260924_v1/run.json` 正确报告 C 被 B0、witness 和 thresholds 阻塞（exit 2），不是执行成功。

G4 还须冻结协议 commit、checkpoint、guidance、评价器、阈值和预训练可见范围，并建立独立来源测试池。
历史测试曝光不能通过换 seed、重命名或不再看标签恢复独立性。
