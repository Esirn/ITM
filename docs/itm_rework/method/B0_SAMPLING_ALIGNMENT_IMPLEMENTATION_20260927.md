# 01 → 02：固定 R guidance 修正的实现与 CPU 验证

承接 `1680618`，实现固定 R final200 的 Q/S1 入口，等待 02 独立软件验收。仅新增方法实现，不改冻结协议、历史实现或旧结果。模型有效性尚未验证；原步行、拍手下肢退化及训练动作生成失败继续保留。

## 绑定与准备

| 项目 | 身份 |
| --- | --- |
| 冻结 package | `537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60` |
| 唯一 R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |
| 历史共同初始化 | `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401` |
| 新 implementation | `793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4` |

准备目录：`outputs/itm_rework/method/sampling_alignment_implementation_20260927_v1/`。
[manifest 存档](sampling_alignment_implementation_20260927_v1/implementation.json) 与该目录的 `implementation.json` 逐字节相同。绑定 **743 份文件**，包含全部 **726 份冻结来源**、当前方法入口及其依赖；保存 **60 条 S1、480 条 Q 输入**的数组身份。固定 checkpoint、原训练归一化、来源和环境均校验。CPU 准备只读取已有 R checkpoint，不构造预训练模型、不初始化新参数、不保存新 checkpoint。
[准备核验记录](sampling_alignment_implementation_20260927_v1/preparation_validation.json) 保存绝对入口与 hash。准备资产没有授权记录、运行目录或 GPU 执行许可。

## 实现行为

- [`mdm_sampling_alignment.py`](/home/a200/0proj/ITM/src/itm/models/mdm_sampling_alignment.py)：严格 package/checkpoint/输入验证、浮点比较、Q 分解、运行计数及旁路 trace。读取 R final200 并验证原共同初始化、200 次更新、600 exposures、配置、归一化和环境，拒绝兼容 schema 下的错误身份。
- [`run_mdm_sampling_alignment_worker.py`](/home/a200/0proj/ITM/scripts/run_mdm_sampling_alignment_worker.py)：Q 每组一次 batch5，共 96 组；严格 null 的前两分支不调用 encoder，后三分支各作观测／原 O[0] 参考编码。复制同一真实长度 x_t，无 padding。输出 480 份预测及 288 条分解、384 条组内比较、192 条差值之差；负交叉项不截断，有限但差的 Q 不改变 S1。
- S1 复用原四分支 batched CFG、1000 步 DDPM 和旧完整噪声地址，仅将文本权重从 2.5 改为 1。条件缓存、实际输入、O[0]、caption、真实长度和历史对照身份校验；不使用直接 f11 快路径。输出兼容旧 R 的数组，并显式保存新 case/arm/guidance、comparison_case_id、输入与旧产物引用。
- 每条采样在实际 900/500/250/50 步截取已有 batch4 输出及同次返回结果，保存独立 float32 快照、frame mask、分支 flags、scaled t、输入／噪声／模型身份和返回 hash。hook 不替换张量、不调用模型、不改 RNG 或 cache。raw NPZ 先落盘，再记录软件检查，检查失败仍保留有限数组。保留的分支可离线重组两种权重；记录文本增量和概念组合差，仅解释为 S1 同状态局部对照。
- [`run_mdm_sampling_alignment_bounded.py`](/home/a200/0proj/ITM/scripts/run_mdm_sampling_alignment_bounded.py)：预检前登记 480/60/240 分母，验证 manifest 后执行固定队列。Q 600 秒，60 条 S1 共享 2400 秒，整队从预检前最多 3600 秒；单 GPU、allocated 最多 24 GiB。最多 240480 MDM branch examples／696 encoder examples，0 updates。计数前置保留，加载、保存和检查计入时间。
- 超时、OOM、非有限值、身份／计数／软件一致性失败即停止；保存完成、失败、未运行状态，不重试、不换卡、不回退。缺 trace 的 motion 即使存在也标记诊断不完整。新授权必须绑定本 manifest、package、唯一运行目录和 GPU；旧授权不能复用。

全部容差照冻结包：Q static/text、缺失条件分支、RIC CPU 解码、float32 组合的 gamma16 逐元素界和 float64 Q 恒等式界。24 项完整采样检查分别是 6 text/bypass、12 static/text、6 新旧 IMU-only，保存 max/RMS/bitwise/count；缺对照记 not_comparable，不能补造通过。逐来源汇总与模型效果评价由 02 按冻结 evaluation 执行。

## CPU 验证与边界

联合 **258 passed / 0 failed / 0 skipped**（179.61 秒），其中新增方法测试 **33 项**。最终结果见 [联合回归日志](sampling_alignment_implementation_20260927_v1/logs/regression.txt) 与 [JUnit](sampling_alignment_implementation_20260927_v1/logs/regression.xml)。新增方法测试位于 [`test_mdm_sampling_alignment.py`](/home/a200/0proj/ITM/tests/test_mdm_sampling_alignment.py)。

覆盖真实新加载入口及上游 `_apply/train` 不返回 self 的语义、错误 R 身份拒绝、96 组 Q 的实际 branch/gate/shared-x_t、两种权重的实际组合函数、六种条件的完整 1000 步 trace 开关逐位一致、RNG/cache/counter 一致、完整 sample worker 与 CPU 解码，以及全部 24 项比较和缺失对照。

故障覆盖 Q 静止一致性、组合误差、trace 写入、预检失败、Q 超时、采样 OOM、trace 缺失、完整输出一致性、跨槽位期限、非有限值、额外调用和禁止更新；检查已完成产物保留及失败分母。队列测试替换 preflight 和子进程为明确标注的 CPU fixture；没有创建真实授权。1000 步 fixture 的高频 receipt 持久化用内存替身加速，保留实际 guard 计数／限额；真实持久化与队列失败账本另外覆盖。

测试使用小型 CPU 模型／diffusion fixture，实际调用本实现、原 sampler 与 guidance 接口；加载测试替换外部模型工厂并执行上游方法语义。**这些检查不是预训练 MDM 前向、GPU 性能测试或模型效果证据。** 本轮真实训练、预训练模型前向、扩散生成、GPU 查询／预检均为 0。

## 下一步

由 **02 独立软件验收当前代码及 manifest**，重点复核冻结分支、trace 零额外调用、身份／输入绑定及故障保留。通过后 01 再申请绑定最终实现、唯一新目录与 GPU 的单次授权；获新授权后才预检执行。当前不申请或消耗运行预算，不扩大数据准入，不宣称修正有效。

复现命令与测试边界见 [commands.md](sampling_alignment_implementation_20260927_v1/commands.md)。
