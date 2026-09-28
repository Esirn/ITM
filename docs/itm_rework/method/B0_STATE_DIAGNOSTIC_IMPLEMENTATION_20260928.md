# 01 → 02：保存状态 V 实现与 CPU 增量验证

承接 `c0fc63f`，已实现固定 R 保存状态重放入口，等待 02 增量软件验收。旧方法入口、协议、权重、数据和历史产物未改动；本轮没有真实预训练模型前向、训练、采样或 GPU 查询/预检，没有创建新运行授权。

## 固定身份与准备资产

| 项目 | SHA256 |
| --- | --- |
| V 冻结包 | `6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012` |
| R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |
| V implementation | `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46` |

准备目录：`outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1/`。
[manifest 存档](state_diagnostic_implementation_20260928_v1/implementation.json) 与原文件逐字节相同，绑定 **1656 份文件**，包含全部 **1640 份冻结来源**和新方法依赖；保存12个cache／48套分支输入／48个状态的数组描述。准备只读取已有 R checkpoint 张量到CPU并校验原身份，没有构造预训练模型、初始化新参数或保存新checkpoint。详见[准备核验](state_diagnostic_implementation_20260928_v1/preparation_validation.json)。

## 新实现

- [`mdm_state_diagnostic.py`](/home/a200/0proj/ITM/src/itm/models/mdm_state_diagnostic.py)：核对 package、R final200、原初始化/训练来源/统计/环境、全部保存 x_t/target/mask/历史f10/f11。由原数据管线重建 T/P/S/R，与冻结输入描述和已绑定旧样本重建值校验；O[0]、参考、caption、原时间戳保持固定。
- [`run_mdm_state_diagnostic_worker.py`](/home/a200/0proj/ITM/scripts/run_mdm_state_diagnostic_worker.py)：复用已修复的非链式 loader，每组一次原始 batch4，顺序 T/P/S/R，复制保存的同一x_t，无padding。每source/config按旧路径batch1编码caption一次；P/S/R各观测/参考两次编码，跨四个t复用；合计48 MDM调用/192分支、12 caption调用、72 IMU encoder examples。
- 缓存身份绑定实际输入/参考/mask/time、caption、来源/config、checkpoint、归一化、dtype/eval和已缓存张量hash；输入描述与实际数组不一致、缓存张量被改或模型退出eval均拒绝。T保留paired审计输入，presence=false且控制张量严格为0，不经过IMU encoder。
- 保留上游CLIP的既有原生half权重和float32 embedding返回；denoiser/control、输入/预测保持float32。没有把CLIP改成新的精度路径；无AMP、梯度或新的推理模式。
- [`run_mdm_state_diagnostic_bounded.py`](/home/a200/0proj/ITM/scripts/run_mdm_state_diagnostic_bounded.py)：独立CPU准备与授权执行入口。预检前写48组/192分支/144 checks及零调用账本；从预检前起一个300秒deadline涵盖全部加载、编码、保存和检查。指定单GPU、allocated≤24GiB；复用已验收进程组监督和首次故障停止逻辑。
- 新授权必须绑定最终manifest、本V包、唯一运行目录、GPU和用户指令；旧授权不能使用。入口不调用CFG、q_sample、噪声生成、posterior/sampler、motion decode或训练，不提供warmup、重试、换卡或CPU fallback。

## 输出和失败记录

worker输出 `V/`：`inputs/Cxx-{T,P,S,R}.npz`及cache JSON、`states/Vxxx.npz`、`branches/Vxxx-{T,P,S,R}.npz/.json`、`checks/<check_id>.json`和`groups/Vxxx.json`。state文件独立保存精确x_t、target及mask；每分支独立保存float32预测，元数据绑定源trace/目标/原样本、原噪声计划血缘、输入/缓存、checkpoint/package/implementation及计数。正常完成共684份worker产物。

**原始数组和分支元数据先于144项锚点判定落盘。** 分支原始sidecar状态为 `raw_saved_pending_anchors`，保守标记diagnostic_only，并显式指向最终group JSON；最终group JSON及receipt/父账本给出completed/failed和最终diagnostic_only。软件锚点失败的四分支有限预测仍保留，不能把实际输出改成未执行。写入中断时已成功写入的文件和partial文件保留；缺失或hash异常的必要产物导致父队列失败，并将相关组/分支标记诊断用途。

锚点分别为P对原trace f11、T对原f10、S对同组T，固定atol=rtol=1e−6；同时保存bitwise、max_abs、RMS、count、finite_count和pass。有限但质量差不提前停止。本入口交付原始数组与软件锚点，02按冻结evaluation计算1728个分支块、192项差值及历史Q并列评价。

每类调用保存 `reserved/actual/completed` 和对应calls：reserved在调用前持久化，actual表示已进入调用，completed表示返回并通过有限值/期限检查。函数内部OOM/非有限值或返回后超时保留actual但不增加completed。优化更新和denoising计数必须为0；允许的最终上限为MDM192/48 calls、encoder72/72 calls、caption12/12 calls。错误身份、计数、cache或软件锚点，以及超时/OOM/非有限/写入错误均停止整队，保留完整计划分母。

## CPU 验证证据及边界

增量联合回归 **135 passed / 0 failed / 0 skipped**，159.19秒，含 **29项新增V方法测试**；28条warning为既有Transformer nested-tensor提示。最终计数见[增量回归日志](state_diagnostic_implementation_20260928_v1/logs/regression.txt)与[JUnit](state_diagnostic_implementation_20260928_v1/logs/regression.xml)。新增测试位于 [`test_mdm_state_diagnostic.py`](/home/a200/0proj/ITM/tests/test_mdm_state_diagnostic.py)。

覆盖实际worker的全部48组、同状态batch4、精确t/shape/gates、T严格null、12/72缓存及144锚点；错误state/target/reference/历史预测/caption/cache/checkpoint拒绝；调用前预留后超时不产生实际调用；禁止训练/denoising/额外调用；注入第二组锚点、非有限、OOM、写入、超时并核对第一组保留、失败和后续未运行。原始有限预测在锚点失败前落盘已核验。

父队列覆盖成功、预检失败、共享绝对超时、receipt缺失、必要产物缺失/hash变化、计数异常及预检耗尽期限，检查首次停止、原始完整分母与无重试。真实新入口调用的既有loader用上游实际 `_apply/train` 方法和微型外部模型工厂验证，保持不返回self的语义；fixture包含CLIP half参数兼容性。准备测试禁止神经模块前向、CUDA查询、preflight和子进程执行。

这些是CPU微型模型/函数替身；父队列fixture用明确标记的临时文件替换真实预检/子进程，没有创建实际授权。V worker的计数和receipt持久化使用真实实现。它们不能证明真实GPU耗时、144锚点的GPU数值通过或模型有效性；未运行V，未重新计算完整动作评价。

## 下一责任方

**02对新入口、manifest与上述CPU证据做增量软件验收。** 通过后01再申请新单次授权，获批才预检并执行V；本轮不申请提前授权，不复用已消耗预算。V只补齐同状态诊断，联合控制有效性仍未认证，所有旧负例及开发面板解释边界保留。

复现入口：[commands.md](state_diagnostic_implementation_20260928_v1/commands.md)。
