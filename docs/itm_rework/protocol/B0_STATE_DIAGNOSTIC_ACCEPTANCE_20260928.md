# 02 → 01：保存状态 V 增量软件验收

**接受 `0b3642252dbb1c9295d9ee5e16a8eec726ec6fb8`，无新增返工项。** 02 独立执行相关联合回归 **135 passed、0 failed/error/skipped**（含29项新增V方法测试），独立重建 implementation 逐字节一致，1,656份绑定文件与全部输入/保存状态核验通过。**下一责任方为01：申请绑定下述最终身份、唯一新目录及GPU的新单次授权，获批后才预检和执行V。** 相同实现无需再次冻结或重复验收。

本轮没有真实预训练模型构造/前向、训练、采样、GPU查询/预检或实际授权创建。仅在CPU读取既有R checkpoint张量以检查来源、运行NumPy审计及微型CPU模型/函数替身测试。软件验收不代表144个真实GPU锚点已通过，也不代表模型有效性成立。

## 通过验收的身份

| 项目 | SHA256 |
| --- | --- |
| [V冻结包](b0_state_diagnostic_20260928_v1/package.json) | `6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012` |
| [最终implementation](../method/state_diagnostic_implementation_20260928_v1/implementation.json) | `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46` |
| 固定R final200 | `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642` |

implementation原文件位于 `outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1/implementation.json`，文档存档与原文件逐字节一致。02在新的 `/tmp/itm02_state_diagnostic_acceptance_20260928_v1/` 重建，目录仅有同字节的implementation.json，没有新参数或checkpoint。

1,656份绑定文件全部hash通过，包含全部1,640份冻结来源。旧权重、原初始化、final200训练来源、归一化、环境及保存状态身份仍绑定。审计过程对Module前向、CUDA设备API、预检/监督器及子进程设拒绝守卫，触发0次；环境比较只读软件/构建/后端配置，无设备预检。见[独立验收记录](b0_state_diagnostic_acceptance_20260928_v1/acceptance.json)。

## 独立输入与状态重放

02从原始NPZ直接用NumPy插值和SO(3)相对旋转参考重建6个来源的20Hz输入，独立核对12个source/config缓存的全部48套T/P/S/R输入。

- train `000027/000506/004728` 的398个唯一20Hz帧重新计算均值/标准差，与冻结统计一致；没有把配置重复或val混入统计。
- T/P保留原输入，T的IMU presence=false；S为raw acceleration=0和原始O[0]；R只反转`[0,T−1,…,1]`，时间戳不变。
- 各分支原始O[0]参照、caption、head/wrists mask、真实长度和frame mask通过核对。所有48套normalized input及reference的独立重建**最大差均为0**。
- 全48个保存状态的x_t、历史f10/f11、mask与冻结来源精确一致；target由原motion.npy及绑定的预训练均值/标准差独立重建，逐位一致。未重新加噪、对齐或改变目标。

输入与状态逐项结果写入 `acceptance.json`；没有仅依赖01的准备报告或同一变换函数自证。

## 新入口的审查与故障验证

阅读了新增 `mdm_state_diagnostic.py`、worker和bounded入口及调用的既有loader/限额实现；独立回归覆盖以下路径。

| 核验面 | 证据与结论 |
| --- | --- |
| 原始分支 | 每状态一次batch4，同一保存x_t/原t/真实长度，顺序T/P/S/R；没有CFG、q_sample、采样、decode或训练调用 |
| 条件缓存 | 每source/config一次batch1 caption；P/S/R各观测/参照编码，四个t共享；全队12 caption调用、72 IMU examples、48 MDM调用/192 examples；T严格null且不编码 |
| 数值路径 | 复用已修复非链式loader；全eval、denoiser/control与embedding float32，无AMP/梯度；保留上游CLIP原生half权重路径，没有额外精度转换 |
| 身份拒绝 | state、target、mask、历史预测、caption/reference、缓存输入/张量/模式/checkpoint发生变化时拒绝；manifest完整绑定新代码与依赖 |
| 软件锚点 | 48组P/f11、T/f10、S/T，共144项，atol=rtol=1e−6；记录bitwise/max/RMS/count/finite_count/pass，未改冻结容差 |
| 原始输出先保存 | 4分支有限预测、共享state/target/mask与元数据先于锚点判定落盘；锚点失败组保留已执行的原始预测并标为diagnostic_only |
| 计数 | reserved/actual/completed及对应calls分别持久化；先预留后调用，内部OOM/非有限/返回后超时不冒充完成；训练/denoising计数必须为0 |
| 整队停止 | 预检前登记48组/192分支/144checks；300秒绝对期限从预检前起，加载/编码/落盘/核验共用，不重置；单GPU、24GiB allocated上限，复用已验收监督器 |
| 故障保留 | 注入第二组锚点、NaN、OOM、写入、超时，第一组产物保留，当前组失败，后续not_run；父入口另覆盖预检/期限耗尽、receipt缺失、必要产物缺失/hash变化及计数异常，无重试 |
| 授权与交付 | 新授权必须绑定本包、最终manifest、唯一新目录、GPU及用户指令引用；正常worker预期684个产物，缺失/损坏导致父队列失败并标记相关组诊断用途 |

branch原始sidecar保守标记 `raw_saved_pending_anchors` / diagnostic_only，并指向final group JSON；最终group、receipt与父账本给出完成/失败状态。评价时按这一完成链解析，不能单凭原始sidecar声称通过。保留非完成分母；有限但质量差不会触发提前停止。

[02回归日志](b0_state_diagnostic_acceptance_20260928_v1/logs/regression.txt)：**135项通过，159.75秒**；28条warning为已有Transformer nested-tensor提示。包含29项V方法测试及9项V协议测试，其他为直接复用的alignment/限额/correction/loader/IMU control依赖；不把重叠测试重复计数。父入口fixture使用明确标记的临时假授权和预检/监督器替身，不启动真实子作业。实际worker通过微型模型走完48组与故障路径，不能据此保证真实GPU时间或数值。

复现脚本：[accept_b0_state_diagnostic.py](/home/a200/0proj/ITM/scripts/accept_b0_state_diagnostic.py)；[命令与边界](b0_state_diagnostic_acceptance_20260928_v1/commands.md)。02只新增验收代码及协议证据，没有修改01方法代码或历史产物。

## 下一步

01现在可申请新的单次授权：**48个保存状态、192分支、144锚点；72 IMU examples、12 caption调用；单GPU、24GiB allocated、整队最多300秒（预检前起）**。训练、checkpoint、完整采样、posterior steps和新噪声均为0；无warmup、重试、fallback或换卡。授权需绑定上述manifest/package以及实际唯一运行目录/GPU，旧授权不可复用。本次验收没有授予运行权限。

运行后02按已冻结evaluation独立计算192分支的1,728个块指标、192项差值及历史Q并列比较；实际144锚点失败时保持原阈值并保留失败。V只补同状态时间顺序诊断，不产生新动作；旧完整动作负例、3个已暴露来源/每split、单训练seed和相关状态限制保持不变，**联合控制有效性仍未认证**。
