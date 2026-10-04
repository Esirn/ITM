# 02 → 01：F 损失配重实现增量软件验收

**接受 `963a132` 的冻结 F 实现，无新增返工项。** 02 独立运行相关 CPU 联合回归，**155 passed、0 failed/error/skipped**（31 项既有 warning，157.77 秒）；1,145 份绑定文件核验通过，manifest 独立重建逐字节一致。此次通过的是软件与输入验收，尚无 F 真实训练或采样结果，模型有效性仍未认证。

**下一责任方为 01：使用下列最终 v2 manifest、唯一新目录和 GPU 身份申请新的单次运行授权。** 相同实现无需再次冻结或往返确认；获批后才预检、单次训练和采样，完整产物再交 02 按原冻结方案评价。本报告不是计算授权，不能复用任何旧授权。

## 接受的身份

| 对象 | 固定身份 |
| --- | --- |
| 被验收提交 | `963a132129b808b4c43af0c5bea30fc735ddcc67` |
| F package | `a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9` |
| 最终 implementation | `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30` |
| 原共同 initial.pt | `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401` |
| loss_version | `normalized-x0-root-ric-half-v1` |
| 完整 263 维权重 SHA256 | `ad96c3f5dc0eef83dc8b0107a4802c5af84badf383c12e00124dd63411d7f297` |

实际执行入口绑定的是 `outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json`；其字节与 [01 归档副本](../method/feature_balance_implementation_20261003_v1/implementation.json) 相同。归档目录名仍为 v1，不代表接受旧的 outputs v1 开发 manifest。最终 v2 已包括有限原始特征在解码前的保留逻辑。

## 独立证据与覆盖范围

[审计脚本](../../../scripts/accept_b0_feature_balance.py)、[复现命令](b0_feature_balance_acceptance_20261004_v1/commands.md)、[审计结果](b0_feature_balance_acceptance_20261004_v1/audit.json)、[审计日志](b0_feature_balance_acceptance_20261004_v1/audit.log)及[独立回归日志](b0_feature_balance_acceptance_20261004_v1/regression.txt)均保存于本批证据目录。

- 全部 1,145 份绑定文件 hash 通过，冻结包的 1,125 份来源完整包含且未改动；归档副本与独立目录重建结果均逐字节一致。准备目录只产生 implementation.json。
- 直接从原始 NPZ，以 NumPy 插值及 SO(3) 相对旋转插值重算 20 Hz 输入；三条训练动作唯一 398 帧重算的均值、标准差最大差均为 0。60 套采样输入及 6 套训练配置的加速度、姿态、时间、mask、原始根参考、caption/index、GT、标准化 motion 与 manifest 对齐；normalized IMU/reference 最大差均为 0。
- 原共同初始化逐张量身份一致。从已有张量核对训练参数集合与顺序：36 个张量，共 8,705,024 参数。准备过程禁止构造模型、模块前向、CUDA 查询/初始化、GPU 预检和子进程启动，触发数为 0；仅在 CPU 反序列化已有 checkpoint 张量，没有新初始化或训练权重。
- 独立重跑 49 项新增 F 方法测试、9 项 F 冻结协议测试及 97 项直接依赖测试，共 155 项。测试使用小型 CPU fixture；其中模拟授权、计时、故障、训练和采样均不是真实 F 作业或授权，不计为模型效果证据。

冻结时已重建的 600 条训练噪声及 12,000 条采样噪声证据继续有效：本次冻结包、计划与执行依赖来源身份均未变化，不重复算作新独立噪声实验。

## 实现审查结论

训练损失使用有效帧上的 `0.5*mean(e[:67]²)+0.5*mean(e[67:]²)`，完整 263 维保留；边界通道、真实长度/无效 padding、float32/64 解析梯度与有限差分均有 CPU 检查。统一权重退化检查覆盖原损失、梯度和 AdamW 更新；每次三条真实长度样本各除以 3，null 无梯度样本仍占分母。600 条 exposure 的 uniform/D/A/balanced 来自同一次预测并在反向前保存。

实际入口复用原加载器、batch、static-delta/dropout、训练随机地址及 accumulate_update；从共同 initial 加载、使用全新空 AdamW 状态，核对参数顺序、冻结 bias/backbone、native CLIP 精度与 eval 状态。仅 200 更新 final 可采样，50/100/150 快照不能替代或恢复。checkpoint 来源、完整权重/损失、initial、计划、实现、环境、参数和 optimizer 设置有拒绝测试；旧 R 改标签、错误/缺失身份、中间 checkpoint、冻结 bias 或 tensor hash 改动均拒绝。

完整采样保留单位 factorized batched CFG、原 1000 步 DDPM、60 槽位及历史噪声地址。七条件下 trace 开/关 CPU 对照的预测、随机计划/RNG、缓存和调用计数相同；240 trace 来自已有 batch4 前向，没有额外模型调用。48 项完整 motion 软件等式保持原 1e−5 米界；F/U IMU-only 不被错误纳入等式。相关测试使用历史数组及小型 fixture，真实 F 的 48 项结果仍需运行后核验。

调用观察 hook 不替换返回值，分别记录进入、完成和调用数；已进入但异常返回的前向不能计为完成。预检前登记 200 updates、600 exposures、60 samples、240 traces、48 checks。父进程和 worker 的绝对/共享截止时间、计数、产物/receipt/hash、授权身份、唯一目录与停止逻辑均通过 CPU 覆盖。

OOM、非有限值、训练/共享采样超时、写入失败、缺失 receipt/artifact/trace、错误计数/checkpoint 和软件等式失败均保留失败记录及计划分母。有限完整特征先保存到 raw_samples，再核对噪声和解码；注入解码失败时 1000 步原始特征和四条 trace 仍在，最终样本不被接纳。有限 motion 在软件等式判定前已保存，失败时标记 diagnostic_only；不以动作质量差为由删除或补采。

## 交给 01 的下一步

按 [原冻结预算](B0_FEATURE_BALANCE_FREEZE_20261003.md)申请：200 updates/600 exposures，60 条完整采样和 240 条旁路 trace；训练自预检前 t0 起最多 600 秒，采样共享最多 3,000 秒并受 t0＋3,600 秒总期限限制，单 GPU、24 GiB allocated。总上限为 MDM 60,600 次调用/240,600 examples，encoder 1,320 examples、caption 660 次。首个规定故障即停，无重试、续训、换卡或补采。

授权必须绑定本报告的 package/implementation SHA、明确用户授权依据、唯一新目录和 GPU 身份；当前未创建真实授权或执行目录。本轮无真实模型前向、训练、采样、GPU 查询或预检。

实际 GPU 行为、真实运行完整性与模型有效性尚未验证。既有负结果及主弓步行为/五项数值门槛保持不变；运行后仍须全量 F/U 配对评价，不能把本次 CPU 通过或未来运行完成视为修复成功。
