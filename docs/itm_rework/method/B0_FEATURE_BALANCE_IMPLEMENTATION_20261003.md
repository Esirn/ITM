# 01 → 02：F 半权重损失实现与 CPU 验证完成

按 `e214014` 冻结包完成 F 新入口，**155 项联合回归通过，无失败/跳过**，包含49项新增方法测试、9项F协议测试及97项直接依赖测试。下一责任方为 **02 增量软件验收**；验收后再申请新单次运行授权。本轮没有真实预训练模型构造/前向、GPU查询/预检、真实训练或扩散生成。CPU故障测试使用显式微型模型/假监督器，不能据此声称真实运行和模型效果已验证。

## 交付身份

| 项目 | 绑定 |
| --- | --- |
| F package SHA256 | `a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9` |
| implementation SHA256 | `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30` |
| 原 initial.pt SHA256 | `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401` |
| 完整263维权重 SHA256 | `ad96c3f5dc0eef83dc8b0107a4802c5af84badf383c12e00124dd63411d7f297` |
| loss_version | `normalized-x0-root-ric-half-v1` |
| 新 checkpoint schema | `itm-feature-balance-control-v1` |

原manifest为 `outputs/itm_rework/method/feature_balance_implementation_20261003_v2/implementation.json`；[文档副本](feature_balance_implementation_20261003_v1/implementation.json)逐字节一致。准备目录只含该manifest，无新权重或checkpoint。`outputs/.../feature_balance_implementation_20261003_v1`是补充原始特征保留之前的CPU开发准备，未授权、不得用于运行；本交付只绑定v2。没有创建实际授权记录或授权运行目录，GPU身份尚未申请绑定。

## 实现与唯一干预

- [mdm_feature_balance.py](/home/a200/0proj/ITM/src/itm/models/mdm_feature_balance.py)：只在有效帧计算 `0.5*mean(error[0:67]^2)+0.5*mean(error[67:263]^2)`；同一次预测记录uniform、D67、A196、balanced。固定分区/权重与包身份，生产CLI没有调系数、增益或guidance搜索选项。
- [worker](/home/a200/0proj/ITM/scripts/run_mdm_feature_balance_worker.py)：复用原loader、输入装配、addressed noise、static-delta、匹配encoder dropout、真实长度三条loss/3累加及原AdamW。读取原initial而非从R续训；参数集合与顺序独立检查为36个张量、8,705,024个可训练参数。bias冻结、原生CLIP精度和backbone eval不变；训练前后核对冻结参数hash。无AMP、梯度裁剪、scheduler或额外诊断前向。
- [bounded入口](/home/a200/0proj/ITM/scripts/run_mdm_feature_balance_bounded.py)：CPU不可变准备、新授权校验、父监督器、跨阶段账本和48项完整motion软件等式。真实执行没有重试、恢复、换卡或CPU回退入口。

旧方法代码、1125份冻结来源及02协议文件未修改。旧R与S1权重/输出仍是历史对照，F只采样本次新训练且来源完整的final200。checkpoint和样本/trace绑定新package/implementation、loss及完整权重/hash、原初始化/计划、参数审计、归一化/输入来源、环境、optimizer与执行记录；同schema内中间快照也不准采样。已覆盖旧R重贴标签、错误loss/权重、初始化、计划、环境、参数顺序、optimizer、冻结bias和tensor hash的拒绝路径。

每条exposure的有限损失诊断在backward前保存；每次成功更新另存记录。50/100/150快照保留，只有200完成才生成可准入final。未满200不自动恢复或替代权重。新增forward计数hook只观察同一次调用，返回None、不替换tensor；MDM/encoder/caption记录reserved、actual、completed与调用数，失败进入但未正常返回的调用不算完成。

## 完整采样与冻结边界

采样使用原完整DDPM路径，unit CFG `(1,1,1)`、60槽位与完整初始/逐步noise计划不变。240 trace均从同次batch4获取，额外调用0，保留原float32组合界、缺席分支等式和CPU RIC解码检查。trace开启/关闭的CPU夹具覆盖paired、text、IMU-only、static、bypass、numeric-zero、tail-reversed，验证输出/完整噪声记录/RNG/cache/计数不变。

48项motion软件检查严格读取冻结表，保留max/RMS/bitwise/count/finite_count及1e−5米界；仅F static/text、F bypass/text和F/U同条件null比较。**不套用旧guidance实验的F/U IMU-only相等假设**。每条完整采样返回后，在噪声身份与解码检查前先保存`raw_samples/<case>.npz`中的有限原始特征；即使解码检查失败也保留为diagnostic-only，只有最终sample/receipt可以判为完成。有限原始动作已保存后才做完整动作等式，失败保留动作并标记诊断用途；不因动作语义差而提前剔除槽位。

效果评价仍归02：按实际GT裁片核对000506的初始姿态恢复、换侧、中段回收与末端下沉；**不强求末尾站直**。原零退化幅度加浮点松弛、uncertain/missing、其他负例与全部分母保持，01未调整评价门槛。

## 限额与故障保留

预检前登记200更新、600呈现、60采样、240trace、48检查。训练deadline=t0+600秒；采样共享deadline=min(采样起点+3000秒,t0+3600秒)，t0在预检前；不转移训练剩余时间。单GPU隔离、allocated最多24GiB，父监督器复用进程组超时停止。

全队上限：MDM60600调用/240600 examples、encoder1320 examples、caption660调用；训练200updates/600呈现，采样60×1000步，240trace额外调用0。每阶段receipt和全队计数均保留，缺失/损坏产物、错误计数和不完整trace不会冒充完成。

测试注入预检失败、训练/共享采样超时、OOM、非有限值、写入失败、receipt缺失、产物缺失/hash错误、计数异常、trace缺失、无效checkpoint以及软件等式失败。另注入解码检查失败，确认1000步的有限原始特征及全部4个trace保留、最终sample不冒充完成。验证已完成产物保留、完整ID/分母、后续not_run、唯一目录拒绝再次执行与授权身份拒绝。真实GPU时延、144旧V锚点或模型效果不在本次CPU验收结果之内；F协议对应的是48项新完整动作等式。

## CPU 验证证据

[联合回归日志](feature_balance_implementation_20261003_v1/logs/regression.txt)：**155 passed，191.77秒**；31条已有Transformer/依赖warning，无失败/跳过。新方法49项测试位于 [test_mdm_feature_balance.py](/home/a200/0proj/ITM/tests/test_mdm_feature_balance.py)。

- 独立NumPy标量/解析梯度：104/136/158真实长度、67/196边界、无效padding、float32/64、零有效帧拒绝；选定坐标有限差分验证。
- uniform诊断损失与旧scalar/gradient/AdamW更新一致；缺席无梯度项仍占三分之一，梯度不重新归一化。
- 实际worker使用微型CPU模型走完200更新/600呈现和1000步采样；核对冻结bias、参数顺序、同预测四损失、no-grad跳过、600随机身份、4个快照、trace及解码。没有把微型夹具当真实训练。
- 实际父入口采用明确CPU假授权/预检/监督器替身走完完整队列，并注入上述失败；48项motion等式另用旧S1数组验证，特意改变IMU-only仍不触发错误的软件相等要求。

[准备核验](feature_balance_implementation_20261003_v1/preparation_validation.json)：1,145份文件绑定通过（包含全部1,125份冻结来源），60采样输入/6训练输入核对；398个唯一train帧的加速度均值/标准差独立重算最大差均为0。只在CPU读取既有initial及历史R的权重张量核对，不新初始化控制器或构造预训练模型。

[独立重建](feature_balance_implementation_20261003_v1/rebuild_validation.json)：manifest在另一临时目录重建逐字节一致。准备脚本将nn.Module构造/前向、CUDA设备API、预检/监督器设为拒绝，触发0次；只读软件环境属性不等于GPU预检。

复现CPU回归：

```bash
bash docs/itm_rework/method/feature_balance_implementation_20261003_v1/regression.sh
```

复现准备（输出必须为不存在的新目录）：

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python   docs/itm_rework/method/feature_balance_implementation_20261003_v1/prepare_cpu.py   --output /tmp/itm01_F_review_prepare
```

## 交给02

请对新损失/参数来源、checkpoint准入、真实入口及故障保留做增量软件验收，原数据与不变组件无需重新冻结。若通过，01再申请绑定本manifest、新唯一目录、指定GPU的单次预算：**200 updates＋60完整采样＋240零额外调用trace，单GPU/24GiB/整队最多1小时**。当前预算仍未授权，历史授权不可复用；F效果和B0有效性均未认证。
