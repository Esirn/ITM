# 02 → 01：D/L/R 修正实现的独立软件验收

读取 **`8e4df1e`**，对照 `62eed84` 冻结的 D/L/R 包完成独立软件验收。**通过，无新增返工项；01 可进入新预算授权申请。** 不再重复冻结设计或验收相同实现。此次结论仅覆盖软件、输入和准备资产；真实 GPU 执行、D 诊断和 L/R 模型效果仍待取得授权后验证。

本轮未加载或前向预训练 MDM，未启动真实训练、完整动作采样或 GPU 预检，未创建真实授权记录。联合测试包含明确的微型 CPU 模型/假 diffusion/故障 fixture，它们不是 D/L/R 实验产物。改动提交由 `git log -1 -- docs/itm_rework/protocol/B0_CORRECTION_ACCEPTANCE_20260926.md` 定位。

## 验收身份

| 对象 | 固定身份 |
| --- | --- |
| 设计/评价包 | `canonical-B0-correction-package-v1`；SHA256 `e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a` |
| 方法实现 | `8e4df1e`；逐文件身份在 v3 implementation manifest 中 |
| 最终准备 manifest | `outputs/itm_rework/method/correction_implementation_20260926_v3/implementation.json`；SHA256 `449b8d278a5a4d1ee31c6dad8a9eedbe59c2990f1efceedc45ad4ae70c805676` |
| 共同初始权重 | 同目录 `initial.pt`；SHA256 `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401` |
| 公共数据契约 | `itm-rework-v0.3-draft`；原 3 train / 3 development-val，原准入范围不变 |

01 文档中的 manifest 副本与上述 v3 原文件逐字节相同；126 个绑定代码/包/数据/骨干文件和 initial 文件全部 hash 核对通过（合计127）。v1/v2 早期准备资产没有替换为本批批准对象。后续授权应绑定 **01 上述 v3 manifest**；02 的重放目录用于验收证据，不作为另一个新运行计划。

## 已执行的独立验证

**联合回归：203 passed，0 failures / errors / skipped，34 warnings，100.94 秒。** 包含本次方法新增的25项测试；warnings 是既有 Transformer nested-tensor 配置提示。使用最终 `8e4df1e` 全部代码重新运行，覆盖01最终状态字段修改，不仅沿用其早期203项日志。

证据：[regression.log](b0_correction_acceptance_20260926_v1/regression.log)、[JUnit](b0_correction_acceptance_20260926_v1/regression.xml)。

另行编写02独立参考核验脚本，未仅复述方法侧输入检查：

- 从6个已交付 raw IMU NPZ 出发，以 `numpy.interp` 和 SO(3) relative-log 插值重建20 Hz观测，再独立构造 static/tail-reversed/numeric-zero。**全部120槽位**的观测、参考、原始O[0]、sensor mask、时间、caption、GT与motion normalization通过；normalized input/reference相对独立参考的最大绝对差均为 **0**。
- 两臂对应的 **60份独特条件输入**逐数组哈希相同。tail-reversed是在插值后联合反转第1至T−1帧；第0帧和静止参照不动。numeric-zero的参照来自原O[0]；没有把零矩阵当旋转。
- 重新由3个train片段的 **398个独特20 Hz帧**检查归一化人口，均值逐值相同，标准差满足原 `rtol=1e−7, atol=0` 核对要求；不计重复配置，不纳入val。
- 在新目录 `outputs/itm_rework/protocol/b0_correction_acceptance_prepare_20260926_v1/` 重放准备。用拦截器禁止任何 `nn.Module` 前向、GPU初始化/设备查询/同步和GPU预检；准备仍完成，禁止调用计数为0。
- 新准备的环境、126个绑定文件列表、60份输入metadata与01 v3相同。共同初始权重**逐张量hash相同，整个initial.pt也逐字节相同**；初始state记录的tensor hash再次从实物独立计算通过。准备manifest仅因新initial文件路径及其引用而变化。

证据：[independent/acceptance.json](b0_correction_acceptance_20260926_v1/independent/acceptance.json)、[independent.log](b0_correction_acceptance_20260926_v1/independent.log)、[02核验脚本](/home/a200/0proj/ITM/scripts/accept_b0_correction_implementation.py)。该脚本只产生准备/核验资产，不调用 execute，不产生授权。

## 对冻结要求的代码核对与测试覆盖

| 项目 | 核验结果与边界 |
| --- | --- |
| 真实长度梯度累积 | 每条用自身T，normalized-x0 MSE按有效263×T取均值；三条各loss/3，200次step/600次呈现；null不改分母，无训练图可跳过backward。新路径未调用旧梯度裁剪；CPU梯度参考及第201次拒绝通过。 |
| R参照差分 | 同一encoder权重，观测/参照各在同一登记seed的局部RNG上下文；参考不detach，caller RNG恢复。相同输入共享dropout时差分零，CPU梯度满足冻结容差；bias不使用且冻结，原strict-null/raw-bypass行为通过。 |
| 训练随机流/共同初始化 | 直接消费冻结600行，不另行shuffle/抽t/presence；全部训练noise hash重建通过。两臂实际worker循环的微型CPU fixture各完成200次更新、600次呈现，只发布50/100/150/200；这不是新B0训练。 |
| D数量与上下文 | 384个单条＋32组三条padding＝480 branch examples；96组有效噪声重建、exact/padded共同有效x_t检查通过。158帧仍是无新增padding参照；backbone mask_frames保持false。 |
| D可复算记录 | 21个hook点在同一次模型调用内读取，排除前置token/padding后保存float32数组；worker fixture检查保存形状、reductions及匹配x_t。paired/static/reversed/text-only/padded差值由已存数组计算，不额外调用模型。 |
| 采样与缓存 | 120固定槽位，原val随机地址保留；每臂绑定自己的final200。缓存核对输入/参考/mask/time/presence/checkpoint/mode；R分别编码并缓存差分，1000步、4000 branch examples及static/text-only完整输出一致在CPU fixture通过。真实修正模型尚未采样。 |
| 全队列限额 | 总7200秒含预检前核验；D600，L/R各600且共享1200，全部采样共享5400；代码按更早的总/阶段绝对deadline执行，R采样不重置。24GiB allocated限制在真实模型加载前配置。硬件上的实际显存/速度尚未实测。 |
| 故障/未运行分母 | CPU顶层stub覆盖成功、预检失败、D超时、R部分199-update、第二采样OOM/超时、R-static超容差、槽位间共享期限耗尽。完整480/120计划分母及已完成文件保留，停止后无重试；launched=false区分未启动槽位。真实派生进程组超时由联合限额测试覆盖。 |
| 身份/授权 | 原授权、改目录/GPU/implementation hash不能通过新入口；准备与执行分离，实际执行需新授权、唯一目录/GPU及预检。本批没有真实授权，测试内显式CPU_FIXTURE记录不是用户批准。 |

代码复核范围为 `mdm_correction.py`、`mdm_correction_limits.py`、新worker/父运行器及 `mdm_imu_control.py` 的增量，并核对所复用的DDPM、CFG、旧限额和预训练MDM输入入口。未发现需要修改冻结包或方法代码的差异；没有新加模型效果门槛或验收之外的防护要求。

## 下一步：01申请具体新队列授权

本批独立软件验收已经完成。**下一步由01对已实现、已冻结的整队申请新授权，然后GPU预检并单次执行：**

- D：480 branch-example探针，最多10分钟。
- L/R：各200 updates / 600 exposures，训练各最多10分钟、合计20分钟。
- L/R：各60条完整采样，合计120条，共用最多90分钟。
- 总计：单GPU、allocated≤24GiB、整个队列最多 **2小时**；任一超时/OOM/非有限值/身份或计数异常停止整队，保留完成及失败产物，不自动重试或回退。

授权记录须绑定本报告列出的package/01 v3 implementation SHA、一个尚不存在的绝对run_directory、具体GPU UUID和实际用户指令。保持准备时相同环境（含`OMP_NUM_THREADS=2`）；**本次验收不等于授权**，旧B0剩余额度不可沿用。01一旦取得新授权，无需为本批相同实现再次往返确认；若绑定代码/输入/环境发生实质变化，应明确交付差异，不能复用旧manifest身份。

02在真实队列完成后按冻结设计独立重算D记录和L/R全部样本，保留所有失败与负结果。当前没有修正有效性、真实IMU利用、泛化、物理观测闭环或相容编辑的新增结论。

## 执行记录与未改动项

完整命令及范围保存在 [commands.md](b0_correction_acceptance_20260926_v1/commands.md)。01原准备资产只读，旧36输出/评价/数据/协议保持不变；02新增核验脚本、日志、报告并更新自身交接与公共证据索引。用户未跟踪的任务目录未纳入提交。
