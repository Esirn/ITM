# 02：固定 R 采样对照独立软件验收

读取方法提交 **`f9b3cb18053892dd9e66dbdf31085883a99b76d1`**。**软件验收通过，无新增返工项；下一责任方为01，申请绑定最终实现的新单次运行授权。** 本次未运行真实预训练模型前向、训练、采样、GPU查询或预检，没有创建实际运行授权。通过CPU验收不代表模型有效性通过认证。

冻结协议仍为[固定R采样对照冻结](B0_SAMPLING_ALIGNMENT_FREEZE_20260927.md)，package SHA256：

`537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60`

验收implementation：`outputs/itm_rework/method/sampling_alignment_implementation_20260927_v1/implementation.json`，SHA256：

`793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4`

固定R final200 checkpoint SHA256：

`91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`

## 独立证据

- **258 passed，0 failures/errors/skipped，180.72秒**；包含本次33项方法测试。02重新执行完整联合回归，日志/XML位于[验收证据目录](b0_sampling_alignment_acceptance_20260927_v1/)，不是引用01的测试结果。45条warning为既有Transformer nested-tensor提示。
- **743份绑定文件**逐一SHA256核验通过；方法文档manifest副本与原文件相同。在新目录独立prepare得到相同manifest hash，**逐字节一致**，准备目录只有manifest，没有新checkpoint。CPU读取旧R状态，核对final200/600 exposures、原始初始化、配置、环境、归一化和训练输入来源。
- [独立审计脚本](../../../scripts/accept_b0_sampling_alignment.py)的[审计结果](b0_sampling_alignment_acceptance_20260927_v1/acceptance.json)记录**480 Q＋60 S1＝540槽位**、66种按split/source/config/transform区分的输入。直接从原始NPZ以独立SO(3)插值重建20 Hz输入，核对原始acceleration、orientation、timestamps、sensor/frame masks、caption、GT和HumanML263目标；静止参考始终来自原始O[0]。normalized input/reference最大绝对差均为0，未放宽冻结阈值。
- **398个唯一train帧**重算均值/标准差，与绑定统计一致；val不进入统计。S1每槽位11个输入数组同时按旧R输出及冻结hash核对。独立prepare禁止nn.Module前向、CUDA查询、preflight和子进程启动，触发次数为0。

命令、复现路径和审计开发期一次字段访问错误的保留记录见[commands.md](b0_sampling_alignment_acceptance_20260927_v1/commands.md)。该错误在02脚本中修正后，使用新目录完整重跑；方法文件未改动。

## 实现核对结论

| 冻结要求 | 软件证据与结论 |
| --- | --- |
| Q仅96次batch5，480分支 | 实际worker的CPU替代模型执行全部96组；顺序f00/f10/paired/static/reversed、同组同x_t、真实长度及严格null门控通过。前两支不调用IMU encoder，后三支各观测/参考两次，共576。288派生、384内部对照、192差值之差齐全；Q无f01。 |
| 固定R、S1只改wt | loader核对固定checkpoint/config/provenance，eval且冻结参数；复用原四分支batched factorized composer和1000步addressed DDPM，没有改为直接f11。旧noise地址/plan和输入继续绑定；IMU-only真正关闭文本。 |
| trace额外调用0 | 六种条件下，实际guidance/sampler CPU fixture完成1000步，开/关trace的输出、随机状态、cache及计数逐位相同；同次四分支hook捕获，返回原张量，保存预测hash与实际返回hash相同，hook随后移除。四个t按900/500/250/50访问。 |
| 算术、null与decode | 两种wt及不同数值规模通过原float32舍入界；Q float64分解保留交叉项正负。静止/关闭条件对应分支等式、CPU decode、24项完整采样比较及缺失counterpart的not_comparable路径覆盖。 |
| 首错即停与产物保留 | CPU注入静止等式、组合、trace写入、预检、Q超时、sample OOM、缺trace、一致性、共享deadline、非有限及额外调用/训练错误。480/60/240完整分母在预检前登记，已完成产物保留，后续not_run_after_stop；motion存在而trace不全时不记完整交付。 |
| 整队限额 | Q最多600秒；60条S1共享2400秒；从预检前起总3600秒，阶段与总deadline取较早值；单GPU/24GiB allocated。调用前预留计数，总MDM分支240480、encoder最多696、updates为0；不重试、回退或续用旧授权。 |

CPU fixture使用微型替代模型/外部factory替身，模拟预检与子进程；未用实际预训练权重执行这些前向。1000步trace fixture将高频receipt持久化替换为内存操作，队列/故障测试另行覆盖磁盘保留。故本轮不能证明真实GPU运行时间、显存、数值一致性或动作质量；实际运行必须继续执行冻结检查。

## 下一步

01申请一次**480 Q＋60完整采样＋240条零额外前向trace**的新授权，绑定上述package/implementation、唯一新run目录和GPU UUID；保持单GPU、24GiB allocated、合计最多1小时，禁止训练及旧臂重跑。收到授权后由01预检并执行一次，交付所有成功/失败槽位和产物，再由02按已冻结协议独立评价。相同实现无需再次冻结或重复软件验收。

旧R全部负例、旧60样本、原阈值和120条新旧混排审阅顺序保留。Q有限但差的结果不改变S1、不选guidance；静止/null一致性不构成动态IMU收益，IMU-only在此干预下预期不变。现有n=3/分割、单训练seed、K=1的开发面板仍不支持泛化、严格生成IMU闭环、物理相容编辑或真实设备结论。
