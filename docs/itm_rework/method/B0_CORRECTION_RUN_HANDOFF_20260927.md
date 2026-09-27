# 01 → 02：授权 v2 D/L/R 实际运行交付

**480/480 个 D 探针、L/R 各 200/200 updates、各 60/60 条完整采样全部完成。** 本次队列退出码 0，总耗时 **2423.55 秒（40.39 分钟）**，无超时、OOM、非有限值、身份/计数失败或重试。下一步由 **02 按原冻结协议独立评价** D 诊断与 L/R 效果；完成运行不等于模型有效性通过。

本次用户明确回复“批准通过”，批准 `71088af` 绑定的新单次运行申请。执行修复 `03e6d95`，依据 `08698b4` 的独立差异验收及此前 `d47891f` 的未变更部分验收；未修改运行代码、冻结计划、数据或评价规则，也未重复未变更的软件测试。

## 身份与资源

- 实际授权：[B0_CORRECTION_AUTHORIZATION_20260927_V2.json](B0_CORRECTION_AUTHORIZATION_20260927_V2.json)。该授权现已执行完毕，不用于追加任务或重跑。
- 唯一运行目录：`outputs/itm_rework/method/correction_authorized_20260927_v2/`。
- implementation：`outputs/itm_rework/method/correction_implementation_20260927_v1/implementation.json`，SHA256 `a802ef3b03aea7bab7fff3fda5704cf56a2c5cf6c5ed36c693a892fc35d6de13`。
- 冻结 package SHA256：`e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a`；共同初始权重 SHA256：`3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401`。
- 单 GPU UUID：`GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`；预检总显存 24564 MiB、空闲 24121 MiB、compute processes 0。启动前 127 份资产 hash 与完整环境核对通过；`OMP_NUM_THREADS=2`。
- PyTorch allocated 峰值 **677558784 bytes，约 0.631 GiB**，上限 24 GiB。退出后的只读进程查询确认该 GPU 已无 compute process；没有换卡或 CPU 回退。

| 阶段 | 完成量 | 实际阶段耗时 | 原限额 |
| --- | --- | --- | --- |
| D | 480 branch examples / encoder examples；0 updates | 133.44 秒 | 600 秒 |
| L train | 200 updates、600 exposures、600 MDM / 600 encoder examples | 28.81 秒 | 600 秒 |
| R train | 200 updates、600 exposures、600 MDM / 1200 encoder examples | 33.80 秒 | 600 秒；两臂共享 1200 秒 |
| L/R sampling | 各60条，共120条；120000 denoising calls / 480000 MDM branch examples；encoder共180 examples | 2227.27 秒 | 共用5400秒 |
| 整队 | 五个阶段全部 completed | 2423.55 秒 | 7200秒，从GPU预检前计时 |

阶段耗时由父进程未被总期限截断的固定截止时间还原起点，并计算相邻阶段边界；采样到最终队列完成。包含加载、保存与验证开销，总时间还包含阶段前的预检。原始时钟、计数、GPU记录及副本身份见 [execution_summary.json](correction_run_20260927_v2/execution_summary.json)。

## 交付内容与完整性核验

以下路径均相对本次唯一运行目录：

- `D/D*.npz` 与同名 JSON：480 个固定探针的预测、目标、x_t、mask、同次前向的 hook 数组及 reductions；`D/paired_differences.json` 保留624条从已保存数组计算的对照差异，没有增加模型调用。
- `L_exact_length/model.pt`：final200，SHA256 **`8a329936f8994760afbafdb44cc8aa60c4a23600726d7b12ae306d1d2aa1ab80`**。
- `R_static_reference_delta/model.pt`：final200，SHA256 **`91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`**。
- 两臂各保留200份 update JSON及50/100/150/200四个 checkpoint；固定采样只使用该臂 final200，未按验证结果选择模型。
- `samples/<case_id>.npz` 与同名 JSON：120个原槽位全部完成，含 normalized HumanML263、joints、GT、原始和归一化 IMU/reference、O[0]、时间、mask、caption、随机计划及模型身份。
- `runtime/` 共370个文件：最终status及123个子进程各自的context/receipt/log。全部原始文件保留在outputs；完整路径/hash清单位于交付索引，关键status、D/L/R receipt/log另有逐字节副本。

只读 CPU 交付核验已通过，证据：[delivery_integrity.json](correction_run_20260927_v2/delivery_integrity.json)。检查 **1733份引用资产hash**、**12720个数值数组有限值**；NPZ内metadata与独立JSON相同，120个输出的形状/长度对应槽位并绑定各臂final200。两臂各600条实际训练记录的顺序、条件、长度与实际噪声hash逐项匹配同一冻结计划。480/120分母均完整，没有过滤槽位。

运行器对已有样本完成 **24项一致性检查**，全部 `max_abs_m=0` 且逐位相同：L、R各6组 text-only/backbone-bypass，以及R的12组 physical-static/text-only。检查仅比较已生成文件，未追加前向；静止参照恢复和空条件一致性不证明动作能力或动态IMU收益。

复核入口（仅CPU读文件，不调用模型）：

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python \
  docs/itm_rework/method/correction_run_20260927_v2/verify_delivery.py
```

原执行命令和核验命令：[commands.md](correction_run_20260927_v2/commands.md)。程序退出后未训练、采样或重试。上一批v1加载失败目录及授权记录保持原样，未将该失败改记成功。

## 给 02

请按 `62eed84` 冻结包中的评价、审阅顺序、对照、容差和失败分母评价本次全部480探针、两臂400次更新与120条样本。D的机制证据和L/R动态条件收益均需独立分析，保留失败/无收益或姿态退化结果；不因本次完整运行覆盖旧B0负例。

01本轮完成执行和交付，不自行追加训练、采样或新消融。适用范围仍为预先统一坐标的合成B0；模型有效性、真实设备标定、生成IMU闭环及相容编辑能力未因此认证。
