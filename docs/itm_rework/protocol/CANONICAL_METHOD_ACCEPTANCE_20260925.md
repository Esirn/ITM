# 02 独立验收：合成 B0 输入与归一化

读取方法提交 **`1d7a32d`**，数据提交 `c36ea56`，公共契约保持 `itm-rework-v0.3-draft`。本次**输入与统计验收通过，无输入接入返工项**；不等于训练预算已获授权或运行限额已实现。

## 独立重放结果

产物目录：`canonical_method_acceptance_20260925_v1/`。主报告 `replay/acceptance.json`；代码 `scripts/accept_canonical_b0_handoff.py`。

先直接读取交付 NPZ，以 `numpy.interp` 计算 20 Hz acceleration，以相邻 SO(3) 旋转的相对 log/exp 计算 orientation；独立参考构建完成后才调用方法适配器。没有复用方法的时间重采样或归一化 helper 作为真值。

| 核验 | 实测结果 |
| --- | --- |
| train 来源 | `000027` 104 帧、`000506` 136 帧、`004728` 158 帧，共 **398 帧** |
| val 来源 | `000433` 41 帧、`000382` 94 帧、`000139` 163 帧，共 298 帧；不进入统计 |
| 归一化 | 全六槽逐轴 train-only 总体均值、std 与方法保存 float32 值**逐值相等**；std 实际最大差 0，未使用放宽历史 acceleration 阈值的做法 |
| 错误统计范围反证 | 使用原 30 Hz train 数组，均值最大差约 **0.01432 m/s²**；混入 val，均值最大差约 **0.05162 m/s²**，均不匹配实际统计 |
| 六条输入 | acceleration、SO(3) 插值后 orientation 均与独立参考逐值相等；motion、GT 直接读取新 20 Hz 文件，时间与 caption index 一致 |
| 训练/采样路径 | 三条 × head/wrists 共 **6 个输入逐值相等**；实际 pretrained Mean/Std 下的 motion 归一化也一致 |
| 验证条件 | **36/36** 槽位的真实输入、sensor mask、text/IMU presence、bypass 标志、完整长度和直接 GT 全部通过 |
| 随机计划 | 6 个 case/config 噪声组，每组 6 条件共用原长度与 seed；只核对地址保留，**未抽取噪声或生成** |
| 01 报告 | CPU `--input-only-report` 重放的统计、provenance、6 个 input metadata 与原报告一致；仅本轮输出/预检报告路径不同。01 的 36 槽位汇总 JSON 逐字段相等 |

provenance SHA256 保持 **`9f39bf7ffd93b2595dfdcea4ee3288761b14a6e75a6585fedb841fd490295ef1`**。独立核对 payload 摘要、实际文件 hash、selected train IDs、normalization 摘要与来源字段。head/wrists 未重复计数；仅从数值均值无法辨别“把全部样本等量复制一次”，因此同时核对 unique_frames=398、item/source 列表及方法的去重分支。

IMU-only 在追溯信息中保留审核 caption，但 `text_condition_present=false`；包装器屏蔽文本 embedding，不能把这段追溯文字误当成实际使用了文本。物理静止为零原始 acceleration 加首个观测 orientation 固定，使用新 train 统计；numeric-zero 则把归一化 tensor 置零、presence 保留，两者已分开验证。

## 软件回归与证据范围

独立执行 01 列出的完整联合回归：**143 passed、23 warnings、无 skip**，日志 `regression.log`。包含微型 fixture 的真正文本缺失、bypass/strict-null、异常恢复及新数据接口检查；不计为真实 MDM 前向或生成能力证据。

旧 caption 和 cache-migration 脚本分别在本批新路径重放，结果均与先前 02 保存报告逐字段相等，见 `legacy_report_comparison.json`。没有修改旧报告、新数据映射、原四项 acceleration 失败或数据准入清单。5 条 uncertain 和 1 条方向冲突继续排除，真实设备标定与生成严格 IMU 闭环仍不在认证范围。

本次执行了 1 次 CPU 训练输入预检、1 次方法汇总重放、独立数组计算及软件测试；**真实模型前向、FK、训练更新、扩散采样和 GPU 作业均为 0**。输入预检没有创建 checkpoint。

## 运行准备：已具体化，仍未批准启动

`execution_plan/execution_plan.json` 给出一条训练 argv 模板和 **36 条单 case 采样队列**，每条都绑定原槽位、caption、噪声组、seed 和独立输出路径。`execution_plan_validation.json` 验证这些参数能由当前 CLI 解析；**解析通过不是限额执行验收**，模板的 `<AUTHORIZED_GPU>` 尚未赋值，也未创建运行目录。

采用每条单独调用，目的是某一后续样本超时/OOM 时，之前完成的样本已单独落盘。每条加载模型的开销必须计入共享采样期限，不能给 36 个作业各分配 3 小时。若启动成本导致预算内不能完成，保留未完成分母，不偷偷扩大预算。

01 执行前需落实以下具体限额；不需要再重复改已验收的数据输入：

1. 训练从进程开始计时最多 **3600 秒**，包括加载/保存；实际 optimizer update 硬上限 **200**。非有限 loss/梯度在更新前停止，不能仅靠最终输出检查。
2. 采样所有子进程共享一个 **10800 秒**截止时间；最多 36,000 去噪步调用、144,000 分支样本前向。总 GPU 运行最多 14400 秒，单 GPU。这里沿用原预算，不新增计算额度。
3. **24 GiB allocated memory 上限**在模型分配前落实，保存峰值与失败原因；当前 `min_free_gib` 检查和事后峰值记录均不能替代硬上限。
4. 队列启动前写槽位状态；完成后提交产物 hash 与完成状态。超时、OOM、NaN、身份不符即停，保留已完成输出、部分 checkpoint 和全部失败/未运行槽位，不自动重试或转 CPU。
5. 只有完成 update 200 的最终 checkpoint 可进入既定验证；已有每 epoch checkpoint 可以保留，不得自动将部分训练 checkpoint 当成计划的最终模型。

01 的交接已准确说明本批未落实完整受限运行调度；因此这不是新发现的输入返工。建议用 CPU 微型 fixture 注入超时、OOM/非有限值和更新次数上限，验收后再选定实际 GPU 做只读预检。

用户预算授权仍待获得，原 `b0_budget.json` 保持不变。未来授权应另存一份绑定该预算 hash 的授权记录，不在已绑定到 package/provenance 的旧预算文件中改状态。相容 witness 不阻塞 B0；本批不要求额外 witness 或重复 143 项验收来替代限额准备。

下一步：01 完成限额与失败保留，02 核对该具体执行方案；预算授权与 GPU 预检满足后，在既定预算内执行一次 B0。无需为本次已通过的输入验收再往返确认。

## 复现

```bash
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python scripts/accept_canonical_b0_handoff.py --output <新的独立重放目录>
/home/a200/miniconda3/envs/itm/bin/python scripts/plan_canonical_b0_execution.py --output <新的计划目录>
```

第一条命令只调用有 `--input-only-report` 的训练 CLI；第二条只写模板和队列，不启动其中命令。143 项回归命令沿用 `method/CANONICAL_B0_HANDOFF_20260925.md` 的最终完整列表。
