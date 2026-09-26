# 01 → 02：授权 D/L/R 队列加载失败与 CPU 修复

用户批准后，01 按 `d47891f` 已验收的 v3 实现启动了一次 D/L/R 队列。**GPU 预检通过，但 D 在模型加载处失败，整队停止，未自动重试。** 这不是 D 的数值诊断结果，也没有产生新的训练 checkpoint 或动作样本。修复与新的 CPU 准备已完成，下一步由 02 核验本次加载器差异；之后再申请新队列授权。

## 实际执行与失败保留

- 授权记录：[B0_CORRECTION_AUTHORIZATION_20260927.json](B0_CORRECTION_AUTHORIZATION_20260927.json)。该记录仅绑定本次旧 manifest、GPU 和唯一 run 目录，不用于修复后的新运行。
- 原实现：`8e4df1e`；独立验收：`d47891f`；本次 manifest SHA256：`449b8d278a5a4d1ee31c6dad8a9eedbe59c2990f1efceedc45ad4ae70c805676`。
- 原始运行目录：`outputs/itm_rework/method/correction_authorized_20260927_v1/`；退出码 **1**，从队列入口到停止 **4.04757 秒**。
- 单 GPU UUID：`GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。预检总显存 24564 MiB、空闲 24121 MiB、compute processes 0；receipt 记录峰值 allocated **500039680 bytes（约 0.466 GiB）**。
- CLIP/MDM 权重加载和设备迁移已发生；MDM/IMU encoder 前向计数均为 **0**，训练尝试/完成更新 **0**，采样去噪调用 **0**。
- D stage 为 `failed`；480 个尚未开始的 probe 全保留为 `not_run_after_stop`。两臂训练与采样阶段未启动，120 个 sample 全保留为 `not_run_after_stop`，没有删去失败分母。
- 停止后只读 GPU 进程查询确认本次设备上无 compute process；另一块 GPU 的已有进程保持不动。

原始四份 runtime 文件已逐字节复制保存：[status.json](correction_run_20260927_v1/runtime/status.json)、[D log](correction_run_20260927_v1/runtime/D_probe.log)、[receipt](correction_run_20260927_v1/runtime/D_probe.receipt.json)、[context](correction_run_20260927_v1/runtime/D_probe.context.json)。[run_summary.json](correction_run_20260927_v1/run_summary.json) 记录原路径/hash、计数、失败及新旧准备差异；原目录未改写。

## 原因、修复与验证边界

异常是 `AttributeError: 'NoneType' object has no attribute 'eval'`，发生于 `run_mdm_correction_worker.load_model()` 的 `model.to(device).eval()`。上游 `outputs/mdm/runtime_root/model/mdm.py` 的 `_apply()` 与 `train()` 均无返回值；PyTorch 的 `.to()` 因而返回 `None`，不能继续链式调用 `.eval()`。

生产代码仅把调用拆为 `model.to(device)`、`model.eval()`、`original.eval()`，保留对同一模型的设备迁移和 eval 操作，不修改上游代码、模型参数定义、随机计划、输入、评价规则或预算。

此前 worker CPU fixture 替换了整个 `load_model`，因此没有覆盖真实加载接口。新增测试直接调用实际 `load_model`：从已绑定上游源码提取原样 `_apply/train` 方法，接入微型 CPU 模型，仅替代重量级模型工厂/兼容导入；使用真实 state 加载、控制器安装、设备转换及 eval。测试禁止所有 `nn.Module` 前向，并核对模型身份、所有模块及 SMPL 替身的 eval 状态、CPU 参数与控制器权重。

新增测试在修复前 **1 failed**，复现同位置同异常；修复后 **1 passed**。日志：[before](correction_run_20260927_v1/logs/loader_before_fix.txt)、[after](correction_run_20260927_v1/logs/loader_after_fix.txt)。**完整联合回归 204 passed / 0 failed / 0 skipped，36 warnings，100.94 秒**；计数与文件 hash 见 [cpu_verification.json](correction_run_20260927_v1/cpu_verification.json)。结果见 [regression.txt](correction_run_20260927_v1/logs/regression.txt) / [JUnit](correction_run_20260927_v1/logs/regression.xml)。这些 CPU fixture 不计作 D/L/R 实验结果。

## 新准备身份与交接

仅 CPU 准备的新目录：`outputs/itm_rework/method/correction_implementation_20260927_v1/`。没有在失败后执行真实模型前向、训练、采样或另一轮 GPU 预检。

| 项目 | 身份/结果 |
| --- | --- |
| 新 implementation SHA256 | `a802ef3b03aea7bab7fff3fda5704cf56a2c5cf6c5ed36c693a892fc35d6de13` |
| initial.pt SHA256 | `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401`；与已验收 v3 整文件及逐张量相同 |
| 126 份绑定资产 | 路径集合相同，仅 `scripts/run_mdm_correction_worker.py` 的 hash 改变；加 initial 共 127 份重新核验通过 |
| 60 份输入 metadata / 环境 | 与 v3 完全相同 |
| 设计包 SHA256 | `e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a`，保持原冻结包 |

新 manifest 的逐字节副本：[implementation.json](correction_run_20260927_v1/implementation.json)。所有执行、测试与准备命令见 [commands.md](correction_run_20260927_v1/commands.md)。

请 02 仅对本次加载接口修复、新 CPU 测试、失败保留及新 manifest 绑定做独立差异验收，无需重做已通过的冻结设计/输入核验。验收后由 01 申请另一次相同上限的队列：480 probes、L/R 各 200 updates 与 60 条完整样本，单 GPU / 24 GiB / 总 2 小时；仍按 D 600 秒、每臂训练 600 秒、采样共享 5400 秒及首次故障停止执行。

**本次旧授权已用于上述失败队列；本提交不创建新授权，不重启队列。** 真实 D/L/R 执行与模型效果仍未验证，原 B0 负结果与准入边界保持原结论。
