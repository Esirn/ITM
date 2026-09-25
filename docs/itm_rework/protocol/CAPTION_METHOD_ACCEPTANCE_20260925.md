# 02 对 caption-seconds-v3 的独立验收

方法对象 `1a7e887ae4c258cdd7325ca5853c2ff3001009d0`；本次读取 HEAD `d360e91`。
公共契约保持 `itm-rework-v0.2-draft`；接纳 opt-in `itm-method-caption-seconds-v3` 的输入实现，没有冻结研究协议或批准训练。
02 改动提交可用 `git log -1 -- scripts/accept_caption_handoff.py` 定位。

## 已核实

- 方法快照中 5 个 owner-01 文件和 5 个公共依赖 hash 匹配。
- 独立执行 `check_mdm_caption_handoff.py`，输出 JSON 与 01 保存报告逐字段完全一致。
- 直接核对旧 smoke checkpoint hash 和其中 normalization；方法输入检查所用模型卡统计完全匹配，没有重算真实训练统计。
- 固定 9 个通过来源 × head/wrists = **18 条输入**（6 train、12 val）；`000004` 两配置按计划拒绝，没有将 199 帧全段静默截成 196。
- `000006` 使用 `[0,100)` 特征切片，`000133` 明确保留 caption index 1。三条成功 train 来源的两配置均重现了训练/采样 loader 的 IMU 数组和文本一致。
- 固定 13 条 SSM 原区间继续因为源支持不足拒绝，未插入时钟假设、时间拉伸或 FPS 四舍五入。
- 诊断 sidecar 的 CLI 训练准入继续拒绝；即使旧 diagnostic override 打开，也不能绕过该新路径的准入门槛。
- 新输入 metadata 保留 `training_admissible=false`、`uncertified_diagnostic`、`notebook_clock_correction=none`；旧 checkpoint 的训练契约仍为 `itm-rework-v0.1-draft`。

方法、数据和协议联合回归 **109 passed, 19 warnings**。warnings 为既有 PyTorch 提示。
包含显式索引、非零起点双时间轴、训练/采样一致、裁片统计、陈旧 hash/时间数组/文本/长度覆盖拒绝、设备初始化前准入拒绝与非零起点 witness 隔离。
此前 68 项是 01 的检查；109 项是本批独立联合验收，旧日志未改写。

## 验收边界

真实固定计划的裁片起点均为 0；**非零起点只由合成 fixture 验证**，未认证真实非零裁片的根参考、部署坐标或语义。
独立真实输入重放使用 paired 路径，没有执行实际采样 CLI/新生成，也不把代码支持的各 mode 全部写成新的生成实验。
读取本地源数据并正确执行 caption 秒轴，不代表源数据本身与 HumanML 物理时钟对齐。
`HUMANML_CLOCK_20260925.md` 的 SSM 时钟诊断及 `PREFIX_ALIGNMENT_20260925.md` 的前缀诊断仍是独立问题，本次没有安装它们的变换。

本验收不提供新的 B0 checkpoint、compatible witness、训练/生成预算、冻结阈值或正式 P/C/D 成功率。
原 seconds-v2 仍可用于既有诊断复现；不能将旧 v0.1 checkpoint 标成新输入管线训练产物。

## 复现

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/accept_caption_handoff.py --output-dir docs/itm_rework/protocol/caption_acceptance_20260925_v1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py > docs/itm_rework/protocol/logs/caption_acceptance_regression_20260925.txt 2>&1
```

输入重放单线程 CPU、子进程上限 120 秒，模型前向/训练/生成/GPU 均为 0；软件测试的小型模型 fixture 与研究作业分开。
`caption_acceptance_20260925_v1/` 保存新验收 JSON、重放 JSON 和日志，输出目录必须全新。
验收脚本、模型卡/checkpoint 与源文件 hash 都记录在 `acceptance.json`；联合回归日志在上方命令路径。
本批只增加 02 验收及文档/证据，没有修改 01 方法文件、公共计划、候选准入或原始产物。
