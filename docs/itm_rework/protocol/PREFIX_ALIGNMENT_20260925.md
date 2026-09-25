# 五类前缀裁剪：固定轨迹核查与独立缓存重建

读取提交 `9439818`；公共契约保持 `itm-rework-v0.2-draft`，没有修改运行时源映射或原 cache。
本批 02 改动提交可用 `git log -1 -- scripts/audit_humanml_prefix.py` 定位。

## 固定十条来源核查

前批 2,177 条仅是代码推导的待核查范围。本批预先固定每类两个 train-exclusive 来源组：按 motion ID 排序，完整 full-clip caption、长度不超过 196 帧，且长度大于名义前缀帧数加 20。
选择只使用既有 metadata；`prefix_audit_20260925_v1/plan.json` 在轨迹比较前生成并锁定输入/code hash。没有根据残差换样本。

三个预先定义的时钟是名义 `i/20`、仅复演 notebook 整数步长、整数步长加数据集前缀。
三种时钟仅比较共同支持的相同目标帧，至少 20 帧；不可估计样本保留且不替换。
本轮全部 10 条可估计，且源 FPS 为精确 60 或 120，前两个时钟重合。因此这轮确实区分了前缀补偿的影响。

| 数据集 | ID | 共同比较帧 / 特征帧 | 未补前缀误差 m | 补前缀误差 m |
| --- | --- | ---: | ---: | ---: |
| Eyes_Japan_Dataset | 000123 | 81 / 140 | 0.062537 | 0.043540 |
| Eyes_Japan_Dataset | 000161 | 113 / 172 | 0.239834 | 0.019265 |
| MPI_HDM05 | 000015 | 92 / 151 | 0.099170 | 0.014499 |
| MPI_HDM05 | 000165 | 100 / 159 | 0.255418 | 0.014190 |
| TotalCapture | 000506 | 117 / 136 | 0.138520 | 0.015535 |
| TotalCapture | 004659 | 164 / 183 | 0.199904 | 0.007918 |
| MPI_Limits | 004728 | 139 / 158 | 0.130932 | 0.019708 |
| MPI_Limits | 005546 | 121 / 140 | 0.087070 | 0.016189 |
| Transitions_mocap | 000027 | 95 / 104 | 0.194241 | 0.010815 |
| Transitions_mocap | 000415 | 110 / 119 | 0.171985 | 0.010847 |

误差是 root-relative 22-joint 的固定 oracle yaw 对齐残差；每种时钟单独拟合一个固定 yaw，没有搜索时间偏移。
**10/10 下降**，逐片段等权均值 **0.15796107 → 0.01725056 m**。
yaw 使用目标全身真值，禁止进入部署预处理；该改善不等于全片段物理/语义同步，更不能外推成 2,177 条已认证。

## 每类首条的五份 CPU 诊断缓存

按固定计划中每类首条选择 `000123/000015/000506/004728/000027`，没有按改善幅度选样本。
先保存 CPU 预算（2 threads、5 个 SMPL forward、每条最多 300 输出帧、最多 180 秒脚本期限），再执行已有标准合成路径。
只对精确 60/120 Hz 的这些固定源显式移动 index 裁片起止，生成新目录中的诊断 cache；没有将该移动装入默认桥接，也不处理 SSM 的时钟倍率问题。

| ID | 新 cache 帧数（30 Hz） | 完整特征帧数（20 Hz） | 完整片段 oracle yaw 误差 m | 重叠加速度 allclose |
| --- | ---: | ---: | ---: | --- |
| 000123 | 212 | 140 | 0.044639 | 未通过 |
| 000015 | 228 | 151 | 0.014371 | 未通过 |
| 000506 | 206 | 136 | 0.015230 | 未通过 |
| 004728 | 239 | 158 | 0.019621 | 未通过 |
| 000027 | 158 | 104 | 0.010788 | 通过 |

5/5 新 cache 支持完整特征秒轴。旧/新 cache 相同物理帧的关节最大差不超过 `3.58e-7 m`，姿态矩阵元素差不超过 `5.97e-7`，均通过既定 `atol=rtol=1e-4`。
加速度比较同时去掉两边不完整的三帧差分/平滑 stencil：最大差为 `0.000128746–0.000129133 m/s²`，**4/5 未通过同一 allclose**。保留负结果，没有放宽容差，也不把 max-abs 和含相对项的 allclose 混为同一判据。
完整片段五条均值为 `0.02092982 m`，与上表十条公共帧结果的样本/帧范围不同，不用于直接比较改善幅度。

这些 cache 是从 AMASS 真值重建的诊断数据，不是新生成模型的输出；人体模型/骨架差异、语义和真实测量闭环仍未认证。

## 产物隔离与给 01 的接口

- 新文件位于 `outputs/itm_rework/prefix_rebuild_20260925_v1/`；路径和 sha256 在 `prefix_rebuild_20260925_v1/report.json`，原 cache hash 与十条固定核查时一致。
- 单独的 `diagnostic_manifest.jsonl` 使用 `standard_imu_prefix_diagnostic_v1`，所有 `training_admissible=false`。按该 manifest 调用标准 reader 均被拒绝；未创建任何生产/训练替换 manifest。
- NPZ 内也记录 diagnostic schema、源映射版本和前缀秒数。这里只验证显式 diagnostic manifest 被拒绝，不宣称当前 reader 能阻止调用者另写错误 schema 来重标文件。
- 不复用这些产物重新训练 B0，不静默改旧 checkpoint 的 normalization；后续迁移需版本化源映射、全片段/坐标/语义审查及训练统计重新核验。
- 01 的 caption 区间集成可独立推进，但不等于补好了数据集级前缀和 SSM 时钟问题。

03 可以引用十个预先固定样本的前缀诊断与五个独立重建的时间支持；不得称部署对齐、全数据修复、真实 IMU 闭环或 B/C 生成收益。

## 已执行命令与检查

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_humanml_prefix.py plan --output docs/itm_rework/protocol/prefix_audit_20260925_v1/plan.json
timeout 60 /home/a200/miniconda3/envs/itm/bin/python scripts/audit_humanml_prefix.py run --plan docs/itm_rework/protocol/prefix_audit_20260925_v1/plan.json --output docs/itm_rework/protocol/prefix_audit_20260925_v1/report.json > docs/itm_rework/protocol/prefix_audit_20260925_v1/run.log 2>&1
timeout 180 /home/a200/miniconda3/envs/itm/bin/python scripts/check_prefix_cache_rebuild.py --plan docs/itm_rework/protocol/prefix_audit_20260925_v1/plan.json --output-dir docs/itm_rework/protocol/prefix_rebuild_20260925_v1 --cache-dir outputs/itm_rework/prefix_rebuild_20260925_v1 > /tmp/itm_prefix_rebuild_20260925_v1.log 2>&1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_humanml_prefix.py tests/test_humanml_clock.py tests/test_physical_time.py > docs/itm_rework/protocol/logs/prefix_regression_20260925.txt 2>&1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_prefix_cache_rebuild.py tests/test_humanml_prefix.py tests/test_humanml_clock.py tests/test_source_crop_support.py tests/test_physical_time.py > docs/itm_rework/protocol/logs/prefix_final_regression_20260925.txt 2>&1
```

最初 10 项，最终相关回归 **13 passed**（新增 3 项），覆盖共同帧公平比较、帧数错误拒绝及双侧有效 stencil。
原重建日志复制至 `prefix_rebuild_20260925_v1/run.log`；另执行了五条文件/hash、有限加速度、合法旋转和 manifest 拒绝核查，见 `verification.json`。
源码快照见 `code_snapshot_prefix.json`；本批无扩散模型前向、训练 update 或 GPU 作业，预算中的 5 次前向专指 CPU SMPL 真值 FK。
既有 31 / 17 项测试日志属于前批，没有改写成此次执行。
