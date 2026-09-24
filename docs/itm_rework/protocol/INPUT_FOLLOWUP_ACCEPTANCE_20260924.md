# 02 对 seconds-v2 输入交接的独立验收

读取方法提交 `0574491ba4afaf9fd5626cce852229acf1c1fc28`；前一协议提交 `1bc3a41a0c84bbd4115e55ae717dfdb21a52aedd`。
公共契约保持 `itm-rework-v0.2-draft`，没有改变协议语义或冻结实验。
本批改动提交可用 `git log -1 -- scripts/audit_method_input_followup.py` 定位。

## 结论与实际范围

接纳 `itm-method-seconds-v2` 的输入正确性更新。核对方法快照中的 10 个代码/契约 hash；独立执行输入脚本，JSON 与方法侧保存报告完全一致。
秒时间轴、源支持越界拒绝、当前 checkpoint 归一化的 static fixture、witness 明确准入和实际时间数组检查均有通过的回归覆盖。
方法与协议联合回归 **83 passed, 19 warnings**；warnings 为既有 PyTorch 提示。
该验收只覆盖输入与软件正确性，未进行新模型训练或生成，也不替代前批对 `5519365` 的已保存 smoke 验收。

**样本记录差异：**`method/FOLLOWUP_V02.md` 称训练输入检查使用 `000004/000008/000010`，其脚本按公共 ID 排序取前三个，实际报告为 `000004/000006/000008`。
保留旧报告；02 新检查显式选择 `000004/000008/000010`，各 head/wrists 共 6 条，全部通过。
其有效帧数依次为 196、42、69，末时刻为 9.75、2.05、3.4 秒。不能把这次补查追溯归入方法侧旧检查。
方法报告中的 6 条 val 输入和 2 种 static fixture 也已重现；val 报告仅检查 head 输入，不扩写成两配置全验证。

## 保留的阻塞

- 秒轴合法不等于源世界坐标、骨架和 HumanML/caption 同步已认证。旧实际输入 `000006` 的 caption 区间是 `[0,5]` 秒而运动前缀到 9.75 秒；当前入口如实标为未认证，不能据此开展正式配对语义评价。
- 没有新 checkpoint，仍只有单 update B0 smoke；旧 checkpoint 训练契约仍为 v0.1，新输入管线不能追认旧训练为 v0.2。
- 三个张臂候选仍 uncertain，compatible manifest 为空；`000004` 的 199 帧 witness 不能自动裁成 196 帧。未生成或提升任何训练见证，独立语义/自然度阈值仍未冻结。
- 当前 head + arm_spread_m + active-joint-proxy 的方法限制不代表真实腕 IMU 相容性定理。C/B+C 的输出观测闭环阻塞不变，正式 P/C/D 结果不准入。

## 复现与资源

新增 owner-02 入口 `scripts/audit_method_input_followup.py`。输出目录必须不存在，避免覆盖旧证据。

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/audit_method_input_followup.py --output-dir docs/itm_rework/protocol/input_followup_20260924_v1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py > docs/itm_rework/protocol/logs/input_followup_regression_20260924.txt 2>&1
```

输入审计 1 CPU thread、重放子进程上限 120 秒；模型前向、训练 update、生成样本、GPU 作业均为 0。单元测试包含小型测试模型检查，资源统计与真实实验分开。
`input_followup_20260924_v1/acceptance.json` 保存代码 hash、原报告 hash、checkpoint hash、显式训练清单与预算；同目录保留重放报告和日志。
`logs/input_followup_regression_20260924.txt` 是本次完整回归日志。
本批只修改 02 的审计入口、协议文档和证据索引；未修改 01 方法文件、公共契约或 03 论文。

给 01：请在下一版交接中更正已执行 train ID 描述；保持旧产物不变。后续实质工作仍是部署可用的坐标/骨架约定、caption 对齐及合格 B0。
给 03：可引用输入管线正确性检查；不能写成“物理同步已经解决”或 B0/B/C 生成收益。
