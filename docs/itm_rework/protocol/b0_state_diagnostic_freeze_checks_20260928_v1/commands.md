# 02 CPU-only 冻结与复现命令

工作目录 `/home/a200/0proj/ITM`，Python `/home/a200/miniconda3/envs/itm/bin/python`。仅读已有产物；freeze/replay 阻止神经框架导入与子进程；无模型权重张量加载、真实前向、训练、采样、GPU 查询或新授权。

首次构建在读取 symlink 依赖时因路径表示失败，输出目录尚未创建；异常见 `development_symlink.txt`。修正为真实路径在仓库外时保存绝对路径，未改来源或容差。

首次成功构建（已完成；现有冻结目录拒绝覆盖）：

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/freeze_b0_state_diagnostic.py > docs/itm_rework/protocol/b0_state_diagnostic_freeze_checks_20260928_v1/build.txt 2>&1
```

独立临时目录复建提案 7 JSON 与冻结 13 文件、核验 1,640 来源（不覆盖包）：

```bash
/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_state_diagnostic_freeze_checks_20260928_v1/replay.py > docs/itm_rework/protocol/b0_state_diagnostic_freeze_checks_20260928_v1/replay.txt 2>&1
```

相关协议测试，9 passed，0 failed/skipped；没有重跑无变更的旧方法联合回归：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_state_diagnostic_protocol.py --junitxml=docs/itm_rework/protocol/b0_state_diagnostic_freeze_checks_20260928_v1/tests.xml > docs/itm_rework/protocol/b0_state_diagnostic_freeze_checks_20260928_v1/tests.txt 2>&1
```

结果：528 条独立残差 × 9 块完全相同、600 exposure 重计相同；48 state/192 branch/144 checks、72 encoder/12 caption 调用仅是冻结计划，尚未执行。所有状态/输入及其 hash 已绑定，9 测试包括有益/有害解析例、带权分区、非有限/错误长度拒绝、全 paired 人口、原始参考/时钟的静止及倒序变换、预算未授权和不可覆盖。
