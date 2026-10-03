# 02 V 独立效果评价命令

读取`260d934`；工作目录`/home/a200/0proj/ITM`。所有命令仅CPU读已保存数组，禁止神经框架导入与子进程/GPU查询；没有执行V、模型前向、训练、采样或授权。

首次评价（输出目录已存在，拒绝覆盖；再次评价须指定新的--output）：

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_state_diagnostic.py > docs/itm_rework/protocol/b0_state_diagnostic_evaluation_checks_20261003_v1/evaluation.txt 2>&1
/home/a200/miniconda3/envs/itm/bin/python scripts/summarize_b0_state_diagnostic.py > docs/itm_rework/protocol/b0_state_diagnostic_evaluation_checks_20261003_v1/tables.txt 2>&1
```

相关解析/协议测试：17 passed，0 failed/error/skipped，0.66秒。8项新增评价测试含有利/有害/null残差、带权分区、非有限/shape错误、Q先平均重复再平均来源、同t配对与差值方向、最终状态链拒绝；另9项既有V协议测试。未重复运行无变更的方法135项测试。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_state_diagnostic_evaluation.py tests/test_b0_state_diagnostic_protocol.py --junitxml=docs/itm_rework/protocol/b0_state_diagnostic_evaluation_checks_20261003_v1/tests.xml > docs/itm_rework/protocol/b0_state_diagnostic_evaluation_checks_20261003_v1/tests.txt 2>&1
```

临时目录独立复算全部15份JSON/CSV/表文件，逐字节一致；额外重放01的只读交付核验并比较结果JSON相同。不会调用真实worker或覆盖历史模型/运行文件：

```bash
/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_state_diagnostic_evaluation_checks_20261003_v1/replay.py > docs/itm_rework/protocol/b0_state_diagnostic_evaluation_checks_20261003_v1/replay.txt 2>&1
```

评价按已完成的最终group/receipt/父账本纳入全部48/192，原始branch sidecar的pending不是最终失败。144项锚点精确重算；192 V分支、192 V对比、384 Q对比、192逐来源V/Q并列完整保留。无额外模型结果或新动作审阅。
