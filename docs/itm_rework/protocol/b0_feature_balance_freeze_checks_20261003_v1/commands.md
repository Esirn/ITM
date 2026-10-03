# 02 F冻结复现记录

工作目录`/home/a200/0proj/ITM`，读取`2b8a842`。仅文件、NumPy和GT绘图；没有模型框架导入、checkpoint张量加载、前向、训练、采样、GPU查询或授权。初次开发构建遇到旧槽位可选original_case_id缺失，在输出目录创建前停止；日志保留development_optional_case_id.txt，改用完整历史case后缀后成功，未改变旧文件。

首次成功构建（现有冻结目录拒绝覆盖，再次构建请指定不存在的--output）：

```bash
/home/a200/miniconda3/envs/itm/bin/python scripts/freeze_b0_feature_balance.py > docs/itm_rework/protocol/b0_feature_balance_freeze_checks_20261003_v1/build.txt 2>&1
```

相关解析/协议测试9 passed，0 failed/skipped；不是尚未实现F的模型软件验收：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_feature_balance_protocol.py --junitxml=docs/itm_rework/protocol/b0_feature_balance_freeze_checks_20261003_v1/tests.xml > docs/itm_rework/protocol/b0_feature_balance_freeze_checks_20261003_v1/tests.txt 2>&1
```

独立临时目录重建01提案7文件、02冻结16文件，全部逐字节相同；运行时拒绝torch/tensorflow/jax及子进程，不执行GPU命令：

```bash
/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_feature_balance_freeze_checks_20261003_v1/replay.py > docs/itm_rework/protocol/b0_feature_balance_freeze_checks_20261003_v1/replay.txt 2>&1
```

GT136帧全帧曲线与12个固定姿态，只用于核对可见裁片边界，不渲染F候选结果。绘图曾由matplotlib自动采用/tmp缓存（用户配置目录不可写），无计算结果影响；重放可显式指定缓存：

```bash
MPLCONFIGDIR=/tmp/itm02_feature_balance_mpl /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_feature_balance_freeze_checks_20261003_v1/render_gt.py
```

GT-only窗口是描述性审阅参照，不是自动分类或warp参数。新旧F/U正式审阅仍使用冻结120行顺序、每段10个均匀姿态＋全帧曲线/完整序列。不会要求裁片外的开始迈步/最后站直，旧完整动作负例不追改。
