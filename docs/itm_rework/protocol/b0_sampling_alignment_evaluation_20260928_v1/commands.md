# 02 保存数组独立评价命令

工作目录 `/home/a200/0proj/ITM`，方法交付 `f6da7a8066384700b06b427845aab969b20695a3`。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_sampling_alignment.py evaluate --output /tmp/itm_alignment_evaluation_replay_20260928_v1
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_sampling_alignment.py render
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_canonical_b0_evaluation.py tests/test_b0_correction_evaluation.py tests/test_b0_sampling_alignment_evaluation.py --junitxml=docs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1/logs/tests.xml
/home/a200/miniconda3/envs/itm/bin/python scripts/summarize_b0_sampling_alignment.py
```

正式评价输出保存在本目录。临时目录完整重跑输出逐文件核对/复制到此，见`replay_verification.json`；已存在结果不覆盖，复现时使用新的`--output`目录。绘图产物在`outputs/itm_rework/protocol/b0_sampling_alignment_evaluation_20260928_v1/review`，20页路径/hash见`review_rendering.json`。按冻结120行全部审阅后记录`review_notes_hidden.txt`，再运行summarize解码条件；此前已知旧结果和S001对应关系，不称为完全盲评。

新脚本开发中曾有Python3.10星号索引语法错误（未运行评价），以及在Q阶段只构造cosine原betas、遗漏全步SpacedDiffusion累计alpha重建步骤的hash断言失败。两条日志分别保留`development_syntax.txt`、`development_schedule.txt`。按绑定runtime源码补齐同一identity timestep map的float64计算后，betas hash一致，所有Q保存x_t与独立重建最大差为0；不改变模型、数据、阈值或已有输出。最初已完成的样本/对照结果与最终完整CPU重跑逐字节相同；旧R60条指标与前批评价完全一致。

最终解析测试18 passed（既有指标6、既有聚合/噪声6、新增分解/差值/identity-respacing6）；测试无模型调用。全部评价主体运行于禁止nn.Module前向和CUDA查询的patch上下文，使用numpy及纯CPU RIC解码。未执行preflight、模型加载器、训练/采样入口或GPU命令，未创建新授权。时间、显存来自已有receipt。未重复258项方法软件回归。
