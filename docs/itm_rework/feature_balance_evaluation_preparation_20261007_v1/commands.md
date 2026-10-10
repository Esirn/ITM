# CPU 评价准备的复现命令

在仓库根目录执行。以下命令不查询 GPU，不运行真实模型，也不创建实验运行目录。评价输出目录须不存在，避免覆盖已归档证据。

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_feature_balance_evaluation.py tests/test_canonical_b0_evaluation.py

OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_feature_balance.py --output /tmp/itm_f_cpu_replay_fresh

/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/feature_balance_evaluation_preparation_20261007_v1/verify_preparation.py

node docs/itm_rework/feature_balance_evaluation_preparation_20261007_v1/review_smoke.js outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1/review
```

最终留档入口实际使用 `--output outputs/itm_rework/protocol/feature_balance_evaluation_preparation_20261007_v1`，日志见 replay.txt。来源 hash 与输出 hash 位于该目录 sources.json / artifact_index.json；只读脚本检查当前留档与历史 U 报告逐字段一致。

真实 F 运行完成后，使用同一 CPU 入口读取实际运行目录，输出到新评价目录。完成审阅后，将全部 120 行 hidden-ID notes 通过 `--review-notes` 传入另一个新输出目录；入口保存提交 notes 的 hash 后才写 decoded_review.json。没有 notes 时行为项保持 missing，不能产生局部支持结论。本次没有真实 F，也没有提交行为 notes。

入口只使用本机已保存数组；Torch 仅用于已有 HumanML→关节 CPU 解码，禁止模块前向、CUDA 查询/初始化和子进程。没有反序列化模型 checkpoint。合成运行/授权样式 JSON 仅在 pytest 临时目录出现，不能用于真实运行授权。
