# 02执行记录（2026-09-27）

读取43af316；全程CPU文件评价，无模型前向、训练、采样或GPU查询。工作目录为仓库根；Python为 `/home/a200/miniconda3/envs/itm/bin/python`，Torch相关命令 `OMP_NUM_THREADS=2`、`PYTHONPATH=src`。

```bash
# 重跑01只读交付检查，结果另存02，不覆盖01
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/correction_run_20260927_v2/verify_delivery.py > /tmp/correction_delivery_independent.json
# 冻结120顺序、20页隐藏标签绘图
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_correction.py render
# 独立探针数组/噪声/激活/reductions/624对照重算
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_correction.py probes
# 全部动作逐条笔记完成并记录hash后，解码身份，再运行数值评价
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/evaluate_b0_correction.py samples
# 分析测试，不重复未修改的方法联合回归
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_correction_evaluation.py tests/test_canonical_b0_evaluation.py --junitxml=docs/itm_rework/protocol/b0_correction_evaluation_20260927_v1/tests.xml
# 02另行核对授权、上下文、123receipt、计数和历史时钟/显存
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/audit_b0_correction_execution.py
```

首轮绘图进程完成16页后返回143（SIGTERM，原因未确定），只续绘缺少的17–20页；没有重新模型生成或改动作文件。两段日志保留。脚本遇已存在图片只登记hash，不覆盖；重复作图应通过 `--images` 指定新目录。

运行receipt审计首次将内嵌context字典当成路径，解析阶段停止、未写审计结果；改为核对内嵌context与单独文件后通过，两个日志分别保留。这是02解析器开发问题，不是01运行失败，不更改480/120成功/失败分母。

最终样本重放使用新目录 `/tmp/itm_correction_eval_replay_20260927_v1`：`unittest.mock.patch` 禁止 `torch.nn.Module._call_impl` 及CUDA `_lazy_init/is_available/device_count/get_device_properties/synchronize/current_device/set_device`，在此上下文调用 `samples()`。原样本JSON所有既有字段完全相同，最终脚本另增显式arm/split分母；contrasts/training/historical_comparisons/per_case.csv逐字节相同。结果记录于 `sample_replay.json`。重放不是第二次模型实验。

再次复算应给 `--output` 新目录；samples/probes拒绝覆盖其已有结果。输入与模型字节仍在原outputs，所有原文件不改写。`tables.md` 由samples/contrasts/probe_contrasts中的已保存聚合字段格式化为小数，JSON为完整精度依据。
