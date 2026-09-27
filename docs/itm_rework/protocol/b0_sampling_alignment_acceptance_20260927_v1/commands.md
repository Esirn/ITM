# 02 独立 CPU 软件验收命令

工作目录 `/home/a200/0proj/ITM`；读取方法提交 `f9b3cb18053892dd9e66dbdf31085883a99b76d1`。

```bash
PYTHONPATH=src bash docs/itm_rework/protocol/b0_sampling_alignment_acceptance_20260927_v1/regression.sh
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/accept_b0_sampling_alignment.py --output docs/itm_rework/protocol/b0_sampling_alignment_acceptance_20260927_v1/acceptance.json --prepare-dir /tmp/itm_sampling_alignment_acceptance_20260927_v2 > docs/itm_rework/protocol/b0_sampling_alignment_acceptance_20260927_v1/logs/acceptance.txt 2>&1
```

回归文件列表与原01最终联合回归相同，只将证据输出改到02目录。CPU fixture 使用微型替代模型，不加载真实预训练模型执行前向；队列 fixture 的预检和进程启动被替换，临时假授权不是实际授权。

独立审计从原始30 Hz NPZ重建20 Hz acceleration、SO(3) orientation；按冻结变换生成paired/static/reversed/numeric-zero，重算398帧train-only统计，检查全部480 Q＋60 S1输入、caption、GT、mask与原始O[0]参考。prepare期间禁止nn.Module前向、GPU查询和运行器预检/启动，只在CPU读取既有R checkpoint。743文件核对与manifest独立重建不依赖01的验证结论。

首次审计开发运行完成manifest重建后，在Q槽位访问仅S1具有的`gt_path`字段而报KeyError；保留`logs/audit_development_attempt.txt`和临时v1目录。修正02审计脚本为从冻结caption产物索引读取GT、分别核对Q的`target_feature_path`与S1的`gt_path`，在新v2目录完整重跑。该开发错误未涉及方法代码、模型前向或GPU，不计为01软件失败。复现时请选择不存在的prepare目录与输出文件。
