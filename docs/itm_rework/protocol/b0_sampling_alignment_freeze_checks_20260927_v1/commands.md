# 02实际执行：仅CPU协议构造和测试

读取d2d28fb；工作目录 `/home/a200/0proj/ITM`。

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/freeze_b0_sampling_alignment.py
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_b0_sampling_alignment_protocol.py --junitxml=/tmp/itm_sampling_alignment_tests.xml
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/freeze_b0_sampling_alignment.py --output /tmp/itm_sampling_alignment_freeze_replay_20260927_v1
```

首轮及重建stdout、12项测试日志/XML另存本目录。逐文件比较两个目录文件名、bytes、SHA256，14/14一致，详见`replay_verification.json`。再次构造必须改用不存在的目录；构造器拒绝覆盖包，不会调用或重跑01提案生成器。

构造器只导入Python标准库和NumPy；未导入torch/MDM/CLIP、未加载checkpoint对象或运行模型。checkpoint仅流式读字节核验hash。真实训练、推理前向、采样、GPU查询均为0；未创建运行授权。此处12项是协议/代数测试，不是新方法实现验收或效果证据。
