# 已消耗授权的历史诊断命令

以下入口已执行一次，**不能重跑**；其唯一输出目录已存在，脚本也会拒绝覆盖。

```bash
/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/method/feature_balance_gpu_diagnostic_20261007_v1/execute_once.py
```

脚本读取本次明确批准记录并校验申请 hash，内部仅调用一次：

```text
nvidia-smi --id=GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 --query-gpu=uuid,memory.total,memory.free --format=csv,noheader,nounits
```

`subprocess.run(..., capture_output=True, timeout=10, check=False)`；没有 compute-process 查询，没有模型加载或执行。stdout/stderr 按原始字节保存，status.json 保存各项判定。docs 中副本与 outputs 原始文件逐字节一致。

verification.json 来自随后标准库文件/hash 检查：批准/请求/脚本 hash、精确命令、原始输出解析、三项条件、stderr 为空及较早失败账本未变均通过；没有重复运行查询来验证。
