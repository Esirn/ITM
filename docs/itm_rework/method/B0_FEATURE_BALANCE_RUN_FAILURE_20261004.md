# 01 → 02：F 单次运行在 GPU 预检失败后停止

用户对授权申请 `72a01e9` 回复“批准”后，01于2026-10-04执行一次已验收F入口。**指定GPU的首项 `nvidia-smi` 查询返回退出码9，父入口退出码1；训练worker未启动，模型调用0。** 从入口预检前计时起耗时 **1.006989秒**。本次授权已消耗，没有重试、补查GPU、换卡、续训或采样。

## 绑定身份与证据

- 实现 `963a132`，独立验收 `e0999b7`；软件未修改。
- 最终outputs v2 manifest SHA256：`db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`。
- F package SHA256：`a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`。
- GPU UUID：`GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
- 原目录：`outputs/itm_rework/method/feature_balance_authorized_20261004_v1`，仅有`runtime/status.json`，保留原样。
- [实际授权记录](B0_FEATURE_BALANCE_AUTHORIZATION_20261004.json)、[历史执行命令](feature_balance_run_20261004_v1/commands.md)、[父日志](feature_balance_run_20261004_v1/logs/parent.log)、[原始账本逐字节副本](feature_balance_run_20261004_v1/runtime/status.json)、[执行摘要](feature_balance_run_20261004_v1/execution_summary.json)。

## 失败位置与解释边界

父入口已完成文件和运行环境检查，取得指定GPU的队列锁，随后调用既有`preflight_gpu()`。首项命令为：

```text
nvidia-smi --id=GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004 --query-gpu=uuid,memory.total,memory.free --format=csv,noheader,nounits
```

`subprocess.run(..., check=True)`抛出`CalledProcessError`，返回码9。尚未解析UUID/显存字段，也未执行第二项compute-process查询或任何worker启动。既有helper捕获stdout/stderr，但异常前没有持久化它们；**现有证据不足以确定具体设备/驱动原因**，不能据此断言GPU被占用或显存不足，也不能用事后查询替代当时证据。

## 完整失败分母

| 项目 | 计划 | 完成 | 停止后未执行 |
| --- | ---: | ---: | ---: |
| updates | 200 | 0 | 200 |
| exposures | 600 | 0 | 600 |
| samples | 60 | 0 | 60 |
| traces | 240 | 0 | 240 |
| software checks | 48 | 0 | 48 |

账本无active job，completed receipts为空、job counts为空；未产生worker context、receipt、checkpoint、损失日志、采样或trace。模型调用0依据是父入口在首个job之前失败；原始账本没有worker计数栏，归档未伪造计数。没有可报告的worker显存峰值。

[CPU交付核验](feature_balance_run_20261004_v1/delivery_integrity.json)通过：1,145份绑定文件hash一致，manifest/package/授权身份一致，原始与归档账本逐字节一致，全部计划ID和失败状态与冻结表对齐。[核验脚本](feature_balance_run_20261004_v1/verify_delivery.py)仅使用标准库读取文件，不导入模型或查询GPU。没有改动实现或重复相同软件回归；此前155项CPU验收结果继续有效，但不代表真实GPU行为通过。

## 交给02

下一责任方02核对本次失败交付和完整分母；本批没有模型产物可评价，F效果和B0有效性仍未认证。后续尝试需新的唯一目录和新单次授权，本次及历史授权均不可复用。

若下一轮需要查明预检失败原因，应先安排有限的GPU诊断，或改进预检原始输出保留并经CPU验证/增量验收后绑定新manifest；本轮未修改已验收入口，也未放宽任何预检或运行限制。
