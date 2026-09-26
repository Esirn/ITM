# 01 → 02：授权 B0 实际运行交付（2026-09-26）

读取 `030c191`，执行其独立验收通过的 `1d44adc` 方法代码。本次 **200/200 updates 与 36/36 条完整采样全部完成**，顶层运行器退出码 0，无超时、OOM、非有限值或失败重试。契约仍为 `itm-rework-v0.3-draft`，适用范围仅为预先统一坐标的合成 B0。

用户在一次 B0 的明确额度请求后回复“可以”，随后“continue”。实际授权保存在 [B0_AUTHORIZATION_20260926.json](B0_AUTHORIZATION_20260926.json)，绑定原预算/计划 hash、唯一运行目录和 GPU UUID；这次授权已经执行，不能用剩余额度自行重跑。没有修改 02 原文件中的历史 pending 字段。

## 实际预算和设备

| 项目 | 实际结果 | 上限 |
| --- | --- | --- |
| 训练 | attempted=completed=200；约 33.65 秒 | 200 updates / 3600 秒 |
| 固定采样 | 36 completed，0 failed，0 not-run；约 526.33 秒 | 36 条 / 共用 10800 秒 |
| 总时长 | 约 561.13 秒（9.35 分钟，0.156 GPU 小时） | 14400 秒 |
| 去噪与分支样本前向 | 36,000 / 144,000 | 36,000 / 144,000 |
| allocated 显存峰值 | 737,469,952 bytes，约 0.687 GiB | 24 GiB |
| GPU | 单 RTX 4090，GPU 0，UUID `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004` | 单 GPU |

时间取父进程阶段起点至最后子进程持久化 receipt，父进程末尾哈希核验/退出另有少量开销；观察到退出码 0 时距启动约 561.7 秒。运行器以原绝对 deadline 约束各阶段，未重置或延长预算。显存沿用已验收的 PyTorch allocated 口径，不等同于 nvidia-smi 总占用。

当前沙箱中 nvidia-smi 无法访问驱动；沙箱外只读查询正常，因此实际命令在沙箱外执行。正式运行器预检记录：指定 GPU 总显存 24564 MiB，空闲 24124 MiB，已有 compute processes=0。所有子进程绑定同一 UUID。无 CPU 回退或换卡；`OMP_NUM_THREADS=2`、`PYTHONUNBUFFERED=1`。运行代码没有修改，启动前的已验收代码/依赖 hash 核对见 `b0_run_20260926_v1/code_identity.json`。

## 交付入口

运行目录：`outputs/itm_rework/method/canonical_b0_authorized_20260926_v1/`。

- 最终模型：`model.pt`，SHA256 `44d2c3307f27bca2c79ab7d367890f968f78777168c3c37a306a7bf0560d6ac2`。`execution.final_update_200=true`，训练来源记录与已验收输入报告完整相等；使用该报告绑定的 train-only IMU normalization。
- 已发布的 100 份 epoch checkpoint 全部保留；固定验证只使用最终 `model.pt`，没有根据验证结果选择 epoch。
- 样本：`samples/<case_id>.npz` 及同名 `.json`；3 val × 2 配置 × 6 条件，按 02 原计划顺序执行，没有替换或过滤槽位。
- `runtime/status.json` 保存全部 36 槽位、原 argv、授权、预检、截止时间和 artifact hash；训练与 36 条采样各有 context、receipt 和 log。
- [delivery_integrity.json](b0_run_20260926_v1/delivery_integrity.json) 提供逐槽位路径/hash/形状/耗时、计数和显存摘要；`runtime_manifest.json` 绑定原运行记录与交付副本。全部 runtime 文件副本在 `b0_run_20260926_v1/runtime/`，模型与样本实体留在 outputs，不写入 git。
- 原启动命令保存在 `b0_run_20260926_v1/launch.sh`，仅为历史记录，不可再次执行同一次授权。

本轮方法侧 CPU 交付检查已通过：全部样本数值数组有限；NPZ 内 metadata 与独立 JSON 相等；模型和产物 hash 与 receipt 相等；输出长度/形状对应固定 case；全部 36 条绑定同一最终 checkpoint；6 个 source/config 组内各六条件的完整随机计划相等。最终 checkpoint 来源与原已验收训练输入相等。复核脚本可只读检查产物并重建相同摘要：

```bash
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python \
  docs/itm_rework/method/b0_run_20260926_v1/verify_delivery.py
```

这是运行与交付完整性检查，不是新的独立效果验收。方法代码未改，因此没有重复跑已由 02 通过的 166 项软件回归。

## 下一步由 02 独立评价

请按原固定 36 槽位和原阈值评价全部输出，核对统一 checkpoint、输入/随机计划与 joint-proxy 指标，并保留所有负结果。当前完整成功率仅指作业完成，不代表动作质量或控制收益。01 本轮不新增训练、消融或 B/C 作业。

真实设备标定、严格生成 IMU 闭环、相容编辑能力和模型有效性均未因此认证；没有声明 B0 泛化有效、优于基线或满足严格传感器一致性。
