# 实际命令与执行记录

工作目录 `/home/a200/0proj/ITM`；conda 环境 itm；全部 CPU，GPU 作业数 0。
输出均为独立路径；原命令再次执行会拒绝覆盖，请使用新 run ID。预算见 cpu_budget.json。

```bash
conda run -n itm python -m pytest tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_random_plan.py -q
conda run -n itm python scripts/audit_mdm_method.py --output docs/itm_rework/method/checkpoint_audit_20260924_v1.json --full-sampling
conda run -n itm python scripts/validate_mdm_identity.py --output docs/itm_rework/method/identity_overfit_20260924_v1.json
```

第一轮新增测试出现一项异常类型不一致（cache shape mismatch 在 shape 检查前触发 RuntimeError），
已把长度校验前移并保持 cache 清理；最终测试输出见 logs/regression.txt。
真实 checkpoint 审计成功；严格 null 单步及完整 1000 steps 均与原始输出完全一致，详细数值见 audit.md。
合成非对称过拟合成功，不能用作完整生成效果。

真实 MDM 一次训练 update 已执行：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 conda run -n itm python scripts/train_mdm_imu_control.py \
  --mdm-root outputs/mdm/runtime_root \
  --mdm-checkpoint outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt \
  --mdm-args outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json \
  --manifest outputs/manifests_full/train.jsonl \
  --standard-imu-manifest outputs/manifests_full/train_standard_imu.jsonl \
  --output /home/a200/0proj/ITM/outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/model.pt \
  --epochs 1 --batch-size 2 --max-records 1 --device cpu --allow-unsynchronized-diagnostic
```

实际配对训练记录 `000004`，head/wrists 两项，seed 1234。CLI 默认 B0、diffusion-only。
日志：epoch=1 loss=0.030115 diffusion=0.030115 trajectory=0.002957 velocity=0.057311 jerk=4087.534180。
velocity/jerk 权重为 0；它们使用 seconds 单位，不能与旧 frame 单位直接比较数值。
没有隐式恢复旧 checkpoint。新 checkpoint 与 epoch001 均落盘。

保存 checkpoint 后的真实采样命令：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 conda run -n itm python scripts/sample_mdm_imu_control.py \
  --control-checkpoint outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/model.pt \
  --mdm-args outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json \
  --standard-imu-manifest outputs/manifests_full/train_standard_imu.jsonl \
  --spec docs/itm_rework/method/b0_smoke_spec.json \
  --output /home/a200/0proj/ITM/outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/samples.npz \
  --device cpu --guidance-mode factorized --branch-execution batched \
  --text-scale 1 --imu-scale 1 --joint-scale 1
```

首次 spec 的 motion_id 无 IMU 缓存，采样前失败，见 b0_sample_failure_v1.json。
改正为训练实际来源 ID 后完整运行成功。paired/null 均保留，未按质量过滤。

B 训练使用同一训练 CLI；在 02 提供合法 train witness 后添加
`--response-manifest <path> --response-loss-weight <frozen-weight> --response-proxy-weight <frozen-weight>`。
无权重、协议和见证时未启动此实验；不是已执行命令。
`configs.json` 列出 null-only/identity-only/combined/B0/FiLM/目标函数消融，CLI 用相应 kebab-case 参数。
正式 validation、B 的完整采样收益和 C 均未运行。

## v0.2 交接后的输入适配

新增实际命令和版本差异见 FOLLOWUP_V02.md。输入脚本成功检查 6 个 train/config 项、6 条 val、2 种静止输入，模型前向 0。
48 项相关回归通过，日志 logs/input_followup_v2.txt。没有重新训练或重新完整采样，首批命令/产物保持历史版本。
