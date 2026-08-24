# ITM Next Counterfactual Experiments

本文档对应 `scripts/run_mdm_control_experiment_suite.py`。它把下一阶段
Text-IMU 互补实验做成可重复的离线 runner：先生成每组实验的 `spec.json`，
可选调用 MDM sampler，再写出浏览器可读 JSON 和 proxy 指标。

## 目标

当前阶段只验证两个核心现象：

1. **同一模糊文本，不同 IMU**：固定 `a person walks`，换测试集 IMU，
   观察生成动作的速度、步频、轨迹和摆动是否随 IMU 改变。
2. **同一头部 IMU，不同文本**：固定 head-only IMU，换 walking 相关文本，
   观察文本是否补全摆臂、步幅、转向等全身语义，同时保持头部轨迹。

`outputs/mdm_control/stage1_full_pilot_v2.pt` 是默认 checkpoint。实验不改
模型权重。

MDM 资产默认读取：

```text
outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/
```

如果该目录不存在，可从本地 zip 解压一次：

```bash
mkdir -p outputs/mdm/checkpoints_extracted
unzip -q /home/a200/0proj/datasets/mdm-need/motion-diffusion-model/save/humanml_trans_enc_512.zip \
  -d outputs/mdm/checkpoints_extracted
```

官方 MDM 代码还需要在运行目录下看到 `body_models/smpl/SMPL_NEUTRAL.pkl`。
不修改外部 relatedworks 目录的做法是创建一个 ignored runtime root：

```bash
conda run -n itm python scripts/prepare_mdm_runtime_root.py
```

## 快速检查

只写一个 matrix 实验的 request/spec，不占 GPU：

```bash
conda run -n itm python scripts/run_mdm_control_experiment_suite.py \
  --suite matrix \
  --matrix-count 1 \
  --max-runs 1 \
  --output-root /tmp/itm_experiment_plan_smoke
```

小规模真正采样，用于确认 GPU、checkpoint 和输出链路：

```bash
conda run --no-capture-output -n itm python scripts/run_mdm_control_experiment_suite.py \
  --suite matrix \
  --matrix-count 1 \
  --max-runs 1 \
  --execute \
  --device cuda:1 \
  --output-root outputs/mdm_control/experiments_smoke
```

## 推荐正式命令

生成完整计划但不采样：

```bash
conda run -n itm python scripts/run_mdm_control_experiment_suite.py \
  --split test \
  --output-root outputs/mdm_control/experiments
```

执行完整 suite：

```bash
conda run --no-capture-output -n itm python scripts/run_mdm_control_experiment_suite.py \
  --split test \
  --execute \
  --skip-existing \
  --device cuda:1 \
  --output-root outputs/mdm_control/experiments
```

如果显存或时间紧张，优先分阶段执行：

```bash
# 1. 先跑同文本换 IMU，验证 IMU 控制
conda run --no-capture-output -n itm python scripts/run_mdm_control_experiment_suite.py \
  --suite same_text --execute --skip-existing --device cuda:1

# 2. 再跑同头部 IMU 换文本，验证文本补全
conda run --no-capture-output -n itm python scripts/run_mdm_control_experiment_suite.py \
  --suite same_imu --execute --skip-existing --device cuda:1

# 3. 最后跑 guidance sweep，选择默认展示参数
conda run --no-capture-output -n itm python scripts/run_mdm_control_experiment_suite.py \
  --suite sweep --execute --skip-existing --device cuda:1
```

## Guidance兼容模式

历史Stage-1/2/2b/3/4实验默认使用：

```bash
--guidance-mode legacy --branch-execution sequential
```

新四分支消融使用：

```bash
--guidance-mode factorized --branch-execution batched --joint-scale 1.0
```

`factorized`显式计算unconditional、Text-only、IMU-only和Text+IMU。它会改变推理定义，因此不能静默替换历史结果。`batched`需要缓存条件，显存不足时改用`sequential`。可通过`--no-condition-cache`做数值兼容诊断，但不能与`batched`同时使用。

## 严格Loss消融

只生成执行计划：

```bash
conda run -n itm python scripts/run_mdm_loss_ablation.py
```

正式训练后复测：

```bash
conda run --no-capture-output -n itm python scripts/run_mdm_loss_ablation.py \
  --execute-train --execute-suite --skip-existing --device cuda:1
```

四个变体都从`stage1_full_pilot_v2.pt`开始，避免用不同训练阶段的checkpoint代替严格loss ablation。

## SMPL与虚拟IMU验证

先对GT执行round-trip：

```bash
conda run --no-capture-output -n itm python scripts/evaluate_generated_virtual_imu.py \
  --results path/to/results.npz --motion-key gt \
  --output path/to/gt_roundtrip.json --device cuda:1
```

只有`gt_roundtrip.json`中`validation.passed=true`，才允许把它传给生成动作评价：

```bash
conda run --no-capture-output -n itm python scripts/evaluate_generated_virtual_imu.py \
  --results path/to/results.npz --motion-key motion \
  --validation-summary path/to/gt_roundtrip.json \
  --output path/to/generated_virtual_imu.json --device cuda:1
```

当前HumanML关节位置无法可靠恢复骨轴twist，初步GT round-trip没有通过orientation门槛。因此正式论文仍应把现有加速度指标称为joint-derived proxy。

## 输出结构

每个 run 目录包含：

- `request.json`：实验元数据、scale、device、样本 ID。
- `spec.json`：传给 `sample_mdm_imu_control.py` 的条件列表。
- `results.npz`：采样输出、GT、IMU、metadata。
- `result.json`：浏览器可读的 panel/IMU JSON。
- `metrics.json`：轻量 proxy 指标。
- `summary.md`：指标表格摘要。
- `generation.log`：sampler stdout/stderr。

suite 根目录还会写：

- `suite_index.json`
- `SUMMARY.md`

## 当前指标

这些指标是定性实验的 proxy，不等同于真实 IMU orientation error：

- `root_trajectory_error_m`
- `head_trajectory_error_m`
- `wrist_trajectory_error_m`
- `active_sensor_trajectory_error_m`
- `root_relative_motion_error_m`
- `jerk_ratio`
- `arm_swing_proxy_m`
- `root_travel_m`
- `step_frequency_hz`
- `pairwise_root_relative_distance_m`

优先看趋势：

- Text+IMU 是否比 Text-only 更贴近 head/wrist trajectory。
- 同文本换 IMU 是否改变 root travel、step frequency 和 motion distance。
- 同 head IMU 换文本时，arm swing proxy 是否随文本合理变化。
- jerk 是否没有明显恶化。

## 解读规则

- 训练集样例只能作为 debug，不作为论文证据。
- 当前首选 `test` split；必要时用 `val` 复核。
- 若同文本换 IMU 成立、同头部 IMU 换文本仍不稳定，下一步应改训练目标，
  加入 head/wrist trajectory consistency、velocity consistency 和 jerk
  regularization，而不是立刻引入完整 ControlNet、cross-attention 或 SMPL
  decoder。
