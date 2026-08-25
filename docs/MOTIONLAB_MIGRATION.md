# MotionLab 复现与 ITM 迁移

## 范围

本次只复现和迁移 MotionLab 的 Text-to-Motion 生成能力。风格迁移、动作编辑、补间、重建及其他输入模态均不进入 ITM 主线。

## 已确认资源

- 源码：`/home/a200/mount/a40/relatedworks/MotionLab`
- 官方统一模型：`checkpoints/motionflow/motionflow.ckpt`
- 文本编码器：本地 CLIP ViT-L/14
- 动作表示：HumanML3D 263D，20 FPS，最长 196 帧
- 输出：由 `recover_from_ric` 解码的 22 关节三维位置
- 独立环境：conda `rfmotion`，PyTorch 2.1.1 + CUDA 11.8

源码目录位于只读 sshfs 挂载。ITM 不修改该目录，也不依赖 MotionLab 的渲染器。当前审计对应 MotionLab commit `98cc88c8e31e43be9f94c2d1edf4c21e4715c683`。

## 最小复现链

`scripts/sample_motionlab_text.py` 只保留官方生成路径中的必要组件：

1. 读取官方配置、CLIP、MotionFlow denoiser 和 flow-matching scheduler。
2. 跳过风格编码器、内容编码器和评估器等无关初始化。
3. 用文本、动作长度和随机种子生成归一化 263D motion。
4. 反标准化并恢复为 `[B,T,22,3]` joints。
5. 保存 NPZ 与包含版本、guidance、耗时和设备的 JSON 元数据。

该精简仅减少无关模块，不改变 checkpoint、文本编码、denoiser、采样器或 CFG 公式。

示例：

```bash
cat > /tmp/motionlab_request.json <<'JSON'
{"text":["a person walks forward"],"lengths":[120],"seed":1234}
JSON

conda run --no-capture-output -n rfmotion python scripts/sample_motionlab_text.py \
  --motionlab-root /home/a200/mount/a40/relatedworks/MotionLab \
  --checkpoint /home/a200/mount/a40/relatedworks/MotionLab/checkpoints/motionflow/motionflow.ckpt \
  --request /tmp/motionlab_request.json \
  --output outputs/motionlab/text_only_smoke/results.npz \
  --metadata outputs/motionlab/text_only_smoke/metadata.json \
  --device cuda:0
```

`MotionLabBackbone` 通过 subprocess 调用上述脚本，使 MotionLab 与 ITM 可以保持各自的依赖环境。它实现统一接口中的完整 `sample`/`decode_motion`；逐步 `predict` 和单独 `encode_text` 不跨进程暴露。

生成结果可用 ITM 的统一骨架渲染器查看：

```bash
conda run --no-capture-output -n itm python scripts/render_motionlab_results.py \
  --results outputs/motionlab/text_only_smoke/results.npz \
  --metadata outputs/motionlab/text_only_smoke/metadata.json \
  --output outputs/motionlab/text_only_smoke/motion.gif
```

## IMU 迁移边界

MotionLab 原生的 `hint` 是 22 个关节的三维轨迹，共 66 维，并不是原始 IMU。不能把其 Text+Hint 结果直接称为 Text+IMU。

迁移分两步：

1. 先把已有标准 IMU 对齐样本对应的 active-joint trajectory 输入 Text+Hint，验证 MotionFlow 的稀疏轨迹控制上限。该实验是 oracle trajectory-hint baseline。
2. 再训练 `12D IMU + sensor mask -> MotionLab hint/control tokens` 的适配器，冻结 MotionFlow 和 CLIP。只有第二步才是 MotionLab 版 ITM。

第一步可以判断更强生成 backbone 是否能改善观感；第二步负责证明控制信号确实来自 IMU，不能混淆两者。

最小入口已支持第一步。`--trajectory-hints` 接收包含 `joints [B,T,22,3]` 的 NPZ；NPZ 可直接提供同 shape 的布尔 `mask`，否则 request JSON 必须提供 `active_joints`（head 为 15，wrists 为 20/21）。输出元数据会把模式记录为 `text_trajectory_hint_oracle`。

## IMU adapter

`train_motionlab_imu_adapter.py` 使用现有标准 IMU 缓存训练：

```bash
conda run --no-capture-output -n itm python scripts/train_motionlab_imu_adapter.py \
  --train-manifest outputs/manifests_full/train.jsonl \
  --train-imu-manifest outputs/manifests_full/train_standard_imu.jsonl \
  --val-manifest outputs/manifests_full/val.jsonl \
  --val-imu-manifest outputs/manifests_full/val_standard_imu.jsonl \
  --sensor-configs head,wrists --device cuda:1 \
  --output outputs/motionlab/imu_adapter/adapter.pt
```

网络将每帧六槽 `12D IMU` 投影为 256 维，加入 sensor/temporal encoding，经过两层时序 Transformer 后预测 MotionLab 的 66D 归一化关节轨迹。损失只作用于 active sensor joint，并包含位置与一阶速度。MotionFlow 与 CLIP 不参与该阶段训练。

`predict_motionlab_imu_hints.py` 将 checkpoint 和标准 IMU 序列转换为 `sample_motionlab_text.py --trajectory-hints` 可读取的 NPZ。调用采样器时应在 request JSON 中写入 `"hint_source":"imu_adapter"`，元数据会将其记录为 `text_imu_adapter`，避免与 oracle GT trajectory 混淆。该链路的控制来自 IMU 预测而非 GT trajectory，但当前仍是两阶段适配器，并非对 MotionFlow 的端到端联合微调。

初步测试表明，两阶段绝对轨迹回归在 1333 条 test motion 上误差较大：head 为 `0.631 m`，wrists 为 `0.709 m`。原因是稀疏 IMU 的加速度和方向不能唯一恢复初始位置、初始速度与全局平移。因此该方案保留为失败诊断，不作为 MotionLab 主迁移模型。

## Direct IMU control

`train_motionlab_direct_imu_control.py` 是当前主迁移方向：

- 冻结 MotionFlow 和 CLIP。
- IMU encoder 直接输出 66D learned control tokens，而不声称这些 token 是关节坐标。
- token 进入 MotionLab 已训练的 hint embedding 与 Text+Hint attention。
- 用 MotionFlow 原始 flow-matching denoising loss 训练 IMU encoder。
- 文本和 IMU 分别以 0.1 概率 dropout。

16 条数据的反向传播 smoke 已通过，flow-matching loss 为 `0.2235`。推理时先用 `predict_motionlab_direct_imu_control.py` 生成 token NPZ，再传给 `sample_motionlab_text.py --control-tokens`；输出模式记录为 `text_imu_direct_control`。

完整训练使用 7009 条 train、434 条 val、head/wrists、batch 8 和 5 epochs。最后一轮 train loss 为 `0.1317`。早期 val 评估没有固定 diffusion timestep/noise，数值在 `0.1317-0.1601` 间波动，因此不能仅按该 val 最小值可靠选模型；后续代码已固定验证 RNG，已有五个 epoch checkpoint 将通过固定条件可视化和复评选择。

首个 test 诊断样例 `004822` 中，相同文本、seed 和长度下，head active-joint GT error 从 Text-only 的 `1.102 m` 降至 `0.870 m`，wrists 从 `1.134 m` 降至 `0.839 m`。该结果未做轨迹对齐且只有单样本，只说明 direct control 通道产生了与目标一致的初步信号，不作为正式结论。四路 GIF 位于 `outputs/motionlab/direct_imu_control/qualitative/004822_four_way.gif`。

```bash
conda run --no-capture-output -n rfmotion python scripts/train_motionlab_direct_imu_control.py \
  --motionlab-root /home/a200/mount/a40/relatedworks/MotionLab \
  --motionlab-checkpoint /home/a200/mount/a40/relatedworks/MotionLab/checkpoints/motionflow/motionflow.ckpt \
  --manifest outputs/manifests_full/train.jsonl \
  --imu-manifest outputs/manifests_full/train_standard_imu.jsonl \
  --sensor-configs head,wrists --batch-size 8 --epochs 5 --device cuda:1 \
  --output outputs/motionlab/direct_imu_control/adapter.pt
```

这条路径只使用 Text、IMU 和 motion generation，不包含 MotionLab 的其他输入模态或任务。

## 后续验收

- 固定提示词和种子生成 Text-only 样例，并确认无 NaN、长度和关节维度正确。
- 与官方 demo 的同条件输出核对，允许因批处理和随机数初始化导致的差异，但模型路径必须一致。
- 接入现有 ITM 可视化，人工比较 MDM 与 MotionLab 的语义匹配、抖动和动作自然度。
- 完成 oracle Text+Hint smoke 后，再实现和训练 IMU adapter。
