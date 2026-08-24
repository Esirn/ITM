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

## 后续验收

- 固定提示词和种子生成 Text-only 样例，并确认无 NaN、长度和关节维度正确。
- 与官方 demo 的同条件输出核对，允许因批处理和随机数初始化导致的差异，但模型路径必须一致。
- 接入现有 ITM 可视化，人工比较 MDM 与 MotionLab 的语义匹配、抖动和动作自然度。
- 完成 oracle Text+Hint smoke 后，再实现和训练 IMU adapter。
