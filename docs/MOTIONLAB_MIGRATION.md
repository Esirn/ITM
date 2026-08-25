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

### 固定协议复评

五个 epoch checkpoint 使用相同 validation 样本、timestep、noise 和 seed 复评，并增加 zero control 与 batch-shuffled control：

- epoch 5 head：paired `0.13882`，zero `0.15240`，shuffled `0.14266`。
- epoch 5 wrists：paired `0.13942`，zero `0.15240`，shuffled `0.14239`。
- paired 在两种配置上都优于 zero/shuffled，说明 control token 包含实例级 IMU 信息，而不只是固定条件偏置。
- 综合 paired loss 选择 epoch 5，固定为 `outputs/motionlab/direct_imu_control/adapter_selected.pt`。

完整结果：`outputs/motionlab/direct_imu_control/checkpoint_selection.json`。

### Test-20 固定种子诊断

20 条随机选择的 aligned test motion 使用相同 text、length、seed 和初始 diffusion noise：

| Config | Text-only active error | Text+IMU active error | Change | Improved | Motion difference | Jerk Text | Jerk Text+IMU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 0.1158 m | 0.1044 m | -0.0114 m | 11/20 | 0.0708 m | 2.333 | 1.858 |
| wrists | 0.3954 m | 0.4088 m | +0.0135 m | 11/20 | 0.0727 m | 2.333 | 1.881 |

误差采用逐帧 root-relative HumanML joints，只是控制 proxy，不是真实 generated-IMU orientation error。head 显示小幅平均收益；wrists 的中位数略有改善，但少数严重失败导致均值恶化。当前多传感器在时序编码前做 mask average，可能丢失左右腕独立结构，是下一版 sensor-token adapter 的直接动机。

- 汇总：`outputs/motionlab/direct_imu_control/test20/summary.md/json`
- 人工筛选：`outputs/motionlab/direct_imu_control/test20/manual_review/`

### Sensor fusion 与 consistency follow-up

固定槽位 concat fusion 保留左右腕独立特征，但在同一 test-20 上没有改善：head active error change 为 `-0.0010 m`，wrists 为 `+0.0135 m`。因此 concat 是负消融，主线保留 mean pooling。

Direct-2 在 flow-matching loss 外，从训练时 velocity prediction 恢复 predicted x0，经可微 HumanML `feats2joints` 加入 active-joint root-relative position/velocity loss。MotionFlow 和 CLIP 仍完全冻结。使用 position weight `10`、velocity weight `2` 训练 5 epochs 后：

| Model | Head change | Head improved | Wrists change | Wrists improved | Head jerk | Wrists jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct-1 mean | -0.0114 m | 11/20 | +0.0135 m | 11/20 | 1.858 | 1.881 |
| Direct-1 concat | -0.0010 m | 10/20 | +0.0135 m | 10/20 | 2.085 | 1.952 |
| Direct-2 consistency | -0.0078 m | 11/20 | -0.0039 m | 13/20 | 1.920 | 1.834 |

Direct-2 牺牲了一部分 head 平均收益，但修正了 wrists 平均退化并提高 wrists 改善率，是当前更平衡的 MotionLab 候选。收益仍然较小且存在明显失败样例，不能据此声称已解决 IMU control。候选 checkpoint 为 `outputs/motionlab/direct_imu_consistency/adapter_selected.pt`，test-20 汇总和人工样例位于同目录下的 `test20/`。

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

### Direct-V2：预嵌入控制与可分解 guidance

Direct-1/2 都让 IMU encoder 输出 66D token，再经过 MotionFlow 为几何轨迹提示训练的
`hint_embed1`。这条接口能产生控制信号，但 test-20 上平均收益较小，paired control 与
shuffled control 的间隔也不稳定。Direct-V2 保留外部 MotionLab 仓库只读，并作以下兼容扩展：

- IMU encoder 直接输出当前官方 checkpoint 的 512D `token_dim`，绕过 66D 几何提示投影；
- 512D token 继续走 MotionFlow 原生 hint stream，在全部 6 个联合 Transformer block 中与文本和动作交互；
- 输出投影可零初始化，使训练起点不改变冻结 MotionFlow；
- 增加 paired-vs-shuffled active-joint ranking loss，显式要求配对 IMU 优于错误 IMU；
- 推理可选四分支 `uncond/text/imu/text+imu`，独立调节 text、IMU 和交互项。

四分支公式为：

```text
v = f00
  + text_scale  * (f10 - f00)
  + imu_scale   * (f01 - f00)
  + joint_scale * (f11 - f10 - f01 + f00)
```

16 条数据的 512D smoke 训练和 1 条样本的两步 factorized sampling 已通过，产物位于
`outputs/motionlab/direct_v2_smoke/`。128 条零初始化预筛中，epoch 2 的 paired flow loss
在 head/wrists 上均优于 zero 和 shuffled，因此启动了正式训练。

正式 Direct-V2 使用 7009 条 train、434 条 val、head/wrists、batch 8、5 epochs、学习率
`5e-5` 和 GPU 1。checkpoint 位于 `outputs/motionlab/direct_v2_full/`，固定 434 条 validation
复评选择 epoch 5：

| Config | Paired | Zero | Shuffled | Gain vs zero | Gain vs shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| head | 0.13605 | 0.14603 | 0.14146 | 0.00997 | 0.00541 |
| wrists | 0.13650 | 0.14603 | 0.14039 | 0.00952 | 0.00389 |

同一 test-20 上的生成结果如下。误差是 root-relative active-joint proxy，不是真实 IMU error：

| Guidance | Config | Error change | Improved | Text jerk | Text+IMU jerk |
| --- | --- | ---: | ---: | ---: | ---: |
| legacy | head | -0.0073 m | 13/20 | 2.333 | 2.166 |
| legacy | wrists | -0.0140 m | 11/20 | 2.333 | 2.072 |
| factorized | head | -0.0090 m | 12/20 | 2.333 | 2.110 |
| factorized | wrists | -0.0147 m | 11/20 | 2.333 | 2.071 |

factorized guidance 在平均误差上略优于 legacy，且未增加 jerk，但改善率仍只有 55%-60%。
这表明 512D 预嵌入控制和 ranking loss 提高了 paired/shuffled 可辨识性，并改善了 wrists
平均控制误差，但实例级控制仍不稳定。完整 summary 位于
`outputs/motionlab/direct_v2_full/test20_legacy/` 和 `test20_factorized/`；最佳/最差 GIF 位于
`test20_factorized/manual_review/`。一次误启动的两个并发训练曾混写同一目录，已终止并隔离为
`outputs/motionlab/direct_v2_mixed_invalid/`，该目录不得用于任何实验结论。

### Direct-V3 预筛（停止全量训练）

Direct-V3尝试在每帧先对保留sensor embedding的槽位执行sensor attention，并将原先batch
标量ranking改为同配置负样本的逐样本paired/shuffled/zero ranking。三个512-train、
128-val、2-epoch受控变体为：A仅sensor attention，B仅per-sample ranking，C两者同时使用。

固定validation复评选择C；其head paired/zero/shuffled分别为
`0.11822/0.12401/0.12071`，wrists为`0.11895/0.12401/0.12037`。但是训练中的
margin-satisfied rate只有约`3.1%`。更关键的是，C在test-20 factorized采样中仅有head
`11/20`、wrists `8/20`改善；head平均误差下降`0.0061 m`，wrists反而增加`0.0056 m`。
该结果未达到预设的双配置70%改善门槛，因此不启动7009条全量V3训练。V3作为负消融说明：
保留传感器身份和简单单步ranking仍不足以保证实例级生成控制。

### Direct-V4 对比条件辨识预筛（停止全量训练）

Direct-V4不再通过单个扩散时刻的重建误差间接排序，而增加仅训练期使用的condition
matcher。matcher从512D control token和normalized 263D motion的时序统计中提取128D
embedding，并在相同sensor config内执行双向InfoNCE，同时加入paired-vs-zero cosine
margin。该模块不参与采样，因此不增加推理成本。

在512条train、128条val、2 epochs的干净预筛中，validation retrieval top-1从
`56.4%`升至`61.9%`，明显高于同配置batch内约`25%`的随机水平；paired-beats-zero为
`96.9%`。这说明直接条件辨识比V3 ranking更可学习，但仍未达到预设`70%`门槛。
因此不启动7009条全量训练。完整协议和结果见
`docs/MOTIONLAB_V4_CONTRASTIVE_PRESCREEN_2026-08-25.md`。

### 竞争主模型评价

新增backbone-neutral的批量生成与官方HumanML evaluator桥。MotionLab Text-only在相同
672条IMU-mapped subset、5次重复上的结果为：Matching Score `2.7223 +/- 0.0230`、
R@1/2/3 `0.5292/0.7438/0.8321`、FID `0.2683 +/- 0.0217`、Diversity
`9.8051 +/- 0.5640`。这些结果优于当前MDM subset baseline，说明MotionLab有资格作为
竞争主模型。加入V2控制后的5-rep结果为：

| Model | Matching | R@1 | R@2 | R@3 | FID | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MotionLab Text-only | 2.7223 | 0.5292 | 0.7438 | 0.8321 | 0.2683 | 9.8051 |
| MotionLab V2 head | 2.8010 | 0.5170 | 0.7280 | 0.8193 | 0.3173 | 10.4175 |
| MotionLab V2 wrists | 2.8025 | 0.5196 | 0.7185 | 0.8205 | 0.3879 | 10.1779 |

Head的Matching、R@3和FID相对退化约`2.9%/1.5%/18.3%`，通过预设生成质量门槛；
wrists的FID增幅约`44.6%`，未通过30%门槛。

固定100条test control suite进一步比较Text-only、paired、zero和shuffled control：

| Config | Text error | Paired error | Zero error | Shuffled error | Paired<Text | Paired<Zero | Paired<Shuffled | Text jerk | Paired jerk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head | 0.1733 | 0.1548 | 0.1534 | 0.1618 | 60/100 | 42/100 | 63/100 | 2.319 | 2.005 |
| wrists | 0.3924 | 0.3615 | 0.3679 | 0.3774 | 63/100 | 54/100 | 67/100 | 2.319 | 1.971 |

两种配置都未达到70%实例改善门槛，且head paired平均误差略高于zero。因此MotionLab不替换
MDM Stage-2b作为主模型，而作为跨backbone迁移实验：更强Text-to-Motion先验可显著改善
生成质量，但当前IMU适配器仍不足以稳定地进行实例级控制。

## 后续验收

- 固定提示词和种子生成 Text-only 样例，并确认无 NaN、长度和关节维度正确。
- 与官方 demo 的同条件输出核对，允许因批处理和随机数初始化导致的差异，但模型路径必须一致。
- 接入现有 ITM 可视化，人工比较 MDM 与 MotionLab 的语义匹配、抖动和动作自然度。
- 完成 oracle Text+Hint smoke 后，再实现和训练 IMU adapter。
