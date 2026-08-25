# MotionLab Direct-V4 Contrastive Prescreen

## 目的

Direct-V3 的逐样本 ranking 依赖单个扩散时刻下的 active-joint 重建误差，validation ranking satisfied rate 约为 3.1%。Direct-V4 改用训练期 condition matcher，直接学习“这段 IMU control token 是否属于这段动作”，避免把条件辨识完全寄托在噪声较大的单步去噪误差上。

## 实现

- MotionFlow 与文本编码器保持冻结。
- IMU adapter 和推理接口不变，matcher 只在训练期间使用，不增加采样开销。
- control token 与 normalized HumanML3D motion 分别提取整段均值、标准差和速度 RMS，再投影到 128D embedding。
- 负样本只来自相同 sensor config，使用双向 InfoNCE。
- 额外使用 paired-vs-zero cosine margin，防止 matcher 只学到传感器配置。
- checkpoint 独立保存 matcher；旧 checkpoint 在 contrastive weight 为零时保持兼容。

## 协议

| Setting | Value |
| --- | --- |
| Train / validation | 512 / 128 aligned records |
| Epochs | 2 |
| Batch size | 8 |
| Sensor configs | head, wrists |
| Control space | MotionFlow native 512D hint tokens |
| Learning rate | 5e-5 |
| Flow / trajectory / velocity | 1.0 / 10.0 / 2.0 |
| InfoNCE / zero margin | 0.2 / 0.2 |
| Temperature / margin | 0.07 / 0.1 |
| Seed | 1234 |

最终 checkpoint：

`outputs/motionlab/direct_v4_contrastive_prescreen_clean/adapter.pt`

## 结果

| Epoch | Train flow | Train contrastive | Train top-1 | Val flow | Val contrastive | Val top-1 | Paired > zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.14391 | 1.15602 | 45.8% | 0.13417 | 1.03893 | 56.4% | 98.8% |
| 2 | 0.12718 | 0.77556 | 65.8% | 0.13402 | 0.96497 | 61.9% | 96.9% |

同配置 batch 内随机 top-1 约为 25%。因此 matcher 学到了可泛化的配对信息，但验证 top-1 没有达到预设的 70% 全量训练门槛。按预筛协议停止，不运行 7009 条训练，也不将该 checkpoint 纳入论文主模型竞争。

## 解读

- 相比 V3 的 3.1% ranking satisfied rate，直接条件辨识目标明显更可学习。
- 极高的 paired-vs-zero rate 只说明非零控制容易被识别，不代表模型能稳定区分两个真实 IMU 实例。
- 验证 top-1 从 56.4% 提升至 61.9%，但训练与验证仍有差距，继续增加 epoch 可能主要强化 matcher，而不保证生成动作更服从 IMU。
- 下一步若继续这条线，应先把 matcher 施加到生成动作表征或多个扩散时刻，并验证检索提升是否转化为 paired-vs-shuffled 生成误差改善；当前不值得直接全量扩展。

## 工程验证

- 16-record backward smoke 通过。
- 16-record validation smoke 通过。
- matcher、optimizer 和 history 均进入 checkpoint。
- resume 现在恢复 adapter、matcher 与 AdamW state；早期脚本只恢复网络参数的问题已修正。
