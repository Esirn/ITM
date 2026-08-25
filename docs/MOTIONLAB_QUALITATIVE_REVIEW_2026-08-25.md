# MotionLab Direct-V2 Qualitative Review

## Protocol

检查目录：

```text
outputs/motionlab/direct_v2_full/test20_factorized/manual_review/
```

对8个best/worst GIF各均匀抽取6帧，比较Ground truth、MotionLab Text-only和
Text+IMU。`best/worst`只由active-joint trajectory proxy排序，不代表文本语义质量。

## Findings

| Config | ID | Metric label | Visual finding |
| --- | --- | --- | --- |
| head | 014310 | best | GT包含跌倒与起身；Text+IMU主要保持直立/弯曲，未可靠完成完整语义。轨迹proxy改善不能视为语义成功。 |
| head | 014087 | best | 文本要求边走边不自然地抬手；Text-only有明显抬臂，Text+IMU多数帧手臂更低，文本细节被削弱。 |
| head | 014025 | worst | 前后走并低头的粗粒度动作存在，但Text+IMU未显示比Text-only更清楚的方向/低头控制。 |
| head | 014603 | worst | jumping-jack式动作可辨认，Text-only与Text+IMU均只部分匹配，控制结果没有稳定优势。 |
| wrists | 004117 | best | 跳跃开合的宽站姿可见，但手臂V形和回收不稳定；指标改善未转化为完整动作语义。 |
| wrists | 014087 | best | 与head版本相同，Text+IMU削弱抬手语义，不能作为成功主图。 |
| wrists | 011573 | worst | 文本描述肩高屈臂/拍动；Text+IMU对上肢时序补全较弱，失败与proxy排序一致。 |
| wrists | 014603 | worst | 可辨认基础开合动作，但Text+IMU没有比Text-only更稳定地匹配手臂幅度。 |

总体上未观察到持续骨长变化或早期回归模型式高频果冻抖动，且定量jerk低于Text-only；
但这8个候选中没有同时满足“文本语义清楚、IMU影响可解释、视觉自然”的强样例。

## Paper Use

- 不把上述metric-best样例包装成视觉成功案例。
- 正文可用一组paired/shuffled条件交换说明IMU确实影响输出，但必须同时展示失败例。
- MotionLab结果更适合作为backbone迁移和trade-off实验，而不是替换当前MDM Stage-2b主图。
- 最终人工结论仍需在浏览器/GIF连续播放下由作者复核；本记录基于均匀抽帧，不能完整判断速度和微小抖动。
