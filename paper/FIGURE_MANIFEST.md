# ITM 论文图清单

本文档固定论文制图所使用的实验来源、优先样例和结论边界。原始产物是 Web demo JSON、GIF 和 NPZ；论文最终版本应从同一结果重新导出静态关键帧和轨迹曲线，不使用 GIF 截图作为最终图片。

## Figure 1: Same Text, Different IMU

**目的**：固定模糊文本、长度、seed 和扩散噪声，只改变 IMU，展示具体运动细节发生变化。

**模型**：MDM Stage-2b。

**来源目录**：

```text
outputs/mdm_control/experiments_stage2b_balanced_b/
  same_text_different_imu/test_walk_head_wrists/
```

**固定文本**：`a person walks`。

**优先 motion ID**：

1. `004817`：低 jerk，head/wrists root travel 有差异。
2. `011608`：head/wrists root travel 对比更强，作为备选。
3. `009477`：数值对比最大，但只有人工复核姿态自然后才使用。

**面板**：GT IMU source、MDM Text-only、ITM+head、ITM+wrists、同步 IMU 曲线、root trajectory。

**允许主张**：同一句文本下，更换 IMU 能改变轨迹、位移、节奏或身体摆动。

**禁止主张**：每条 IMU 都被精确重建；joint proxy 等价于真实 orientation consistency。

## Figure 2: Same Head IMU, Different Text

**目的**：固定 head IMU 与生成噪声，观察文本是否补全未观测身体语义。

**模型**：Stage-1 用于最清楚的文本补全诊断；Stage-2b 可作为平滑性对照。

**来源目录**：

```text
outputs/mdm_control/experiments/
  same_head_imu_different_text/test_prompts/
outputs/mdm_control/experiments_stage2b_balanced_b/
  same_head_imu_different_text/test_prompts/
```

**正文成功候选**：`003005`。人工观察中快慢、大步和转弯相对可见，但摆臂与普通走路接近。

**正文失败候选**：`007322`。快慢尚可，大步和转弯不明显。

**补充材料候选**：

- `004344`：快慢明显，大步不明显。
- `005697`：快慢一般，大步明显。

**面板**：共享 head IMU、`walks`、`walks while swinging arms`、`turns while walking`，以及一个快慢/大步弱响应对照；显示 head trajectory 与 arm-swing proxy。

**允许主张**：摆臂和转向文本在部分样本上改变未观测关节；该现象存在但不稳定。

**禁止主张**：快慢、步幅和摆臂都已被稳定解耦。

## Figure 3: Text/IMU A-B Matrix

**目的**：证明任务不是普通重建，并展示 Text A + IMU B 的条件交换。

**模型**：MDM Stage-2b。

**正文主候选**：

```text
outputs/mdm_control/experiments_stage2b_balanced_b/
  matrix_test_wrists/pair_001_004488_004222/
```

**备选**：

```text
outputs/mdm_control/experiments_stage2b_balanced_b/
  matrix_test_head/pair_013_007089_006756/
```

**布局**：GT A/B 单独位于首行；下方使用 Text None/A/B 与 IMU None/A/B 的 3x3 网格，None-None 留空并显示条件、seed 和传感器配置。

**允许主张**：文本和 IMU 均会影响输出，交叉条件可以产生不同动作。

**禁止主张**：Text A + IMU B 总能被视觉上严格拆解为 A 的全部语义和 B 的全部运动细节。

## Figure 4: Nymeria Preliminary Study

**定位**：预备实验或补充材料，不进入主结果图。

**来源**：

```text
outputs/nymeria_pilot/stage2b_zero_shot/per_clip/
```

包含 3 条有效片段及 `index.json`。面板展示 GT、MDM Text-only、ITM+head、ITM+wrists和真实 IMU 曲线。

**允许主张**：三个零样本片段中 head 输入显示初步跨域控制信号。

**禁止主张**：已经完成 Nymeria 泛化验证，或已经与 Ego4o 公平比较。

## MotionLab Appendix Figure

MotionLab 不进入正文核心定性图。其 metric-best 样例没有同时满足语义清楚、IMU 影响可解释和视觉自然。成功与失败各选 4 条放入附录，具体 ID 和观察记录见：

```text
docs/MOTIONLAB_QUALITATIVE_REVIEW_2026-08-25.md
```

该图只支持“方法可以迁移到第二种 Text-to-Motion backbone，但实例级控制仍不稳定”。

## 最终导出要求

- 所有动作使用相同视角、骨架样式、帧采样位置和坐标范围。
- 同组动作共享时间轴，至少取起始、25%、50%、75%和结束前五个关键帧。
- IMU 曲线明确注明部位、acceleration axis、归一化方式和时间单位。
- 图注写明模型 checkpoint、sensor config、text/IMU guidance、seed 和 motion ID。
- 正文至少包含一个成功样例和一个失败样例，避免仅按 proxy 指标挑图。
