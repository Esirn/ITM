# HumanML 历史时钟诊断：SSM 整数步长与数据集前缀裁剪

读取提交 `e14431a`；公共契约保持 `itm-rework-v0.2-draft`。
本批新增 02 的诊断代码与证据，没有改变运行时源映射、缓存、模型、split 或阈值。
改动提交可用 `git log -1 -- src/itm/data/humanml_clock.py` 定位。

## 结论及证据强度

本地 `raw_pose_processing.ipynb` 的 `down_sample = int(fps / ex_fps)`（`ex_fps=20`）可以解释上一批 13 条 SSM 来源的时钟差异：
当前源 FPS 为约 59.9999886，整数步长变成 2，抽取后真实采样率约 30 Hz，却用名义 20 Hz 解释帧索引和动作时长。
因此名义时长与物理时长之比约为 1.5。

三个层面的证据分别是：

1. **代码事实：**本地 notebook 确实使用上述规则，`src/itm/data/humanml_clock.py` 只复演该时钟公式，不执行 notebook 或人体模型。
2. **产物结构吻合：**13/13 源文件按该整数步长、index.csv 区间和特征末帧丢弃规则计算出的特征帧数，都与保存的 HumanML 特征帧数一致。
3. **几何诊断支持：**13/13 的关节残差在该来源时间轴上下降；逐片段等权平均 oracle yaw 对齐误差由 **0.10300367 m 降至 0.03377891 m**。

这比仅凭帧数推测更有证据支持，但**仍未认证历史生成链或独立校准源 FPS**。原 `pose_data/...npy` 与裁片前的 `joints/000162.npy` 在该 notebook 根目录不可用，本批没有逐关节重建历史 SMPL+H 产物。
不把“解释吻合”升级为所有 HumanML 数据时钟均已修复。

## 固定 13 条几何比较

仍使用上一批已锁定的 13 条 train 样本，没有根据结果选样本、搜索 lag 或拟合时间缩放。
比较以下两个预先定义的目标时间轴：

- 当前前缀名义时间 `i/20`。
- notebook 对应源帧除以实际 FPS，再扣除现有 cache 的真实源裁片起点。

只在两个时间轴都落于同一 cache 支持内的**相同 HumanML 帧**上比较；每条使用一个固定 yaw 拟合，减去各帧 root 后比较 22 joints。
yaw 使用完整目标关节真值，仅限诊断，禁止作为 head/wrists 部署预处理；骨架/身体重定向误差仍存在。

| motion ID | 共同比较帧 / 特征帧 | 名义时间误差 m | notebook 来源时间误差 m |
| --- | ---: | ---: | ---: |
| 000162 | 105 / 157 | 0.064881 | 0.033625 |
| 002511 | 155 / 199 | 0.143502 | 0.032664 |
| 002941 | 139 / 199 | 0.167663 | 0.033630 |
| 004028 | 95 / 141 | 0.133472 | 0.032581 |
| 006515 | 142 / 199 | 0.081529 | 0.034310 |
| 006749 | 107 / 159 | 0.071448 | 0.036879 |
| 007129 | 109 / 163 | 0.088736 | 0.032590 |
| 007410 | 78 / 116 | 0.121830 | 0.033644 |
| 008802 | 121 / 180 | 0.128682 | 0.033216 |
| 008894 | 80 / 119 | 0.049754 | 0.031296 |
| 009466 | 94 / 140 | 0.082201 | 0.033977 |
| 011120 | 101 / 151 | 0.129304 | 0.038748 |
| 011475 | 103 / 153 | 0.076045 | 0.031965 |

上述不能称全片段同步、生成质量或真实 IMU 闭环通过。缺失帧仍被明确记入支持范围；没有自动填充或拉伸。

## 另外发现的前缀裁剪差异

同一 notebook 在应用 index.csv 前，对五类数据集先裁掉固定的**名义 20 Hz** 前缀：

| 数据集 | 名义前缀秒数 | train 记录 | val 记录 |
| --- | ---: | ---: | ---: |
| Eyes_Japan_Dataset | 3 | 1186 | 68 |
| MPI_HDM05 | 3 | 616 | 43 |
| TotalCapture | 1 | 53 | 4 |
| MPI_Limits | 1 | 108 | 4 |
| Transitions_mocap | 0.5 | 93 | 2 |

共 **2177 条 train/val**（2056 train、121 val）命中这些数据集名。
当前 `source_crop_at_fps` 没有前缀补偿。这是代码与来源元数据推导的待核查范围，**不是 2177 条已逐条确认错位的缓存**；其名义秒数也不自动等于已校准物理秒数。
完整清单 `notebook_clock_20260925_v1/prefix_candidates.jsonl` 均标为未认证、不可训练；没有改现有 split 或隔离名单。
此前来源区间的 20 Hz index 应继续视为名义索引，不能无条件解释成原始 AMASS 物理秒。

## 对后续工作的具体约束

- 不把约 60 Hz 强行四舍五入后继续声称与既有 HumanML 特征时序相同。
- 不仅修改 cache FPS 或缩放输入时间轴来追认配对：动作速度与加速度物理量也会改变，必须单独验证。
- SSM 的 13 条继续按现有秒轴拒绝，保留旧 cache 与诊断；上一批新建缓存的源越界 guard 保持有效。
- 01 可继续使用 caption 区间接口，但它不修复这些时钟差异；方法不得把本诊断映射自动当作训练/部署许可。
- 后续 02 应对五类前缀样本做预先固定的轨迹对应核查，再决定可信子集或显式迁移协议。C 的输出 IMU 闭环、身体重定向和相容见证仍未通过。

## 复现与预算

```bash
timeout 60 /home/a200/miniconda3/envs/itm/bin/python scripts/audit_humanml_notebook_clock.py --output-dir docs/itm_rework/protocol/notebook_clock_20260925_v1
timeout 60 /home/a200/miniconda3/envs/itm/bin/python scripts/audit_notebook_clock_geometry.py --clock-report docs/itm_rework/protocol/notebook_clock_20260925_v1/report.json --output docs/itm_rework/protocol/notebook_clock_20260925_v1/geometry.json
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_humanml_clock.py tests/test_source_crop_support.py tests/test_caption_time.py > docs/itm_rework/protocol/logs/notebook_clock_regression_20260925.txt 2>&1
```

**17 passed**，其中 4 个为新增时钟测试，覆盖近 60 / 精确 60 Hz、前缀、名义/物理双时间轴和诊断性越界标记。
最多 13 个既定源 hash/几何案例、7443 条现有 train/val 元数据、两个固定时钟假设；无新 body-model forward、训练、生成或 GPU 作业。
notebook/hash、所用源码、完整逐片段结果和公共支持帧保存在 `notebook_clock_20260925_v1/`，原 notebook 未修改。
