# G0 库存与旧结论影响表（2026-09-24）

读取 HEAD：`663414f1d9bffd6a70189846739bc9d9b1d0d873`，与任务书参考提交相同。
启动时只有未跟踪的 `docs/itm_rework/`，没有覆盖任何旧 checkpoint、缓存、实验输出或其他会话文件。
本批为未提交工作树；公共契约 `itm-rework-v0.1-draft` 尚未与 01 冻结。

机器经沙箱外只读查询有两张 RTX 4090，各 24564 MiB；查询时占用分别为 1902/15 MiB。
未启动 GPU 训练或采样。全部工作为 CPU 审计、三个源片段重合成、已有输出复评和软件测试。

## 可追溯资产

`audit_20260924_v1/assets.jsonl` 保存 627 项重点资产的存在性、字节数、SHA-256。
它是重点库存，不是 17 GB outputs 的全文件内容认证。
全量 manifest 与缓存检查独立保存在 `full_cache_audit.json`；原始目录下的历史路径没有修改。

| 资产 | 核查结果 | 开发角色、复用和限制 |
| --- | --- | --- |
| `outputs/manifests_full/*_standard_imu.jsonl` | train 7009 / val 434 / test 1333，均声明 standard_imu_v1、30 FPS | 8776 个缓存全部存在；形状、有限性、SO(3)、ID/FPS 与覆盖检查通过。不是输出观测闭环认证 |
| 原始 AMASS 与 index.csv | 旧 `/home/a200/0proj/datasets/AMASS` 路径全部失效；当前 `/home/a200/mount/a40/datasets/AMASS` 下 8776 条均可解析；index 已保存 hash | 生成只改 source_path 的新 manifest sidecar，不改缓存值。挂载初查短暂不可读，后续已解决 |
| Stage-2b | hash `5d0800e61994c79f862378871e2c149fde052c7419553939a06a6a6632788a98`；weights_only CPU 加载成功，epoch 8、训练 seed 1234 | 训练参数及 normalization 已提取。不是修复后的 B0，也不是最终候选 |
| Stage-1 / Stage-4 / MDM backbone | 重点库存保存 hash；backbone `ce43c1d4232fc2b897999d7996c2d2b889791d99585d87e1289af773ab56ab93` | 保存旧诊断，不能拼接多 checkpoint 的联合能力 |
| MDM 100-motion | 40 个 NPZ / 800 个输出，身份、哈希、长度、有限值通过；原指标 CPU 重算 | 历史 test 已参与开发；可离线复评，不能作为首次封存测试 |
| MDM 30-motion × 5 seed | 90 个 NPZ / 900 个输出；原指标 CPU 重算 | 30 个来源组；先平均 seed，后组 bootstrap。不包含训练随机性 |
| 文本属性 | 6 个 NPZ / 180 个输出；属性方向及幅度重算 | 未确认相容，属于编辑诊断，不能计入正式 C 成功率 |
| 22-joint→SMPL round-trip | 两份 5-case 结果文件及数值存在 | 本轮只从旧逐样本 residual 重算分布，没有重新运行 fitting；严格 IMU 认证仍失败 |
| Nymeria 3-clip | `outputs/nymeria_pilot/stage2b_zero_shot/` 资产 hash 已登记 | 仅 pilot；本轮未重新采样、未重新独立核验硬件标定/时钟转换 |
| HY-Motion A40 新结果 | 本机缺少 prescreen128 与 fresh64 的指定 summary | 当前标记 reported_only；报告文字不升级为本机可复核的结果 |

旧资产生成入口见 `build_standard_imu_cache.py`、`sample_mdm_imu_control.py`、
`evaluate_mdm_imu_control_generation.py` 和各原 run 的 JSON/log。未定位精确生成命令的资产不补造命令。
本轮执行命令见 `commands.md`。原生成 commit、完整后续噪声计划、部分 evaluator 权重来源仍缺失。

## 旧表—资产—风险—重跑范围

| 旧表/结论 | 原始资产 | 已确认风险 | 处理与最小重跑范围 |
| --- | --- | --- | --- |
| 7009/434/1333 独立划分 | 三份 manifest + index.csv | 749 个来源序列跨划分；557 对时间区间重叠：train/test 399，train/val 134，val/test 24 | 数量保留，不能说来源独立。新 panel 排除跨划分来源；不能把旧训练动作改名为新测试 |
| 1333→682→672 的 5-rep 生成质量 | `quality_subset.json`、旧 evaluator/log | 651 条被当前长度/文字规则排除，10 条尾批丢弃；本轮没有重算 FID | 逐 ID 原因已保存。当前顺序重建不是历史每次 batch 顺序认证；保留旧同子集探索性比较。改尾批后全表一致复评，并检查检索候选数 |
| 100-motion / 30×5 控制 | 136 个 NPZ 中相应 130 个 | 旧 sampler 的 `_fit_length_*` 在全段归一化时间轴插值；未保存完整噪声流 | 旧数值可重算；时序跟随主张降级。B0 修复 sampler 后受影响条件重新生成，使用新模型/run ID |
| wrists 74/100 “超过总体 70%” | control_test100 | 旧 CI 65%–82% 包含 70% | 不支持总体超过 70%。这是统计解释变化，不需重生成才能纠正 |
| 同 head 文本编辑 | text_attribute_formal | 未见 witness；转向可改变 head 姿态；原 head 差值 CI 把 150 个编辑当独立样本 | 保留连续属性变化诊断。先按 motion 平均五种编辑，再按来源组 bootstrap；相容成功率须重新构造 manifest |
| joint acceleration → 真实 IMU 一致 | round-trip 文件 | head/wrists 加速度及姿态门槛均失败 | 只能称 joint-proxy；禁止将普通低误差骨架拟合升级为物理闭环 |
| Stage-1 文本能力 + Stage-2b 控制能力 | 不同 checkpoint 输出 | 不能证明同模型联合能力 | G4 使用同一冻结 checkpoint 的全套结果 |

`development_exposure.jsonl` 是保守暴露账本：所有 1333 个映射 test ID 都至少出现在历史评价记录中。
“出现过”不等于逐条证实用于选择；每条具体决策用途仍标待审核。没有把未在扫描中出现的动作宣称为独立封存集。

历史先验不会因本次重新计算而消失。来源重叠、时间拉伸、未确认相容也不会因数值重现而消失。
