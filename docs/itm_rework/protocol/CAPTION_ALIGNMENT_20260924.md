# Caption 区间审计与显式输入计划

读取提交 `2ee7b2e`。本批只新增 02 的数据接口、审计/计划脚本、测试与证据；公共契约保持 `itm-rework-v0.2-draft`，方法入口未改。
改动提交通过 `git log -1 -- src/itm/data/caption_time.py` 定位。

## 新事实

在既有 standard IMU 对应的 **7,009 train + 434 val** 上检查 caption 文本与 HumanML 特征帧数，共 22,235 条标注。
不读取 test caption；特征只读 NPY 头/shape，不执行模型。
当前方法入口使用第一条非空 caption 和 `[0,min(T,196))` 特征前缀，未按 caption 标签裁片。

| 第一条 caption 与当前入口的关系 | train | val |
| --- | ---: | ---: |
| 标注区间恰好匹配前缀，IMU 支持通过 | 3549 | 200 |
| 整段标注对应动作超过 196 帧，当前前缀会截短 | 3167 | 209 |
| 有时间标签，但标注区间与当前前缀不同 | 187 | 15 |
| 第一条标签/区间无效或缺失 | 96 | 10 |
| 标注区间匹配，但 IMU 时间支持不足 | 10 | 0 |
| 合计 | 7009 | 434 |

“截短”不证明 caption 在前缀上语义错误，只说明完整描述是否仍适用尚未审查，不能自动认为配对正确。
“区间匹配”也不认证文字真实描述动作、世界坐标、骨架或安装传感器同步。
所有标注中 331 条被新严格规则拒绝：280 条超出特征区间/区间错误、50 条非有限或负时间、1 条结束不大于开始。
旧解析器容忍异常值不等于它们有可信的全段含义。

10 条时间支持不足均来自源组前缀 `AMASS/SSM_synced/20160330_03333/`，ID 为
`000162/004028/006749/007129/007410/008802/008894/009466/011120/011475`。
例如 `000162` 特征末时刻 7.8 秒，cache manifest 末时刻仅 5.2333 秒。
进一步独立于 caption 状态检查所有当前前缀，共发现 **13 条 train** 支持不足；另 3 条为 `002511/002941/006515`，在上表归入 caption 截短，故不能把表中 10 条理解为全部时间支持失败。
13 条都属于上述 SSM 录制目录。对其源文件的 pose NPY 头和 FPS、实际 cache 帧数/FPS 做了限额复核：源 FPS 约 59.999989，按索引换算的裁片终点全部超过源文件长度；Python 切片截到文件末端后，重采样预测帧数与现有 cache **13/13 一致**。
例如 `000162` 源仅 315 帧，声明裁片却到 474 帧。这确认当前桥接/源数据之间存在区间越界，但未独立校准源 FPS，也未重建历史 HumanML 流程，不能推断应改 FPS 或拉伸。
细节和源/cache hash 在 `caption_audit_20260924_v1/time_support.json`。
此前全 cache 数组/旋转检查通过，并不表示它们与 HumanML 的时长都一致；这是新增跨表示检查的阻塞。

已为 `source_crop_at_fps(..., source_frame_count=...)` 增加边界检查，并在 `build_standard_imu_cache.py::_process_source` 切片前传入真实源帧数，同时检查 pose/translation 长度一致。
新建缓存遇到越界会在人体模型计算前失败，不能静默生成短 cache。未覆盖既有缓存；构建脚本复用已有文件时也不自动追认其时序正确，应使用本次拒绝清单。

## 可交付给 01 的接口

`src/itm/data/caption_time.py` 提供独立 `caption-interval-v1`：

- `parse_timed_caption` 严格读取四字段，拒绝缺失、NaN/Inf、负值和反向区间，保留 `(0,0)` 为全段标注 sentinel。
- `caption_interval` 使用本地上游 MDM 的 `int(tag * 20)` 半开切片规则，记录量化偏差、帧区间与实际末采样时刻。先检查越界，不使用 Python 切片静默夹紧。
- `plan_caption_crop` 要求整个标注区间不超过 196 帧且有 IMU 时间支持。返回原片段秒轴与重置为零的模型秒轴；**IMU 插值使用原秒轴**。不自动截短 caption、不重做 HumanML canonicalization，不把裁片升级为训练准入或 compatible witness。

上游源码路径/hash 已保存在审计 summary，作为区间约定的代码依据；这里的越界/异常拒绝比旧上游更严格。
此接口尚未替换方法入口，不修改既有 checkpoint 分布。01 应显式接纳后集成并记录新的输入管线版本。
非零起点裁片的特征根位置/朝向、观测帧和 witness 还需联合审查；不能只给重置的 timestamp 就称物理对齐完成。

## 固定案例计划

`caption_plan_20260924_v1/cases.jsonl` 固定 4 条已有 train 诊断和原来的 6 条 val，共 10 条；9 条区间计划通过，1 条阻塞。
同时直接检查这 10 个 cache 实际长度/FPS 与 manifest 相等，并记录 text/feature/cache hash；未生成动作或另存训练数据。

| 案例 | 结果 |
| --- | --- |
| train 000004 | 199 帧全段 caption 超出 196 模型上限，保留阻塞；原 witness 不自动裁剪 |
| train 000006 | 第一条 `[0,5)` 秒 caption 可规划为 100 帧，末采样 4.95 秒；不同于旧 196 帧前缀，尚未执行训练 |
| train 000008 / 000010 | 原整段分别 42 / 69 帧，区间支持通过 |
| 原 6 条 val | 原先选定的 caption 与完整片段区间均支持，不改 ID、文本或长度 |

**val 000133 的固定 caption 是索引 1（第二条非空标注）**；索引 0 标记 `[1,3)` 秒，超过 59 帧特征的半开末端 2.95 秒。
因此加载计划时必须保留 `caption_index`/明确文本，不能重新使用默认索引 0。
既有 panel 选择的是完整片段 caption，本批没有按模型表现筛选或换掉该文本。

`caption_audit_20260924_v1/full_clip_candidates.jsonl` 另列 3,550 train / 184 val 条来源仅属于本 split、完整 caption 支持且无需 196 截短的候选。
这只是后续入口修复的候选 sidecar，**所有 training_admissible=false**；不是新 split、训练许可、封存测试池或已审定相容编辑。
source-exclusive 不代表对 backbone 预训练独立。

## 验证、命令与资源

新 caption 测试 11 项，最初联合公共秒轴/方法输入测试共 21 passed；新增源边界和真实构建入口拒绝测试后，最终相关回归 **31 passed**。没有重跑或改写上轮 83 项的历史日志。
覆盖异常标签、完整片段上限、非零起点的双时间轴、半开末端/量化、短 IMU 拒绝及源文件边界。

```bash
timeout 180 /home/a200/miniconda3/envs/itm/bin/python scripts/audit_caption_intervals.py --output-dir docs/itm_rework/protocol/caption_audit_20260924_v1
timeout 60 /home/a200/miniconda3/envs/itm/bin/python scripts/plan_caption_inputs.py --output-dir docs/itm_rework/protocol/caption_plan_20260924_v1
timeout 60 /home/a200/miniconda3/envs/itm/bin/python scripts/audit_caption_time_support.py --caption-audit docs/itm_rework/protocol/caption_audit_20260924_v1 --output docs/itm_rework/protocol/caption_audit_20260924_v1/time_support.json
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_caption_time.py tests/test_physical_time.py tests/test_mdm_conditions.py > docs/itm_rework/protocol/logs/caption_interval_regression_20260924.txt 2>&1
/home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_caption_time.py tests/test_source_crop_support.py tests/test_physical_time.py tests/test_mdm_conditions.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py > docs/itm_rework/protocol/logs/caption_and_source_regression_20260924.txt 2>&1
```

审计最多 7,500 train/val 条、计划最多 10 条、后续源头核查最多 32 条（实际 13）；均已完成，数据审计模型前向、训练 update、生成、GPU 作业为 0。回归中的小模型 fixture 不计为研究作业。
输出目录已存在时拒绝覆盖；复跑使用新版本目录。完整逐条结果、源清单/hash、预算在对应输出目录。

01 下一步可按明确 caption 索引/区间消费 sidecar，保留源秒轴，拒绝静默夹紧和语义未审查的前缀；实现与回归归 01。
02 下一步需核实 SSM 时钟差异、部署坐标/骨架约定与文本语义，完成独立 witness/阈值；不以区间计划替代这些认证。
03 可引用数据入口风险及审计数量；不能把 9 条计划通过写成完整物理同步、B0 训练或生成成功。
