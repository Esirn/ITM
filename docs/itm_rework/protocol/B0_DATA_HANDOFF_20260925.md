# 02 → 01：首批合成 B0 数据与执行协议

读取提交 `1581ba6`。本交付属于 02 数据与协议范围；没有改方法实现，也没有训练或扩散采样。公共契约提升为 **itm-rework-v0.3-draft**，v0.2 原文存于 `contract_history/itm-rework-v0.2-draft.yaml`。01 的 `b4f70fc` 已验收事项无需重复确认。

## 可直接接入的交付

目录：`docs/itm_rework/protocol/canonical_b0_20260925_v3/`。

- `package.json`：交付入口及身份摘要。
- `admission.jsonl`：全部 12 候选的路径、变换、证据、准入/排除原因；不删除负例。
- `train_motion.jsonl`、`train_imu.jsonl`、`train_caption_plans.jsonl`：**3 个训练源组**。
- `val_motion.jsonl`、`val_imu.jsonl`、`val_caption_plans.jsonl`：**3 个验证源组**。
- `plan.json`、`built_pairs.jsonl`、`verification.json`、`geometry.json`：固定选择、源文件 hash、显式映射与核验。
- `semantic_review.json`、`review/`：原 caption 的逐条 GT 审核、16 帧分帧图和全帧运动学曲线。审核者为 02 Codex，**不是人工标注组或多评审一致性实验**。
- `b0_budget.json`、`sampling_cases.json`、`metrics.json`：拟执行预算、36 个固定采样槽位和评价定义。

实际缓存/新目标在 `outputs/itm_rework/canonical_b0_20260925_v3/{train,val}/{ID}/`；未覆盖原数据，输出目录按仓库约定不入 git，但逐文件 hash 已写入交付清单。每条保存 `source_20hz.npz`、`source_30hz.npz`、`imu.npz`、`motion.npy`、`joints.npy`、含末端速度参考帧的 joints 和原 caption 文件。

| 角色 | 准入 ID | 保留但不准入 |
| --- | --- | --- |
| train | `000506`、`004728`、`000027` | `000010` 轻拍手细节不确定；`000123` 电话/说话不确定；`000015` 罐子/准备打斗不确定 |
| val | `000433`、`000382`、`000139` | `000133` caption 的“停止”未确认；`000510` 左右方向冲突；`000134` 物体语义不确定 |

`000027` 保留预先选定的完整 caption **index 1**。没有为通过审核改选 caption。`000510` 右腕在保持阶段相对身体左轴的投影为负（min/median/max 约 −0.438/−0.346/−0.337 m），与文字的左侧描述不符，保留为负例。`000139` 行走时根高上升约 0.64 m，标记非平地支撑，不能用平地足滑/穿地假设评它。

6 个准入源组在原 train/val/test 划分中各自独占，训练和验证不共享源组。它们仍是开发用小面板；MDM 预训练暴露未排除，没有恢复成独立研究测试集。

## 映射与准入范围

新 schema 是 **`canonical_smpl_paired_v1`**，映射为 **`canonical-smpl-pair-v1`**，不能改标签伪装成 `standard_imu_v1`，也不能推广到旧诊断缓存。

1. 仅接收本面板精确 60/120 Hz 源时钟。按已查明的 notebook 前缀补偿源索引，使用同一源姿态、平移、shape 和 male SMPL 模型分别构建 20 Hz 目标、30 Hz IMU。每条记录请求/实际源区间；若文件缺少名义末尾的高频尾帧，只有全部所需 20 Hz 点（包括最后速度参考帧）真实存在才读取可用尾部，无补点或拉伸。该规则独立于旧标准 builder 的越界拒绝。
2. **重新生成配对目标**，不再把 male SMPL IMU 硬对齐到旧 HumanML 的 gender/shape/统一骨长结果。使用 SMPL 前 22 个关节及其原身体比例，缩放为 1，无 `uniform_skeleton` retarget。SMPL 模型身份与每条 shape 已绑定。
3. 在离线合成阶段，计算一次常量正旋转 Q 和平移原点 o：第一帧 pelvis XZ 为零，整个 20 Hz 关节片段最低 Y 为零，解剖左右髋+肩定义的平滑朝向首值转到 +Z。点为 `Q(p−o)`，加速度为 `Qa`，姿态为 `QR`。保存 Q/o，不在推理时从目标 GT 重新求解。
4. 加速度仍是世界系运动学线加速度、m/s²、不加重力、30 Hz 二阶差分后 5 帧平滑，沿用旧边界约定；在源系计算后左乘常量 Q。安装顶点、身体段、槽位不变。30→20 Hz 用线性加速度和旋转 SLERP，不全段拉伸。
5. HumanML263 字段布局保留，编码使用显式无扰动 yaw/骨方向计算。6D 旋转是 HumanML chain-root IK 表示，不是 SMPL 传感器姿态。第一根旋转为解码器 identity；完整片段从零开始，保留逐帧根平移。历史外部四元数 helper 的固定扰动未沿用。

这是**预先统一坐标的离线合成 B0 基准**。`world_alignment_certified=true` 仅在该 scope 下有效；不是任意朝向真实 head/wrists 的部署标定。此范围是本轮构建时已明确说明的工作假设。没有认证从生成 HumanML 输出恢复安装顶点加速度/段姿态；评价仍用明确命名的 joint proxy。

## 已执行验证

- 12/12 数值验证通过；20/30 Hz FK 在公共 10 Hz 时间点逐值一致（最大差 **0**）。
- 新目标 HumanML RIC 独立解码对保存的目标坐标最大差 **4.7684e−7 m**，低于构建前规定的 1e−5 m；未做 GT yaw/平移拟合。
- 补充 6D 骨方向重建最大分量差约 **5.42e−6**，这是连续核验结果，不是事后设置的新语义准入阈值。
- 时间、caption 区间、root、旋转及新编码相关 **26 项测试通过**。12 条原 caption 均完整且有时间支持；语义审核另排除 6 条。
- 构建成功批次 24 次 CPU FK，2 线程、每次不超过 300 帧。另保留前两次中止：源末尾越界（0 FK）与外部四元数扰动（2 FK）。没有 GPU 工作。

## 01 需要处理的具体接口

这次需要接入新数据，不需要再为旧已验收实现增加泛化防护。

1. 为 `canonical_smpl_paired_v1` 增加显式 opt-in 读取，核对嵌入 schema/mapping、交付清单和 hash；不能将新 NPZ 重标为 standard，也不能放行旧 prefix diagnostic schema。
2. 使用交付的 motion/IMU/caption 三份清单。保留完整片段、明确 caption index、20 Hz 时间轴以及仅限合成的 scope；训练只允许 train 三条，val 不进入训练或统计。采样 GT 直接读新 `joints.npy`，不能从 30 Hz 关节再次插值作为监督真值。
3. 从实际送入模型的三个训练片段 **20 Hz acceleration** 重新计算 IMU 均值/方差，每条源片段只计一次，绑定新 checkpoint provenance。MDM motion Mean/Std 是预训练坐标，保留原值并绑定 hash；不在这三条上另拟合。初始化新控制器，不恢复旧诊断 checkpoint。
4. 采样显式支持 reviewed caption 加 `text_present=false` 的真 IMU-only、原 backbone bypass，以及数值零/物理静止条件。静止由首个已观测传感器方向固定加零原始加速度构成。各条件保存完整随机计划和输出。新 scope 的 metadata 不能继续硬编码“旧 HumanML 对齐未认证”，也不能写“真实设备已认证”。

接口完成后 02 只重放这 6 条的输入和新归一化来源，再在授权预算内推进 B0。**相容编辑 witness 是后续 B 分支依赖，不再阻塞本次 B0。**

## 固定 B0 协议与预算提案

一次训练、同一 checkpoint 同时覆盖 head/wrists：batch 3、100 epochs、每 epoch 6 个配置样本，共 **200 updates / 600 examples**，seed 20260925、lr 5e−5、frozen backbone、ordered-concat/additive/strict-null，只有 diffusion loss。仅使用最终 update 200 的模型，无验证择优或超参搜索。

验证为 3 源组 × 2 配置 × 6 条件 × K=1 = **36 条完整动作**：paired、text-only、真正 IMU-only、原 backbone bypass、数值零、物理静止。DDPM 1000 步，seed 20260926；同 case/config 的条件共享寻址噪声、有效长度和权重（text 2.5、IMU 1、joint 1）。最多 36,000 次去噪步调用、144,000 个分支样本前向。

该小面板没有同类近长度的独立 donor，**shuffled 实例增益不可估计**，不以跨动作 donor 代替。K=1 无法估计条件内多样性。本轮只建立 B0 可运行性与小面板反馈，不能支持完整 P/C 主张。

评价先保存全部 36 计划槽位及失败：canonical global/root-relative MPJPE、根轨迹误差、head/wrists joint proxy、速度误差、绝对 jerk 和动作幅度、适用片段的足部速度诊断、strict-null 对 bypass 数值差。语义逐条盲化条件审核。禁止拟合输出朝向/时间；不计算严格 JSR_IMU、相容编辑成功率或小样本 FID 总体结论。详细公式、失败分母和不适用项见 `metrics.json`。

**待授权资源上限：**现有单 GPU，显存最多 24 GiB，训练最多 1 小时、采样最多 3 小时、合计最多 4 GPU 小时；首个上限/NaN/OOM/身份不符即停，保留失败与部分产物，不自动重试或转 CPU。本会话 `nvidia-smi` 无法访问驱动，01 执行前需确认具体空闲设备。预算请求来自用户本轮明确要求“预算须获得相应授权”；当前尚未执行训练或采样。

## 复现命令

Python 使用 `/home/a200/miniconda3/envs/itm/bin/python`。构建目录禁止覆盖；重跑需新目录、新计划并保留身份差异。

```bash
python scripts/build_canonical_b0_panel.py plan --output <new-plan.json>
python scripts/build_canonical_b0_panel.py build --plan <new-plan.json> --output <new-output-dir>
python scripts/check_canonical_b0_panel.py --pairs <new-output-dir>/built_pairs.jsonl --output <new-check.json>
python scripts/audit_canonical_b0_geometry.py --pairs <new-output-dir>/built_pairs.jsonl --output <new-geometry.json>
python scripts/render_canonical_b0_review.py --pairs <new-output-dir>/built_pairs.jsonl --output <new-review-dir>
python -m pytest -q tests/test_canonical_pair.py tests/test_physical_time.py tests/test_caption_time.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py
```

审核不是脚本自动推断结果。最终准入包装命令和显式填写的审核记录由 `finalize_canonical_b0_panel.py` / `semantic_review.json` 保存，不能将未审核的重建目录直接批准为训练数据。
