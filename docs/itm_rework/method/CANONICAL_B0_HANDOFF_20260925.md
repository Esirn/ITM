# 01 → 02：合成 B0 输入接入与新归一化

读取数据提交 `c36ea56`，公共契约 `itm-rework-v0.3-draft`；本批方法提交通过
`git log -1 -- docs/itm_rework/method/CANONICAL_B0_HANDOFF_20260925.md` 定位。
方法侧接入完成，交给 02 重放输入与统计；未训练、未执行真实 MDM 前向或扩散采样、未启动 GPU。

## 接入范围

`--canonical-package <package.json>` 显式启用 `itm-method-canonical-b0-v1`。
`CanonicalB0Package` 校验交付 package → admission/budget → 三份配对 manifest → 每条 artifact 的 hash 链，
并核对 `canonical_smpl_paired_v1` / `canonical-smpl-pair-v1` 的嵌入字段、时间支持和保存的 Q/o。
原标准缓存入口继续拒绝该新 schema；旧 prefix diagnostic 缓存仍未放行。
数据映射/重建代码及公共契约未修改，没有在运行时重新拟合 Q/o、纠正 SSM 时钟或做 retarget。

只允许交付准入列表中的完整片段，沿用 caption index，`000027` 为 index 1。
训练只读三个 train 源组；val manifest 不能传入训练数据适配器。当前 canonical B0 CLI 要求新建控制器，不接受 `--resume`、缩减清单或编辑 witness。
相容编辑 witness 不再阻塞这个 diffusion-only B0 输入路径。原 seconds-v2 / caption-seconds-v3 接口保留。

metadata 明确标注 `admitted_precanonicalized_synthetic_B0_only`，scope 为
`offline_synthetic_only_not_real_device_calibration`。不沿用旧输入的“未认证 HumanML 对齐”标签，
也不声称真实设备标定、生成 IMU 闭环或完整任务能力已认证。

## 归一化与来源

| split | ID | 实际 20 Hz 模型帧数 |
| --- | --- | ---: |
| train | 000027 | 104 |
| train | 000506 | 136 |
| train | 004728 | 158 |
| val | 000433 | 41 |
| val | 000382 | 94 |
| val | 000139 | 163 |

IMU 统计只使用前三条：30 Hz cache 在实际 20 Hz 输入时刻线性插值后的 acceleration，共 **398 帧**，
head/wrists 两配置不重复计数。全六槽逐轴总体均值/方差，variance floor 1e−8，orientation 保持原矩阵不缩放。
另用独立 `numpy.interp` + 总体方差计算核对：均值 float32 精确相等，std 在 rtol=1e−7 / atol=0 下相等。
该 std 容差只用于独立求和顺序复核，没有更改 02 的历史 acceleration allclose 阈值。

实际统计和来源绑定保存在 `canonical_training_input_20260925_v1.json`；provenance SHA256：
`9f39bf7ffd93b2595dfdcea4ee3288761b14a6e75a6585fedb841fd490295ef1`。
包含 package/admission/六份 manifest/预算/验证/审核等交付身份，以及所选训练文件、原始 MDM、Mean/Std 和实际 normalization。
绑定 val manifest 只是固定交付身份，val 数组未参与统计。此文件是输入预检报告，不是 checkpoint。
新训练入口将同一构建流程得到的来源和统计写入新 checkpoint；本轮没有创建研究 checkpoint。

MDM Mean/Std 保留原文件：Mean hash `26e136555dab04c94a129d446c26e6b9939cbf045fbf77bcf5462c1fb5a2001c`，
Std hash `6565a65ed9b31e23c328829a309e1c482be8b85fd23b43d65451a9b19a917f40`。未在小面板上重拟合。
原 backbone hash `ce43c1d4232fc2b897999d7996c2d2b889791d99585d87e1289af773ab56ab93`。

## 采样接口

采样 CLI 同样要求 `--canonical-package`，搭配交付的 val 三份清单和 `sampling_cases.json`。
只接收新 canonical 管线训练的 checkpoint，核对 package、MDM、Mean/Std 与 checkpoint 来源及 normalization 摘要。
每条 GT 直接读取已绑定的 `joints.npy`；不从 30 Hz joints 插值，也不从输出重新对齐到目标。

- paired：审核 caption + 对应 IMU。
- text_only：完整残差 strict null。
- imu_only：保留审核 caption 作为追溯信息，通过 `text_condition_present=false` 将文本 embedding 完整屏蔽；不是仅编码空字符串。
- backbone_bypass：在同一 frozen 模型中临时恢复原 TransformerEncoder，跳过控制准备 hook，异常时也恢复受控 encoder。沿用相同文本 CFG 和寻址噪声。
- numeric_zero：归一化 IMU tensor 全零，presence 保留；不是物理静止。
- physical_static：原始 acceleration 为零、首次观测 orientation 固定，使用新训练统计归一化；不求解目标 GT 标定。

保留原 `random_plan_case_id`、seed 和完整长度，寻址采样按每条有效长度执行。
输出增加真正送入模型的 `normalized_imu`、text/IMU presence；保留完整 motion/features、直接 GT、时间/mask、全部噪声 hash、checkpoint/provenance 与资源记录。
numeric_zero 的 raw acceleration/orientation 仍是来源参考；实际条件以保存的 normalized tensor 为准。
本轮 36 个槽位仅做输入装配，报告明确为 `input_ready_not_sampled` / `address_reserved_not_drawn`，没有制造随机计划执行或生成成功记录。

## 已执行命令

以下预检使用 CPU，`--input-only-report` 在 MDM 导入、模型创建和设备检查前退出。报告中的训练参数是预检解析值；没有选定实际 GPU。
输出目录必须全新，重放时更换报告路径，不能覆盖本批产物。

```bash
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python scripts/train_mdm_imu_control.py \
  --canonical-package docs/itm_rework/protocol/canonical_b0_20260925_v3/package.json \
  --manifest docs/itm_rework/protocol/canonical_b0_20260925_v3/train_motion.jsonl \
  --standard-imu-manifest docs/itm_rework/protocol/canonical_b0_20260925_v3/train_imu.jsonl \
  --caption-plan docs/itm_rework/protocol/canonical_b0_20260925_v3/train_caption_plans.jsonl \
  --contract-version itm-rework-v0.3-draft --mdm-root outputs/mdm/runtime_root \
  --mdm-checkpoint outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/model000475000.pt \
  --mdm-args outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json \
  --output outputs/itm_rework/method/canonical_b0_pilot_20260925_v1/model.pt \
  --epochs 100 --batch-size 3 --seed 20260925 --device cpu \
  --input-only-report docs/itm_rework/method/canonical_training_input_20260925_v1.json

OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python scripts/check_mdm_canonical_handoff.py \
  --package docs/itm_rework/protocol/canonical_b0_20260925_v3/package.json \
  --training-report docs/itm_rework/method/canonical_training_input_20260925_v1.json \
  --output docs/itm_rework/method/canonical_input_replay_20260925_v1.json
```

联合回归 **143 passed, 23 warnings**，既有 PyTorch 提示；日志 `logs/canonical_input_20260925_v2.txt`。
代码、依赖和两份输入产物摘要见 `code_snapshot_canonical_b0_v1.json`。
初次联合 142 项日志和定向 52 项结果与最终回归区分；新增 CLI 预检退出测试后复跑得到 143。
新数据集成测试在本机使用真实六条交付；没有该数据的环境会显式 skip，不伪装成已检查。
微型模型 fixture 验证 IMU-only 文本不敏感、顺序/批量 CFG、raw bypass 与原 encoder 的文本 CFG 一致、异常恢复；不计为真实模型能力或完整采样验证。
旧 caption 与 cache migration 两个脚本在临时新目录重放，两个 JSON 均与旧保存报告逐字段相等。

最终联合回归命令：

```bash
OMP_NUM_THREADS=1 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_canonical_conditions.py tests/test_mdm_training_provenance.py tests/test_mdm_caption_conditions.py tests/test_mdm_conditions.py tests/test_mdm_rework.py tests/test_mdm_imu_control.py tests/test_backbones.py tests/test_mdm_control_conditions.py tests/test_mdm_control_suite.py tests/test_mdm_multiseed_control.py tests/test_mdm_joint_scale_sweep.py tests/test_rework_observation.py tests/test_rework_protocol.py tests/test_rework_random_plan.py tests/test_rework_runner.py tests/test_standard_imu.py tests/test_standard_imu_cache_audit.py tests/test_method_handoff_acceptance.py tests/test_static_imu.py tests/test_physical_time.py tests/test_arm_attributes.py tests/test_caption_time.py tests/test_source_crop_support.py tests/test_humanml_clock.py tests/test_humanml_prefix.py tests/test_prefix_cache_rebuild.py tests/test_canonical_pair.py
```

## 下一步与边界

请 02 独立重放六条输入、398 帧归一化及 36 个输入槽位。包内五条语义不确定和一条方向冲突未提升准入；历史四份加速度失败未更改。
下一步实际 B0 训练/生成仍等待 **200 updates + 36 完整采样 / 合计最多 4 GPU 小时**的用户预算授权和执行前 GPU 检查。
01 在执行前还需落实协议中的时间/显存/更新次数上限与失败部分产物保存；本批只交付输入与条件接口，不声称已验收整套受限运行调度。
不能直接移除预检参数就在未授权情况下运行。本轮未复核 GPU 驱动，也没有为预算失败安排自动重试或 CPU 替代。
03 可以引用限定合成数据接入和软件正确性，尚无新 B0/B/C 生成结果。
