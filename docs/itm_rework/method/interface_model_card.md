# B0 / semantic-response 实现接口与模型卡

状态：代码与 CPU 正确性通过；只有一个 update 的 smoke checkpoint，**没有通过验证筛选的 B0 checkpoint**。
契约 `itm-rework-v0.1-draft`，同步和输出观测认证未完成。

## 结构、冻结与版本

MDM HumanML263、CLIP、8 层 512D Transformer 原样冻结。`frozen_eval` 关闭所有骨干 dropout，
仅 IMU encoder 和 adapters 进入 train。优化器只接收 requires_grad=True 的参数；测试核查其梯度边界。

B0 encoder：6 个固定槽位，每槽 12D（世界系安装顶点运动学加速度 xyz + global orientation row-major 9D）；
未激活槽位清零，拼接 6D mask，78→512→512 MLP，物理秒 sinusoidal 位置，两层 temporal Transformer。
只在 head=[4] / wrists=[0,1] 上训练和采样；不扩展联合配置。姿态不缩放，加速度逐 sensor/axis 归一化。
orientation 18/19 与 proxy joint 20/21 的区别按 02 契约保留。

| 结构 | encoder 参数 | adapters 参数 | 可训练总数 |
|---|---:|---:|---:|
| legacy_mean + additive | 6,314,496 | 2,101,248 | 8,415,744 |
| ordered_concat + additive（B0） | 6,607,872 | 2,101,248 | 8,709,120 |
| ordered_concat + FiLM 对照 | 6,607,872 | 4,202,496 | 10,810,368 |

FiLM 使用 `output + scale(control)*output + shift(control)`，全部门控，初始化为 identity。
参数量不同，不能宣称与 additive 等算力。尚未训练/比较 FiLM 性能。

config 缺少新字段时默认 `legacy_mean / legacy_null / none / additive`，旧 state_dict 原样加载。
新训练 CLI 默认 `ordered_concat / strict / physical_sinusoidal / additive`，checkpoint_schema=`itm-control-v2`。
恢复必须结构配置相同，不静默迁移 legacy→B0；`--resume` 是权重+归一化 warm start，fresh optimizer，非 exact resume。
`--null-mode strict` 可以在采样时显式覆盖旧 checkpoint，metadata 会记覆盖后的配置。

## 条件接口

MDM 的 `y` 输入：

- `imu[B,T,6,12]`、`sensor_mask[B,6]`、`imu_frame_mask[B,T]`。
- `imu_timestamps[B,T]` 单位秒，可省略为 arange(T)/20；不会自动修正数据对齐。
- `imu_condition_present[B]` 独立表示条件是否存在，不能用数值零替代；未提供时由有效 mask 和训练 dropout 决定。
- `imu_uncond` 为 bool 或 [B] flag，关闭完整 residual。无 sensor/无 frame 同样关闭。
- 缓存 `imu_control[B,T,D]` 必须携带 `imu_condition_present[B]`；frame mask 继续传入。
- `text_embed` 为 CLIP [1,B,D]；`uncond=True` 关闭文本。CFG 支持标量或 [B] 权重。

`factorized` 顺序执行 4 次，batched 执行一次 4B；legacy 为 3 次或一次 3B。
四分支公式按任务书；(wt,ws,wi)=(1,1,1)→f11，(wt,0,0)→text guidance，(0,ws,0)→IMU guidance。
这些是同一多模态 checkpoint 的条件消融，不是专门训练的最强 IMU-only。
临时 control 在异常和正常返回均清理；同一个模型对象不支持并发重入 forward。

## 训练目标与 B 分支

新默认 diffusion-only。trajectory-only 用 `--diffusion-loss-weight 0 --trajectory-loss-weight 1`。
`--trajectory-space global_joint_proxy|root_relative_joint_proxy` 仅选择 trajectory 项；velocity 为全局 active joint proxy。
`--derivative-units seconds` 用 20Hz：velocity MSE ×20²，excess jerk MSE ×20⁶；legacy_frame 保留历史公式。
相同权重不能跨这两种单位比较。jerk/velocity 新默认 0，尚无收益证据。
配对训练 IMU dropout 后辅助 GT regularizers 仍按传感器配置生效，这是**配对动作拟合**，不是 measurement loss。

相对响应入口 `itm.models.semantic_response.semantic_response_step` 接收同一 x_t/t、原始/目标 caption、
train-only 见证、可微 decode_joints；所有编辑分支强制文本和 IMU 启用。只在该步骤临时 eval 消除 dropout，
保持 student 梯度，并精确恢复模块模式。teacher 使用原始 backbone bypass，全程 no_grad。
Phi 首版仅为水平左右腕间距的帧平均（m），不声称它等于完整摆臂幅度或自然度。
响应 loss 首选有界方向 margin，exact_delta 可消融；teacher 变化小于声明阈值时不蒸馏，记录覆盖率。
两个编辑结果各计算 active-joint proxy；没有对允许改变的全身部位施加原始完整 GT。
标准配对 diffusion 分支继续训练。额外每 edit batch 为 2 student + 2 teacher forwards，记录 timestep 分桶。

CLI：`--response-manifest` 对接 02 compatibility schema 的 train/compatible/arm_spread_m 记录，
要求 witness 文件、残差、质量、tolerances、reviewer/rule metadata；首版每 motion/config 只支持一个 edit。
通过 `--response-loss-weight`、`--response-proxy-weight` 启用；默认关闭。
公共 manifest 目前为空，loader 会明确拒绝。synthetic fixture 测试只验证梯度和模式恢复，不能作为相容见证。
配置中的 margin/阈值是实现初值，未经过验证选择；正式训练必须由 02 冻结预算与规则。

## 采样 CLI 与输出

实际 CLI 为 `scripts/sample_mdm_imu_control.py`（命令见 commands.md）。
spec 是 JSON list，每条包含 motion_id、sensor_config、text、length，可选 imu_motion_id、imu_mode、三个 scale、random_plan_case_id。
`imu_mode=paired/shuffled/null/zero`；zero 仅数值诊断，不是静止 IMU。null 在所有 CFG IMU 分支都保留 false presence。
同一个 `random_plan_case_id` + seed + backbone hash + shape 重放同一初始及后续噪声。
新默认 addressed DDPM、按每条有效长度执行，避免原始 MDM 的 padding 依赖；代价是跨样本不能整体 batch。
同一条的 CFG 四分支仍可 batch。所有随机地址、噪声 hash、NumPy 版本、timestep_map 均存入 metadata。

新默认 crop_seconds；明确以缓存 clip-local 秒裁切。不拉伸 donor，短 donor 拒绝。
最多取前 196 帧，并记录原长/有效长；这是裁切策略，**不是 HumanML/IMU 同步认证**。
`legacy_stretch` 和 `--random-plan legacy` 仅供历史复现，后者不保证完整随机流重放。

NPZ 提供 `normalized_humanml263[B,263,1,T]`、由同一预测 RIC 解码的 `motion[B,T,22,3]`、20Hz 时间轴、frame_mask、
目标 GT/输入 acceleration/orientation/sensor_mask。`motion` 是 HumanML RIC joints，**不是已认证旋转 FK/SMPL mesh**。
acceleration/orientation 是输入条件，不能作为生成观测。02 必须从输出 motion/表示独立计算合适的 proxy 或认证算子。
JSON 提供完整配置、checkpoint hashes、训练/采样 seeds、guidance、运行时/显存、每条 case 和随机计划。
当前 CLI 对无效输入 fail-fast，未完成整批不会写成功产物；02 runner 应记录计划失败分母。
方法侧 smoke_records.jsonl 是诊断交接清单，不冒充公共 result.schema 已验收记录。

## 阶段限制

只有同步/坐标协议冻结、验证面板认证与明确训练/采样预算后，才能形成可比较 B0。
B 严格传感器 loss 依赖 certified output channels，现仅有 joint proxy。C 尚未实现/运行：输出通道认证为空，
旧 SMPL round-trip 加速度/旋转未过关；不能靠复制目标 IMU 或添加独立 head 绕过。
没有最终性能、编辑成功率、自然度改善或 B+C 结论。
