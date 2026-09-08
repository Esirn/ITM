# HY-Motion 复现与 ITM 迁移

## 当前结论

已使用官方提交 `9993ccd8aeba1f2d34c1061cc2282d14d5193055` 的干净代码完成
HY-Motion 1.0 Lite Text-only 采样。sshfs 挂载源码中的文本编码器被人为修改为跳过
Qwen 并返回零上下文，因此 ITM 不直接导入挂载工作树，而由
`scripts/prepare_hymotion_workspace.py` 从其 Git 对象提取只读的官方代码快照。checkpoint、
统计量、Qwen、CLIP 和 WoodenMesh 资产仍从挂载目录只读加载。

固定提示词 `a person walks forward while swinging both arms`、60 帧、seed 1234 的 smoke
结果位于 `outputs/hymotion/text_only_smoke/walk_swing_60.npz`。GPU 1 上核心采样耗时
1.93 秒，PyTorch 峰值分配显存约 19.6 GB。输出包含有限数值的 201D motion、22关节
6D rotation、root translation、root rotation matrix 和52个 WoodenMesh joints。

## 表示与迁移价值

HY-Motion 每帧201维中，官方decoder明确读取前3维根平移和随后132维的22关节6D
旋转；剩余66维不参与当前FK解码，不能在未核验训练数据构造前武断命名。官方 body
model 用旋转和平移做FK。这比 HumanML3D 263D 只恢复22关节位置更适合正式虚拟IMU评价，
因为可从全局关节旋转计算传感器orientation，而无需从位置猜测骨骼twist。

`HYMotionBackbone` 已接入统一接口，但首版故意限制为单样本完整采样。`encode_text` 和
逐步 `predict` 仍留在隔离进程中，避免 HY-Motion 的 Transformers/Qwen 依赖污染
`itm` 环境。正式融合前应先在固定prompts上做Text-only人工检查，并验证官方旋转经
WoodenMesh FK导出的虚拟IMU round trip。

## 运行方式

```bash
conda run -n itm python scripts/prepare_hymotion_workspace.py \
  --source /home/a200/mount/a40/relatedworks/HY-Motion-1.0 \
  --output outputs/hymotion/runtime_root

conda run --no-capture-output -n hymotion python scripts/sample_hymotion_text.py \
  --hymotion-root /home/a200/mount/a40/relatedworks/HY-Motion-1.0 \
  --runtime-root outputs/hymotion/runtime_root --model lite \
  --text "a person walks forward while swinging both arms" \
  --frames 60 --seed 1234 --device cuda:1 \
  --output outputs/hymotion/text_only_smoke/walk_swing_60.npz

conda run -n itm python scripts/render_hymotion_results.py \
  --results outputs/hymotion/text_only_smoke/walk_swing_60.npz \
  --output outputs/hymotion/text_only_smoke/walk_swing_60.gif
```

独立 `hymotion` 环境基于 `itm` 克隆，并使用 `transformers==4.53.3`、
`numpy==1.26.4` 和 `openai==1.78.1`。Lite 单次推理已经接近24 GB卡的实用上限；
冻结主干训练IMU控制器时，应优先在宿主机大显存GPU上运行。

## 下一步门槛

1. 生成走、跑、坐、转身、跳跃等固定prompts并人工确认Text-only质量。
2. 以AMASS/标准虚拟IMU验证22关节旋转到sensor orientation/acceleration的闭环。
3. 冻结MotionFlow和文本编码器，在宿主机训练sensor-attention加时序编码器。
4. 将零初始化IMU residual注入6个double-stream和12个single-stream block；后者只改
   motion token前缀，不改text token。
5. 先做512条预筛；只有paired显著优于shuffled且Text-only语义基本保留，才跑7009条。

当前结果仅证明官方Text-only路径和统一输出可运行，尚不能证明HY-Motion上的Text+IMU融合。

## 固定提示词定性检查

另外生成了四类固定样例，均位于 `outputs/hymotion/text_only_fixed_prompts/`，每条同时
保存 `.npz`、`.json` 和可直接观看的 `.gif`：

| 文件 | 文本 | 抽帧初查 |
|---|---|---|
| `run_fast.gif` | a person runs forward quickly | 有跑动姿态，但部分相位肢体投影重叠 |
| `sit_down.gif` | a person walks forward and sits down on a chair | 由站立过渡到坐姿，语义最清楚 |
| `turn_walk.gif` | a person walks forward and makes a left turn | 朝向和步态发生变化，需结合完整GIF判断转向方向 |
| `jump.gif` | a person jumps up with both feet and lands | 当前seed的离地语义较弱，是应保留的失败样例 |

这些样例说明 Lite 的动作结构明显优于早期回归模型，但不能仅凭模型规模认定所有文本
语义均可靠。下一阶段应先做多seed小规模Text-only检查，再训练IMU控制预筛；坐下成功和
跳跃失败都应保留，避免只展示挑选后的成功样例。

## 旋转到虚拟IMU

采样脚本同时将22个local 6D rotations按官方WoodenMesh父子关系累积为
`global_rotations_mat`。`scripts/export_hymotion_virtual_imu.py` 可导出head joint 15或
wrists joints 20/21的全局orientation，以及由关节位置二阶差分得到的acceleration。
这条路径无需SMPL fitting，因此比MDM的位置代理更直接；但目前仍是joint-mounted
proxy，没有真实设备的sensor-to-bone外参，也没有加入重力和传感器噪声，论文中不能把它
称为真实IMU误差。下一步需在AMASS paired motion上验证坐标系和外参后再用于正式比较。
