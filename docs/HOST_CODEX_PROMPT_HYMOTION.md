# 给宿主机 Codex 的提示词

将下面代码块粘贴给宿主机Codex，并在末尾补充实际路径、设备和资产获取方式。

```text
请接手ITM项目的HY-Motion大显存迁移。先完整阅读：

1. docs/HOST_HYMOTION_HANDOFF_2026-08-26.md
2. docs/HYMOTION_MIGRATION.md
3. docs/ENVIRONMENT.md

仓库应位于master分支。先检查git状态、实际路径、GPU、HY-Motion源码状态和Lite
checkpoint SHA，不要假定交接文档中的当前机器路径在宿主机有效，
不要修改或reset用户已有改动，也不要修改外部HY-Motion仓库。

HumanML3D的完整数据根目录与MDM统计量目录必须分开：完整数据使用包含29,228条标准split
样本的目录；现有MDM复现使用另一个目录里哈希一致的Mean/Std。不要把当前仅含一条
new_joint_vecs/new_joints文件的MDM目录当作完整数据根。缺少额外04****文件不影响标准
split；缺少mean_motion/std_motion不阻塞HY-Motion，但复跑MotionLab时需要补齐。

按交接文档创建/核验独立的itm与hymotion环境，提取clean HY-Motion runtime，运行全量
pytest和60帧官方Text-only smoke。遇到路径、资产或环境错误先诊断和报告，不要通过盲目
替换路径或跳过文本编码器来绕过。

验证通过后开始实现HY-Motion Lite Text+IMU预筛：冻结HY-Motion和文本编码器，实现六槽
12D IMU的sensor attention与temporal encoder，并通过零初始化residual注入6个double-stream
和12个single-stream block。double-stream只改motion_feat；single-stream只改motion prefix。
支持head joint 15和wrists joints 20/21，保留sensor mask、条件dropout和相同seed/noise的
Text-only、paired、zero、shuffled对照。

先写单元测试和16条backward smoke，再做512 train/128 val、2 epochs预筛。训练和采样入口
必须接受--device，启动前检查GPU，不抢占或终止其他进程。保存配置、seed、checkpoint hash、
分项loss、显存和评估结果。只有paired相对shuffled改善率达到70%、文本语义基本保留且jerk
不恶化，才考虑7009条全量训练。不要引入Text/IMU之外模态，不要先解冻backbone，不要把
joint-mounted acceleration称为真实设备IMU误差。

持续实现、测试和记录，直到预筛完成或出现必须由用户解决的真实阻塞。

```
