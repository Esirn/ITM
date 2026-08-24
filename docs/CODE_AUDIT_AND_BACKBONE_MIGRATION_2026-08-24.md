# ITM代码审查与Text-to-Motion基座迁移评估

## 1. 当前实现状态

当前主模型是冻结MDM/CLIP并训练IMU encoder与8个residual adapters。其优点是改动小、训练成本低、Text-only初始化可追溯到官方MDM；主要耦合点是MDM的8层`seqTransEncoder`、`predict_xstart=true`训练目标和HumanML3D 263D连续动作表示。

本轮修复了训练脚本中的重复前向：此前diffusion loss和关节辅助loss各调用一次模型，两次调用会独立随机执行text/IMU dropout。现在二者复用同一次predicted x0，减少计算，并保证同一batch中的各项loss针对同一组条件。Stage-4的frozen Text-only anchor仍需要额外一次前向，这是其目标定义所必需的。

## 2. 不应直接改动当前checkpoint的结构问题

### 2.1 多传感器过早平均

当前流程先给每个传感器加入部位embedding，然后在进入时序Transformer前对有效传感器做平均。head-only没有问题，但wrists会较早丢失左右腕的独立身份和相位关系。

下一版应保留`[B,T,S,D]`传感器token，先做跨传感器注意力，再生成逐帧控制token。为兼容可变数量传感器，可使用sensor mask屏蔽缺失槽位。该变化会改变IMU encoder参数和行为，需要训练新checkpoint，不能直接加载Stage-2b encoder权重作为等价模型。

### 2.2 文本只有全局条件token

MDM把CLIP整句编码压缩为一个全局向量。`swinging arms`等词语没有与肩、肘、腕建立显式对应，因此细粒度文本编辑不稳定。可行方向包括保留token-level文本特征、body-part query或局部cross-attention，但这些都属于新架构实验。

### 2.3 控制特征在每层完全相同

8个adapter不同，但输入均为同一份`C_imu`。可考虑为浅层、中层和深层增加独立投影或尺度门控，使低层更关注局部轨迹、高层更关注动作语义。第一步可只增加可学习的每层标量门控，作为低风险消融。

### 2.4 关节空间proxy尚未形成传感器闭环

当前trajectory、velocity和acceleration指标来自HumanML 22关节。严格做法是将生成263D恢复或拟合到SMPL，再按虚拟传感器安装位置重新计算加速度和方向，最后与输入IMU比较。该工作比继续微调loss权重更值得优先完成，因为它直接决定论文能否准确声称“IMU一致性”。

### 2.5 外部MDM实现耦合

训练和采样脚本直接导入MDM仓库、访问`seqTransEncoder`并依赖其diffusion对象。迁移前应抽象以下最小接口：

```text
encode_text(caption) -> text condition
prepare_motion(x) -> backbone motion representation
predict(x_t, timestep, text, optional_control) -> prediction
sample(text, length, seed, optional_control) -> motion
decode_motion(motion) -> [T, J, 3]
```

IMU预处理、sensor mask、条件交换spec和结果序列化应留在backbone接口之外，以便不同基座共享同一实验协议。

## 3. 本地候选基座盘点

截至2026-08-24，下列源码目录有效，但没有发现本地预训练checkpoint。

| 候选 | 表示/生成方式 | IMU接入难度 | 当前判断 |
| --- | --- | --- | --- |
| MotionLab | flow/diffusion式统一生成与编辑，原生支持`text_hint` | 中 | 首选试验候选，任务接口最接近Text+控制 |
| LGTM | 局部文本编码与全身优化的两阶段diffusion | 中到高 | 对未观测身体语义补全最有研究价值 |
| MotionDiffuse | 连续扩散生成 | 中 | 与MDM接近，但作为更强基座的收益不明确 |
| MotionGPT | VQ motion tokens与语言模型 | 高 | 时间下采样约4倍，不利于逐帧IMU控制，暂不优先 |

### MotionLab

优势是已有text、hint和text+hint任务路径，理论上可把IMU encoder输出映射为其hint/control条件，而不必重新发明多条件采样协议。风险是依赖Python 3.9、PyTorch 2.1/CUDA 11.8，并且仓库提示较新代码存在复现修复历史。需要单独conda环境，不应直接污染`itm`环境。

### LGTM

LGTM把文本拆成身体部位描述，正好对应当前“head IMU不足时由文本补全上肢”的弱点。合理接法不是把同一IMU向量广播到所有部位，而是将传感器部位映射到LGTM局部分支：head控制head/torso相关分支，wrists控制arm分支，未观测分支主要依赖文本。风险是需要其预训练LGTM、TMR encoders和part-level annotations，且控制注入位置需要重新设计。

### MotionGPT

MotionGPT先把动作离散为低帧率motion tokens。IMU是20 FPS连续信号，若直接对齐到约5 FPS token会丢失高频加速度和步态细节；若改tokenizer又需要大规模重训。因此它更适合未来研究“语言推理与动作语义”，不适合当前第一轮迁移。

## 4. 推荐迁移顺序

### Phase 0：先补当前论文证据

1. 完成generated motion到SMPL/virtual IMU的闭环评价。
2. 补严格loss ablation：diffusion only、+trajectory、+velocity、+jerk。
3. 固化当前672-sample evaluator和condition-swapping suite。

这些工作应先完成，否则更换基座后无法判断提升来自新基座、控制方法还是评价变化。

### Phase 1：Text-only复现门槛

为候选基座建立独立环境并只运行官方Text-only：

1. 获取官方checkpoint及其文本编码器、motion encoder/evaluator依赖。
2. 用固定的20条prompts生成动作。
3. 在HumanML3D官方或同一672子集上核验FID、Matching和R-precision。
4. 只有Text-only质量达到其官方日志附近，才进入IMU接入。

建议先做MotionLab；如果其checkpoint体积超过3 GB或需要新增大数据，下载前单独确认。

### Phase 2：统一backbone adapter

新增独立模块实现上面的最小接口，保留当前MDM实现作为`MDMBackbone`。MotionLab或LGTM实现第二个adapter。第一轮只要求Text-only采样、统一22关节输出和共享seed，不立即接IMU。

### Phase 3：IMU控制迁移

优先使用“冻结新基座、只训练IMU控制分支”的公平协议。对MotionLab先尝试其`text_hint`入口；对LGTM则按身体部位注入局部控制。仍使用head/wrists、相同7009训练映射和相同condition-swapping suite。

## 5. 是否现在开始迁移

可以开始，但当前适合做资产准备、Text-only复现和backbone接口解耦，不适合立刻启动融合训练。原因是本地候选源码均缺少预训练权重，且当前论文最关键的generated-IMU闭环指标仍未完成。

推荐决策是：主论文继续以Stage-2b/MDM为稳定基线；并行启动MotionLab Text-only复现作为“更强基座可迁移性”探索。LGTM作为第二候选，用于验证body-part text representation是否改善head-only文本补全。除非新基座在Text-only门槛和小规模IMU smoke实验中都明显优于MDM，否则不替换当前论文主模型。
