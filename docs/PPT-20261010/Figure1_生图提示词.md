# Figure 1：方法总览的生图提示词

可以直接把下面内容交给其他AI生成草图。目录内另提供可编辑的SVG和PPT原生框图，可直接用于本次汇报。文字较多时，建议让生图AI只画布局和图标，再在PPT中添加准确标签。

## 必须忠于当前实现的结构

输入为文本、**单个头部IMU的世界线性加速度＋全局朝向**和扩散噪声。头部位置/根轨迹不是模型输入。输出为20Hz全身动作，内部为HumanML263表示，显示22关节骨架。

文本经过冻结CLIP；动作先验为冻结MDM Transformer。训练IMU时序编码器和逐层残差适配器。编码器为2层、512维、8头Transformer，带以秒为单位的时间编码。

当前关键控制表示：真实IMU与物理静态参照分别通过**共享权重**的编码器，取编码差`C=E(I)-E(I_ref)`。参照加速度为物理0，朝向固定为首帧；它仍要使用相同归一化，不能画成简单数值零向量。不是先做原始输入差再编码。

MDM共8层，示意画3层并标注“×8”。每层先执行冻结Transformer，再加`A_l(C)`。适配器是训练的线性层，权重在训练开始时为0；当前训练后不再全为0。静态差分下bias固定为0。文本/时间步的前置token不被IMU残差改写。

扩散器预测`x0`，通过1000步DDPM迭代生成。训练损失为现阶段原U100的263维标准化`x0` MSE。**头部局部位置/速度辅助损失候选没有被采用，不能画进主方法。** 可在另一个“已探索但未采用”小面板中介绍。

医院合作是潜在应用背景，不是当前实验输入；不能把患者数据、诊断标签或临床疗效画入已实现方法。

## 中文提示词：学术方法图

绘制一张适合计算机视觉论文与导师汇报的学术方法总览图，横向16:9，白色背景，线条清晰，使用深蓝、青绿和橙色三种克制的配色，优先矢量风格。图分为两部分：(a) Joint-conditioned full-body motion generation；(b) Shared-encoder reference-difference control。

左侧画一个佩戴单个头部IMU的简洁人物轮廓，仅头部有传感器标记，旁边放文本气泡“a person walks forward while swinging both arms”。头部传感器连接两组时间信号：加速度a(t)与朝向R(t)，不要画成位置轨迹输入。文本经过带锁图标的Frozen CLIP送入中央Frozen MDM；扩散噪声x_t和时间步t也进入MDM。

下方控制支路展示两条路径：Observed IMU I与Static reference I_ref，共享同一时序编码器E_theta（两个编码器框之间用共享权重标记连接）。参照的小图展示加速度为物理0、朝向等于首帧并随时间保持。两个编码输出连接到减法节点，得到C=E_theta(I)-E_theta(I_ref)。共享编码器标注“Temporal Transformer, 2 layers, 512 dim, time in seconds”，用青绿色表示可训练。

中央MDM用三层堆叠的Transformer矩形表示，并标注“×8, frozen”。每个冻结层之后接一个小型可训练线性适配器A_l(C)，用加号节点把控制残差加入隐藏状态，标注“zero-initialized at training start”。不要复制整条MDM主干，不要画ControlNet的完整副本，不要画尚不存在的跨模态cross-attention。MDM上方输入文本/时间token，在图注中说明它们不接收IMU残差。

右侧展示3至4个连续的人体骨架动作帧，标注“Full-body motion, 20 Hz, 22-joint visualization”。底部用小回环表示“Predict x0 → DDPM update → 1000 steps”。训练监督GT从图底部用虚线箭头连到x0 reconstruction loss，并明确“training only”；不能从GT连到IMU编码器或推理输入。

右下角留一个简短说明区域：“Current prototype; synthetic paired data; head primary”。保持所有箭头方向明确，文字尽量短，输出可编辑SVG优先。

## English prompt

Create a clean vector-style academic pipeline diagram on a white background, landscape 16:9. Use navy for frozen modules, teal for trainable modules, and orange for IMU signals. Two panels: (a) Text + one head IMU to full-body motion, (b) Shared-encoder reference-difference control. Draw only one IMU on the head. Inputs are a text prompt, world linear acceleration a(t), global orientation R(t), diffusion state x_t and timestep t. Do not use ground-truth joint positions or root trajectories as inference inputs.

Text passes through Frozen CLIP into a Frozen MDM Transformer. Observed IMU I and a physical static reference I_ref pass separately through a shared-weight temporal Transformer E_theta. The reference has physical zero acceleration and first-frame orientation held constant, using the same normalization as I. Subtract the encoded features: C = E_theta(I) - E_theta(I_ref). E_theta has 2 layers, hidden dimension 512, 8 heads, and sinusoidal time encoding in seconds. This is encoded-feature subtraction, not an encoder applied to raw input differences.

Show a frozen Transformer layer followed by additive control A_l(C), repeated over 8 MDM layers. Adapters are trainable linear maps, zero-initialized at the beginning of training, with fixed-zero bias in this reference-difference variant. Do not depict an entire duplicated ControlNet backbone or an unimplemented cross-attention module. Output is predicted x0, iterated with a 1000-step DDPM sampler and decoded into full-body motion at 20 Hz, visualized with 22 joints. Show the text/timestep token separately, without an IMU residual connection.

Ground-truth motion connects only through a dashed training-supervision arrow to normalized x0 MSE. The explored head-position/velocity auxiliary loss is not part of the currently retained main model. Mark this as an offline synthetic-paired-data research prototype. Use short editable labels and consistent directional arrows. No clinical-efficacy claims.

## Negative prompt / 禁止误画

- 六个传感器都装在人身上：本次任务图仅头部一个。
- GT头部轨迹、GT根轨迹、患者诊断信息作为推理输入。
- 已完成真实患者实验、在线实时系统、临床诊断或治疗效果。
- 将静态参照画成归一化后的数值零、将参照差分理解为消除所有静态身体姿态。
- 将所有根轨迹或朝向信息从输出中删除；当前只在部分指标中去根平移。
- 用完整复制的MDM分支代替本实现的逐层线性适配器。
- 把零初始化解释为训练后仍输出零控制。
- 把本轮未采用的局部监督候选画为最终方法。

## Figure 1 图注建议

**中文：**当前原型以冻结的文本动作扩散先验为基础，利用共享时序编码器提取观测IMU相对物理静态参照的特征差，并通过逐层残差适配器注入生成器。文本指定动作意图，头部IMU提供连续运动信息；GT仅用于训练监督。本次实验使用离线合成配对数据，头部配置为主要评测方向。

**英文：**The current prototype augments a frozen text-to-motion diffusion prior with head-IMU control. A shared temporal encoder represents the difference between observed signals and a physical static reference, and trainable layer-wise adapters inject this control into the frozen generator. Ground-truth motion is used only for training supervision. Experiments currently use offline synthetic paired data with a head-primary evaluation scope.
