# 头部主线：官方IMUPoser接口补充核查

在头部候选训练期间，只读检查本机官方源码，发现实际数据加载与模型输入宽度需额外适配。本文补充[先前核查](BASELINE_INTERFACE_AUDIT_20261009.md)，不修改旧冻结报告，不填实验成绩。

- `config.py` 的`h:[4]`和模型`n_input=12*len(config.joints_set)`支持构造12维头部模型，训练脚本按指定combo设置`joints_set`。
- 但`datasets/globalModelDataset.py`当前实现遍历所有`amass_combos`，每个样本保留五个槽位，以零屏蔽未观测传感器，然后将15维加速度与45维朝向拼成60维；`__getitem__`没有按`config.joints_set`截取。对于h配置，这与模型的12维入口不一致，不能只改数据路径就宣称官方训练脚本直接可用。
- 官方加速度除以`acc_scale=30`，不是本项目train-only z-score；head12维适配应输入原始世界加速度/30＋原全局朝向矩阵，不能重复套本项目IMU归一化。五槽位缺失屏蔽是原作者的数据设计，不能把零槽位误描述为额外真实观测。
- 官方入口默认`r6d=True`、MSE＋SMPL关节损失、Adam3e−4；官方RNN是512维两层双向LSTM。仅复制网络结构并用本项目263维监督，不能标为未经改动的官方IMUPoser。
- 原数据加载器按25Hz截窗，本项目共同输入是完整20Hz片段；正式比较需明确20Hz/真实长度适配和同来源划分。官方随机按片段划分的代码不直接沿用，应遵守本项目来源隔离。

后续实现选择：使用绑定官方RNN源码的head12维入口，加速度/30与朝向原值，24关节6D旋转监督＋公平SMPL FK；修正输入选择、20Hz与来源划分，并明确标识为本数据协议上的官方架构适配重训。source局部旋转的root仍需canonical Q转换。模型无平移输出，不填GT根轨迹再比较全局误差。所需SMPL/骨架映射必须先数值核验，本文尚未实现该适配或执行训练。

精确来源（只读，本机官方仓库）：`/home/a200/0relatedworks/IMUPoser/src/imuposer/config.py`、`models/LSTMs/RNN.py`、`models/LSTMs/IMUPoser_Model.py`、`datasets/globalModelDataset.py`、`datasets/utils.py`、`scripts/2. Train/1. Train Global Model.py`。这些是当前本机代码事实，不推断作者其他版本或已发布权重的行为。
