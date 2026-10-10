# 头部主线：标准文本动作评测器准备情况

本轮只读核查本机官方MDM评测代码，没有下载资产或计算文本匹配/FID，不把关节误差当作语义指标。

来源：`/home/a200/0relatedworks/motion-diffusion-model/data_loaders/humanml/networks/evaluator_wrapper.py`及`prepare/download_t2m_evaluators.sh`、`prepare/download_glove.sh`。标准HumanML评测器需要`<checkpoints_dir>/t2m/text_mot_match/model/finest.tar`，包含movement/text/motion encoders；文本编码还依赖GloVe词向量与POS信息，使用配套归一化和unit_length4。

已在`0relatedworks`、本项目（含outputs）和`0proj/datasets`用包含隐藏/忽略文件的文件搜索核查`finest.tar`、`our_vab*`、`t2m.zip`、`*glove*.zip`，未找到这些资产。这是所列目录可用性检查，不声称机器其他位置或作者发布情况。

官方脚本有公开Google Drive下载链接，但还包含删除既有`t2m`/`kit`/`glove`目录的命令。后续应将所需公开资产下载到本项目独立目录，记录来源/文件SHA并安全解压，不直接执行原清理脚本。只取HumanML/t2m与对应词向量，不以本轮head任务为理由下载无关KIT资产。

准备后先以GT做输入/归一化/长度/文本token对齐检查；已有生成263维表示应先还原原motion Mean/Std，再转换到评测器所需归一化。不得为了嵌入模型随意拉伸时间或随机裁片；官方卷积下采样和整数长度规则需明确记录。32val是开发集且相对小，FID只作带样本量说明的开发诊断，正式主表需要更大的来源隔离test与共同评测协议。

头部任务仍要求文本描述的全身动作、顺序/保持阶段及合理运动。即使头部局部误差降低，不能据此认证腿部动作完成、语义正确或整体自然。
