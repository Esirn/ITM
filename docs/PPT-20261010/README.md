# 2026-10-10 导师汇报材料

为 **15–20 分钟**讨论准备。已按实际仓库状态制作；所有结果来自既有产物，本次没有新增训练、采样或 GPU 查询。

## 打开材料

- **主稿：**[ITM_导师汇报_20261010.pptx](ITM_导师汇报_20261010.pptx)，25 页，其中 01–22 为主稿、23–25 为附录。文字、表格、框图可编辑，逐页讲稿也已写入备注。14、15 页嵌入完整动作 MP4。
- **静态备份：**[PDF](ITM_导师汇报_20261010.pdf)，同布局预览，视频显示为固定中点封面。
- **备讲：**[汇报逐页讲稿](汇报逐页讲稿.md)、[讨论要点与答疑](讨论要点与答疑.md)。
- **动作：**[离线动作对照页](visualizations/offline_review.html)，下载后直接用浏览器打开，无需 uvicorn、网络或数据服务器；六个固定历史例子可选，可切换骨盆居中／原始世界坐标。
- **图 1：**[中英文生图提示词](Figure1_生图提示词.md)、[可编辑 SVG 框图](figures/Figure1_method.svg)、[PNG](figures/Figure1_method.png)。当前框图也在 PPT 第 5 页。
- **检查：**[全部页面预览](previews/index.html)、[缩略总览](previews/contact_sheet.png)、[来源](sources/SOURCES.md)、[输入 SHA 清单](sources/source_manifest.json)、[验证报告](sources/verification.json)。

## 建议讲述顺序

| 页码 | 内容 | 建议时间 |
| --- | --- | --- |
| 01–04 | 动机、任务、应用、整体进展 | 2.5 分钟 |
| 05–11 | 当前方法、数据、训练与对比设置 | 5 分钟 |
| 12–16 | 实际结果、动作、辅助损失取舍 | 5 分钟 |
| 17–21 | 论文缺口、慢进度复盘、收敛路线 | 4 分钟 |
| 22 | 请导师确认的关键选择 | 1 分钟后转讨论 |
| 23–25 | 公式、更多指标、来源 | 按问题调用 |

时间紧时，06、08、09、11、19、20 页只讲底部结论。讲动作时先展示验证例，再说明弓步失败；熟悉的出拳／右腿后举保留在离线页和独立 MP4，均明确标记为训练参考。

## 汇报最重要的判断

当前已实现“文本＋单头部 IMU → 全身动作”的完整工程原型，在 **128 train＋32 val 的小规模合成配对开发集**上完成四方法共同比较。原 U100 相对纯文本 MDM 的头部位置误差下降 19.0%、速度误差下降 10.9%，但头部 jerk 增加 73.3%。相对简单 IMU-only 或简单融合仍有多项代价，不能宣称全面优越。

保留原 U100，未采用本轮局部监督候选。当前模型仍用 head＋wrists 两配置联合训练，论文主线评 head；尚未重训为 head-only。优先补官方 head 重建基线、文本语义／条件贡献、核心训练消融，再扩大数据与正式测试。现有开发结果不替代正式 test、多种子、真实设备或行为验证。

基金摘要只用作一页研究背景：智能耳机惯性信号、稀疏观测、条件扩散与性能均衡有关联；本文仍保留原本联合条件动作生成内容，不据此声称双耳校准、实时精准下肢重建或临床疗效。

## 播放与兼容

PPT 视频嵌入文件内部，不依赖原输出路径；不同 Office／WPS 对内嵌视频支持可能不同。若点击不能播放，使用同目录独立 [拍手验证 MP4](visualizations/val_000382.mp4)、[弓步失败 MP4](visualizations/train_000506.mp4)、[出拳转身 MP4](visualizations/train_000027.mp4)、[右腿后举 MP4](visualizations/train_004728.mp4)，或打开离线动作页。PDF 为静态备份。

预览 PDF／PNG 由同一布局程序绘制，**不是 PowerPoint 或 WPS 桌面渲染截图**。已核验文档结构、数字、来源哈希、媒体完整帧数和布局边界，并人工检查页面预览；本机没有 Office 渲染器。建议会前在实际演示软件中打开一次。字体使用 Microsoft YaHei，未嵌入字体；替代字体可能改变视觉效果。

默认动作显示仅为可视化去掉各模型自己的骨盆 XZ 平移，保留高度；所有数值指标维持原评测定义。世界坐标模式可以看整体漂移。头部红点是关节 15 的展示标记，并非精确传感器安装点。

## 文件与复现

生成脚本：[build_briefing.py](build_briefing.py)；可编辑文字：[briefing_content.py](briefing_content.py)；依赖：[requirements.txt](requirements.txt)。脚本只读取实验产物，写入本目录；不存在模型调用。

```bash
MPLCONFIGDIR=/tmp/itm-ppt-mpl OMP_NUM_THREADS=2 \
  /home/a200/miniconda3/envs/itm/bin/python docs/PPT-20261010/build_briefing.py
/home/a200/miniconda3/envs/itm/bin/python docs/PPT-20261010/verify_briefing.py
```

生成需要原本地 `outputs/`；已制作好的 PPT、PDF、图片、视频与离线 HTML 可以整体拷走使用，不需要实验原文件。当前机器的新增 PPT 依赖仅装在 `/tmp/itm-pptdeps`，没有修改训练环境。

## Git 边界

提交源稿、生成／验证脚本、依赖声明、结果表、来源 SHA 清单与验证报告，并保留本次汇报的最终 PPT、PDF、可导出图片、独立 MP4 和离线 HTML。离线 HTML 内嵌六例动作，是有意保留的独立审阅交付，新 checkout 不依赖实验 `outputs/` 即可播放。

本目录 `.gitignore` 忽略可再生成的 `previews/`、逐文本框 `sources/layout_checks.json`，以及与离线 HTML 重复的 `visualizations/motion_data.json`；Python 缓存沿用仓库规则。这些文件继续留在本地。对应预览链接在新 checkout 中需要运行生成脚本后才能使用；模型权重、原始数据和实验缓存继续由仓库根规则忽略。
