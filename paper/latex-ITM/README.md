# ITM 中文 IEEE LaTeX 稿

编译入口为 `main-cn.tex`，正文按章节位于 `sec-cn/`，英文 `sec/` 文件保留原状。
内容转换自 `paper/论文草稿.md`；实验数字、方法定位和失败结果均保留。
`IEEEtran.cls` 原样复制自用户提供的 IEEE Transactions 模板。

## Windows / TeXstudio

推荐安装完整 TeX Live（含中文语言支持），所有源文件使用 UTF-8。
打开 `main-cn.tex`，在 TeXstudio 设置中选择 XeLaTeX 为默认编译器、BibTeX 为文献工具。
按以下顺序运行，或配置构建链自动执行：

```text
xelatex main-cn.tex
bibtex main-cn
xelatex main-cn.tex
xelatex main-cn.tex
```

也可以在该目录运行 `latexmk -xelatex main-cn.tex`。
不要直接编译 `sec-cn/` 下的子文件；它们的 TeX root 已指向主文件。
若缺少包，安装 `ctex`、`fandol`、`IEEEtran`（含 `IEEEtran.bst`）、`booktabs`、
`tabularx`、`seqsplit`、`hyperref`。Fandol 字体由 TeX Live 提供，无需依赖宋体等系统字体。

## Overleaf

将整个 `latex-ITM` 目录内容压缩上传为项目，主文档选 `main-cn.tex`，编译器选 XeLaTeX。
文献为 `sec-cn/references.bib`，使用发行版自带的 `IEEEtran.bst` 与 BibTeX。
没有外部绝对路径或模型/数据依赖；正文中出现的实验目录只是文字说明。

## 编辑说明

- 七个章节文件对应摘要、引言、相关工作、方法、实验、讨论、结论。
- 16 张表已转换为可编辑的 booktabs/tabularx 表格；宽表使用跨双栏 `table*`。
- 四张定性图为带图注的占位框。插入最终静态 PDF/PNG 后替换占位框为 `\includegraphics`。
- 正文 citation key 已转换为 `\cite`，文献原样来自 `paper/references.bib`。
- 作者、单位、目标期刊和最终图件尚待填写；没有添加虚构署名、期刊信息或实验结果。
- 这是中文内部草稿，投稿时仍需依目标期刊要求改为相应语言与最终格式。

本机没有 XeLaTeX/BibTeX，当前只完成结构、引用与内容一致性静态检查，尚未验证最终分页和表格溢出。
