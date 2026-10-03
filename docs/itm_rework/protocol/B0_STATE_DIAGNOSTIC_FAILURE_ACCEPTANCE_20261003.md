# 02 → 01：V 预检失败交付核对

**接受 `ef659e0` 的失败交付，账本和来源核对通过，无新增软件返工项。** 此次在父入口首项GPU预检抛出异常，worker未启动，没有可评价的模型结果。**下一责任方为01：准备唯一新运行目录并申请新单次授权；获批后才重新预检或执行。** 原授权已消耗，旧运行目录及失败证据保留。相同实现不要求重复冻结或软件验收。

02本轮仅使用标准库读取文件、hash和账本，未导入模型框架、加载权重张量、运行模型或查询GPU，也未创建授权、重试或换卡。独立检查从冻结包提取完整槽位IDs，另重放01只读核验脚本并与其交付结果逐字段比对相同；没有改01方法文件或旧产物。

## 核对结果

| 项目 | 02独立结果 |
| --- | --- |
| 运行状态 | failed，RuntimeError: GPU identity/free-memory preflight failed |
| 实际整队耗时 | 1.413580719847232秒；deadline仍为启动后300秒 |
| 状态组 | 全48项not_run_after_stop，executed=false |
| 分支 | 全192项not_run_after_stop，executed=false |
| 软件锚点 | 全144项not_run_after_stop，未计为通过 |
| MDM / encoder / caption | 三类reserved/actual/completed及各calls共18个计数均为0 |
| 运行目录 | 只有runtime/status.json；无worker context/receipt/log、V输入或预测产物 |
| 原始/归档账本 | 逐字节一致，SHA256 `afbe17b7a43bcc37a0321b6a51dc2fe6127b768b7a62ea030e765b5e4597c4cb` |
| 来源和身份 | 全1,656份manifest绑定hash通过；授权/申请记录与运行身份、限额一致，6个交付索引文件hash通过 |
| 新模型输出 / 训练 / checkpoint / 动作 | 全0；无worker allocated显存峰值可报告，应为null而非0 |

implementation仍为 `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46`；V package仍为 `6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012`；R final200仍为 `91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642`。此前[独立软件验收](B0_STATE_DIAGNOSTIC_ACCEPTANCE_20260928.md)的135项CPU通过证据继续有效，本轮没有重跑或累加测试数量。

## 能确认与不能确认的原因

已绑定源码 `scripts/run_mdm_b0_bounded.py::preflight_gpu` 的首条查询为指定UUID的GPU信息；异常条件合并了**返回行数不为1、UUID不匹配、free MiB低于4096**。父日志的调用栈与该抛错行一致。由控制流可知，后续compute-process查询、worker context写入和监督器启动均未到达，与目录/零计数证据一致。

这只是**已绑定源码＋保存的调用栈所支持的路径判断**，原始查询stdout及其独立返回记录未保存，无法重建当时显存数值或确定哪一条件触发。不能将此次失败断言为GPU被占用、显存不足或UUID错误；事后查询也不能补作当时证据。02未补查GPU。

运行发生于2026-10-03，目录/授权文件的20260928名称沿用原申请身份，不将其改写为另一运行。授权记录保存历史批准事实，结合本次失败账本判定该单次授权已消耗；不修改旧记录以伪装为未使用。

## 后续交接

现有实现没有因本次预检拒绝而暴露新的模型/缓存/限额返工项。01可准备新申请，绑定最终manifest/package、唯一新目录和指定GPU，继续48状态/192分支/144锚点、72 IMU examples/12 caption调用、单GPU/24GiB/整队300秒；0训练/新完整采样/新噪声，旧授权不可续用。02无需再次确认同一实现。

若01选择补充预检失败的原始查询记录或拆分错误原因，应先完成该具体改动及CPU验证、更新manifest，再由02做差异验收；这是实现变更时的流程，**不是本次交付的必需返工项**，不能借此放宽预检或预算阈值。

本次未产生V结果，既有完整动作负例和冻结评价边界不变；模型有效性仍未认证。

证据：[独立核验JSON](b0_state_diagnostic_failure_acceptance_20261003_v1/acceptance.json)、[只读核验脚本](b0_state_diagnostic_failure_acceptance_20261003_v1/verify.py)、[运行日志](b0_state_diagnostic_failure_acceptance_20261003_v1/verification.txt)。复现命令为 `/home/a200/miniconda3/envs/itm/bin/python docs/itm_rework/protocol/b0_state_diagnostic_failure_acceptance_20261003_v1/verify.py`；已生成结果拒绝覆盖，需在新证据目录调整输出位置后重放。没有执行交接文档中的真实运行命令。
