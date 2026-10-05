# 01 最新状态：F 新单次授权已消耗，预检检测到GPU计算进程（2026-10-05）

用户批准 `82d28a3` 后执行一次绑定的新队列；预检于0.976359秒因`Selected GPU already has a compute process`停止，父入口退出码1，worker未启动。第一项UUID/显存条件通过，第二项查询检测到指定GPU的计算进程；原始输出未保存，不报告具体PID或显存。本次命令经设备访问审批后仅启动一次。
200 updates/600 exposures/60 samples/240 traces/48 checks全部not_run_after_stop，模型调用0；没有重试、追加GPU查询、换卡或干预已有进程。本次及历史授权已消耗。
1,145份来源hash、授权/manifest身份、原始与归档账本及完整槽位CPU核验通过。原20261004失败证据未变；本次占用结论不能解释旧退出码9。方法实现及155项软件验收继续有效，F/B0有效性仍未认证。
详见 [失败交接](B0_FEATURE_BALANCE_RUN_FAILURE_20261005.md)。下一责任方02核对失败交付；后续申请须考虑GPU占用条件，使用新目录与新单次授权，不自动等待重试或终止其他任务。
以下为历史记录。

---

# 01 最新状态：F 失败交付已核对，申请新单次授权（2026-10-05）

已读取 `971909b`：[02失败核对](../protocol/B0_FEATURE_BALANCE_FAILURE_ACCEPTANCE_20261004.md)通过，无新增返工项；实现 `963a132` 和155项软件验收继续有效，无需重复验收。
01本轮只读核验1,145份绑定文件、最终v2 manifest副本及旧失败账本，均一致。新申请继续绑定implementation `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`、F package `a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`、原共同初始化及GPU `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
唯一新目录 `/home/a200/0proj/ITM/outputs/itm_rework/method/feature_balance_authorized_20261005_v1` 已确认不存在，留待获批后入口原子创建；[具体授权申请](B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261005.json)为`pending_user_approval`、`authorized=false`，实际执行授权尚未创建。
申请一次 **200 updates/600 exposures＋60条完整采样＋240条零额外调用trace＋48项软件检查**；训练自预检前起最多600秒，采样共享最多3000秒，整队最多3600秒；单GPU、allocated≤24GiB。首个规定故障即停，保留完整分母和产物，不重试、续训、补采、换卡、warmup或CPU回退。
此前退出码9的具体原因仍未确定，不能保证新预检通过。本轮未查询GPU或执行模型，旧目录/失败证据未变；本次及历史授权不可复用。依照用户本轮要求，下一步等待新的单次批准，获批后01预检执行，再交02核对/评价。模型有效性仍未认证。
以下为历史记录。

---

# 01 最新状态：F 单次授权已消耗，GPU 预检失败（2026-10-04）

用户批准 `72a01e9` 后执行一次F；在预检前起1.006989秒因指定UUID的`nvidia-smi`查询返回码9停止，父入口退出码1。训练worker未启动，模型调用0；200 updates/600 exposures/60 samples/240 traces/48 checks全部保留为not_run_after_stop。
首项查询的原始stdout/stderr未由既有helper持久化，具体原因未确定；不推断显存不足或GPU被占用。没有追加GPU查询、重试、换卡、续训或补采。
1,145份来源hash、授权/manifest身份、原始与归档账本及冻结ID完整性CPU核验通过。实现 `963a132`、验收 `e0999b7`未改；155项软件验收继续有效。本次无模型产物可评价，F/B0有效性仍未认证。
详见 [失败交接](B0_FEATURE_BALANCE_RUN_FAILURE_20261004.md)。下一责任方02核对失败交付；后续尝试须新目录与新单次授权。若要查明原因，应另行安排有限诊断或先改进预检原始输出保留，再按流程验证绑定；本轮未暗改已验收代码。
以下为历史记录。

---

# 01 最新状态：F 增量验收通过，申请新单次运行授权（2026-10-04）

已读取 `e0999b7`：[02 增量软件验收](../protocol/B0_FEATURE_BALANCE_ACCEPTANCE_20261004.md)接受 `963a132`，155项回归通过，无新增返工项；同一实现无需重复验收。01本轮只读核对1,145份绑定文件及manifest副本，全部一致，没有模型前向或GPU查询。
申请绑定最终 outputs v2 implementation `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`、冻结F package `a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9`、原共同初始化和历史GPU UUID `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`；GPU实时身份/可用性留待获批后预检。
唯一新目录 `/home/a200/0proj/ITM/outputs/itm_rework/method/feature_balance_authorized_20261004_v1` 已确认不存在，留待入口原子创建。具体申请：[B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261004.json](B0_FEATURE_BALANCE_AUTHORIZATION_REQUEST_20261004.json)，`pending_user_approval`、`authorized=false`；没有创建实际执行授权。
申请一次 **200 updates/600 exposures＋60条1000步完整采样＋240条零额外调用trace＋48项软件等式**；单GPU、allocated≤24GiB，训练自预检前t0起≤600秒，采样共享≤3000秒且整队≤3600秒。MDM最多60600调用/240600 examples，encoder最多1320 examples、caption最多660调用。
首个规定故障即停，保留完整分母、已完成产物及有限原始特征；不重试、续训、补采、warmup、换卡或CPU回退。历史授权不复用。按用户本轮明确的“申请新运行授权”要求，当前等待本次批准；获批后01预检并执行一次，再交02冻结评价。软件通过不代表F效果或B0有效性认证。
以下为历史记录。

---

# 01 最新状态：F 实现与 CPU 验证完成，待 02 增量验收（2026-10-03）

按 `e214014` 冻结包实现固定半权重损失、原初始化/600条训练计划、新F checkpoint准入、60完整采样/240trace/48软件等式与整队限额。保持GT裁片换侧/中段回收/末端下沉判据，不要求末尾站直。
联合回归155项通过（49项新方法＋9项F协议＋97项依赖），无失败/跳过。CPU准备绑定1145份文件、60采样输入/6训练输入；398帧统计重算差0，manifest独立重建逐字节一致。implementation SHA256 `db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30`。
本轮只有微型CPU夹具与既有来源核验，无真实预训练模型构造/前向、训练、采样或GPU查询/预检；未创建运行授权。详见 [实现交接](B0_FEATURE_BALANCE_IMPLEMENTATION_20261003.md)。下一责任方02增量验收；通过后再申请新单次1小时预算。F效果与B0有效性仍未认证。
以下为历史记录。

---

# 01 最新状态：提出单一损失配重修正，待 02 冻结（2026-10-03）

已读取 `85859c9`：V 诊断通过但 B0 有效性未认证。提出 F：仅把 normalized x0 loss 的 root＋RIC 67维与其余196维各赋予1/2权重，针对训练000506局部整体误差改善却未形成完整弓步的问题；不预设能修复验证集顺序/拍手负例。
沿用原共同初始化及600条随机训练计划，训练一个新候选200 updates，配对60条既有S1输出生成60条新完整动作。拟单GPU/24GiB，训练10分钟、采样共享50分钟、整队预检前起最多1小时，另240零额外调用trace；尚未授权。
本轮仅CPU读取既有数组：192分支/192比较重算、200更新/600曝光核对、60历史对照绑定、573份来源hash，7份提案JSON重建逐字节一致。未加载checkpoint张量、模型前向、训练、采样或GPU查询；未修改方法入口或协议。
详见 [单一修正提案](B0_FEATURE_BALANCE_PROPOSAL_20261003.md)。下一责任方02冻结设计/评价；之后01实现与CPU验证、02差异验收，再申请新授权。模型有效性仍未认证，旧负例与失败证据保留。
以下为历史记录。

---

# 01 最新状态：V 单次运行完成，交 02 独立评价（2026-10-03）

用户批准 `186d9e6` 后，按原已验收实现执行一次 V；48 状态/192 分支/144 锚点全部完成，退出码 0。预检前起 30.615823 秒，峰值 allocated 538,674,688 bytes；144 锚点全部逐位一致、最大差 0。无重试或额外训练/完整采样。
1,656 份来源、684 个 worker 产物、688 个运行文件与 288 个 NPZ 完成 CPU 核验；旧失败证据保持不变。本次成功不能确定上次预检失败原因。
详见 [运行交接](B0_STATE_DIAGNOSTIC_RUN_HANDOFF_20261003.md)。本次授权已消耗；下一责任方 02 按冻结协议评价 V/Q，模型有效性仍未认证。
以下为历史记录。

---

# 01 最新状态：失败交付已核对，申请新单次 V 授权（2026-10-03）

已读取 `485db23`：[02 失败交付核对](../protocol/B0_STATE_DIAGNOSTIC_FAILURE_ACCEPTANCE_20261003.md)通过，无新增返工项；原软件验收 `ea7f802` 继续有效。
01 再次只读核验 1,656 份绑定文件、manifest 副本与旧失败账本，均一致。新申请绑定原 implementation `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46`、原冻结包、固定 R final200 及 GPU `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
唯一新运行目录为 `/home/a200/0proj/ITM/outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1`；已确认不存在，留待入口原子创建。具体申请：[B0_STATE_DIAGNOSTIC_AUTHORIZATION_REQUEST_20261003.json](B0_STATE_DIAGNOSTIC_AUTHORIZATION_REQUEST_20261003.json)，`authorized=false`，实际执行授权尚未创建。
申请一次 **48 状态 / 192 分支 / 144 锚点、72 IMU examples / 12 caption 调用**；单 GPU、allocated ≤24 GiB、从预检前起整队 ≤300 秒。0 训练/新 checkpoint/完整采样/新噪声；首错即停，保留账本与已完成产物，不重试、warmup、换卡或回退。
旧失败具体原因仍未确定，不能保证新预检通过；本轮未查询 GPU 或调用模型。旧授权已消耗，旧目录/失败证据保留。下一步等待本次新授权，获批后由 01 预检并执行一次，再交 02 核对/评价。无需重复软件验收，模型有效性仍未认证。
以下为历史记录。

---

# 01 最新状态：V 单次授权已消耗，GPU 预检失败（2026-10-03）

用户批准 `25e99c1` 后执行一次 V；在预检前起 1.413581 秒因 `GPU identity/free-memory preflight failed` 停止，退出码 1。worker 未启动，48 组/192 分支/144 锚点均 `not_run_after_stop`，所有模型/编码调用计数为 0。没有重试、换卡或回退。
原始预检 stdout 未保存，无法区分身份/结果行数异常或可用显存低于 4096 MiB；不推断 GPU 被占用。原始失败账本保留，1,656 份绑定文件及完整分母 CPU 核验通过。
详见 [失败交接](B0_STATE_DIAGNOSTIC_RUN_FAILURE_20261003.md)。下一责任方 02 核对失败交付；本批无模型产物可评价。任何后续运行需要新目录和新单次授权，历史授权不可复用。未改变已验收实现，模型有效性仍未认证。
以下为历史记录。

---

# 01 最新状态：V已增量验收，申请新单次授权（2026-09-28）

已读取 `ea7f802`：[02增量软件验收](../protocol/B0_STATE_DIAGNOSTIC_ACCEPTANCE_20260928.md)通过，无新增返工项；相同实现无需再次冻结或重复验收。01再次核对1656份绑定文件、manifest副本和环境，均一致；未运行模型或查询GPU。
申请绑定 implementation `71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46`、V package `6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012`、固定R final200及GPU `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
唯一新目录 `/home/a200/0proj/ITM/outputs/itm_rework/method/state_diagnostic_authorized_20260928_v1` 已确认不存在。具体申请：[B0_STATE_DIAGNOSTIC_AUTHORIZATION_REQUEST_20260928.json](B0_STATE_DIAGNOSTIC_AUTHORIZATION_REQUEST_20260928.json)，`pending_user_approval`、`authorized=false`，不是执行授权。
申请一次 **48状态/192分支/144软件锚点、最多72 IMU examples/12 caption调用**；单GPU、allocated≤24GiB、从预检前起整队≤300秒。0训练/新checkpoint/完整采样/新噪声；首错即停，保留有限原始输出及全分母，不重试、warmup、换卡或回退。
下一步等待本次新授权；获批后01预检并执行一次，再交02按冻结协议评价。历史授权不复用，软件通过不代表真实GPU锚点或模型有效性通过。
以下为历史记录。

---

# 01 最新状态：V保存状态入口与CPU验证完成（2026-09-28）

按 `c0fc63f` 冻结包完成新V入口：48个保存状态、192个T/P/S/R分支、144个软件锚点；12次batch1 caption编码和72个IMU encoder examples跨t缓存。复用固定R loader，原始有限预测先落盘，调用预留/实际/完成分别记录，首错停止并保留完整分母。
增量联合回归 **135 passed / 0 failed / 0 skipped**，含29项新增V方法测试；覆盖新入口、复用依赖、缓存、锚点、300秒共享期限及故障保留。没有真实预训练模型前向、训练/采样或GPU查询/预检。
新manifest **`71cb4e2306559295e59ee5c179ce57e971a1a60dcb9f021cf1ddc33c4c45ee46`** 绑定1656份文件，保存12个cache/48套分支输入/48状态身份；CPU准备目录 `outputs/itm_rework/method/state_diagnostic_implementation_20260928_v1/`，无新checkpoint或授权。
详情：[B0_STATE_DIAGNOSTIC_IMPLEMENTATION_20260928.md](B0_STATE_DIAGNOSTIC_IMPLEMENTATION_20260928.md)。下一步由 **02进行新入口增量软件验收**，之后01再申请单GPU/24GiB/300秒的新单次授权；获批才预检执行。旧代码/协议/结果未改，模型有效性仍未认证。
以下为历史记录。

---

# 01 最新状态：接纳 guidance 混合结果，提出同状态时间顺序诊断（2026-09-28）

已读取 `9f82183`。联合控制仍未认证，不将额外文本引导认定为统一根因。CPU重算全部288个Q条件和240个trace的同状态IMU残差：000506在实际S1状态上也有相对文本的单步特征误差改善，但动作仍不完整；000382 wrists仍恶化，晚期下肢RIC问题突出。不能简单归因于“离开GT状态就失效”或直接续训。
最小提案 **V：固定R final200，48个保存的paired状态各一次batch4文本/paired/静止/倒序对照，共192个MDM分支、最多72个encoder examples；0训练、0新采样，拟单GPU/24GiB/总5分钟**。只补缺少的同状态时间顺序比较，不宣称修正有效。
7份提案JSON独立重建逐字节一致，1073份来源hash通过，600条训练呈现计数完整；本轮无模型加载/前向、训练、采样或GPU查询。
具体证据、48组状态、预算和解释边界见 [B0_STATE_DIAGNOSTIC_PROPOSAL_20260928.md](B0_STATE_DIAGNOSTIC_PROPOSAL_20260928.md)。
下一步由 **02审阅并冻结V评价**；之后01实现/CPU验证、02新增入口验收，再申请新运行授权。本提案未冻结、未授权；旧预算不复用，无现有软件返工项。
以下为历史记录。

---

# 01 最新状态：固定 R guidance 授权队列完成，交给 02 评价（2026-09-28）

用户批准 `30b3d7e` 后，按 `a90d3e7` 已验收实现完成一次 Q/S1：**480探针、60条1000步采样、240条trace全部完成**，退出码0，无失败或重试；0训练/新checkpoint。
总耗时 **2043.61秒（34.06分钟）**，allocated峰值约 **0.499GiB**。24项完整采样检查均通过，其中18项逐位相同，总体最大坐标差1.66893e−6m，未放宽冻结阈值。
CPU交付核验通过：1621份登记产物hash一致，3900个数值数组全部有限；完整索引包含1805份运行文件。运行目录为 `outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1/`。
详情、预算实耗和评价入口见 [B0_SAMPLING_ALIGNMENT_RUN_HANDOFF_20260928.md](B0_SAMPLING_ALIGNMENT_RUN_HANDOFF_20260928.md)。
下一步由 **02 按冻结协议独立评价**，无需重复相同软件验收。执行完整不代表模型有效，旧负例及准入边界保留。本次授权已消耗，01不追加作业。
以下为历史记录。

---

# 01 最新状态：固定 R guidance 已独立验收，申请新单次授权（2026-09-27）

已读取 `a90d3e7`：[02 独立软件验收](../protocol/B0_SAMPLING_ALIGNMENT_ACCEPTANCE_20260927.md)通过，无新增返工项；相同实现无需重复验收。本轮再次核对已验收 manifest 的743份绑定文件及环境，全部一致，没有运行模型或查询GPU。
新申请绑定 implementation `793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4`、冻结package `537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60`、固定R final200，以及GPU `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004`。
唯一新运行目录为 `/home/a200/0proj/ITM/outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1`，已确认不存在。
具体申请：[B0_SAMPLING_ALIGNMENT_AUTHORIZATION_REQUEST_20260927.json](B0_SAMPLING_ALIGNMENT_AUTHORIZATION_REQUEST_20260927.json)，`pending_user_approval`、`authorized=false`，不是执行授权。
申请一次480 Q（最多10分钟）＋60条1000步S1（共享最多40分钟）＋240条零额外前向trace；0训练/新checkpoint，单GPU、allocated≤24GiB、从预检前起总计≤1小时。首次超时/OOM/非有限/身份/计数/冻结软件一致性异常即停，保留完整480/60/240账本与产物；不重试、换卡或回退。
下一步等待本次新授权，获批后01预检并执行一次，02评价。历史授权不复用，原负例与模型未认证状态保留。
以下为历史记录。

---

# 01 最新状态：固定 R guidance 实现与 CPU 验证完成（2026-09-27）

按 `1680618` 冻结包实现 Q 96×batch5、S1 四分支完整采样与 240 条零额外调用 trace；固定 R final200，仅 guidance 从 2.5/1/1 改为 1/1/1，无新训练或 checkpoint。
联合 **258 passed / 0 failed / 0 skipped**，含 **33 项新增方法测试**；覆盖真实加载接口、trace 开关逐位一致、全部 24 项完整采样检查及 480/60/240 分母和故障保留。
CPU 准备 manifest `793fdd9a95a102484d2dbc7579aa398f3af0120c66422ba6ec966cc01f8cb7c4` 绑定 **743 份文件**，保存全部 60/480 输入身份，位于 `outputs/itm_rework/method/sampling_alignment_implementation_20260927_v1/`。
详情及复现入口：[B0_SAMPLING_ALIGNMENT_IMPLEMENTATION_20260927.md](B0_SAMPLING_ALIGNMENT_IMPLEMENTATION_20260927.md)。本轮无预训练模型前向、真实训练/采样、GPU 查询/预检或新运行授权。
下一步由 **02 独立软件验收**；通过后 01 再申请新单次授权（480 Q＋60 S1＋240 trace，单 GPU/24GiB/总 1 小时），随后才预检执行。原模型负例与准入边界保留，软件通过不代表模型有效。
以下为历史记录。

---

# 01 最新状态：接纳 D/L/R 负例，提出固定 R 权重的采样修正（2026-09-27）

已读取 `c6aef58`。完整性通过不代表能力通过：L步行→抬腿、R wrists拍手下肢退化、train000506/004728生成失败继续保留；不能只归因于padding或覆盖不足。
最小提案：**固定R final200，仅将guidance从2.5/1/1改为1/1/1，0新训练**；配套480个R最终模型固定探针和60条原槽位匹配完整采样，拟单GPU/24GiB/总1小时。对照与输入/噪声不挑选，新增采样trace不增加模型调用。
本轮CPU读取旧产物、核对两臂1200条训练呈现及代数/提案身份；424份来源hash一致，新增真实模型加载/前向、训练、采样和GPU查询均为0，运行代码未改。
完整证据、逐槽位计划、预算和解释边界见 [B0_SAMPLING_ALIGNMENT_PROPOSAL_20260927.md](B0_SAMPLING_ALIGNMENT_PROPOSAL_20260927.md)。
下一步由 **02冻结具体评价/容差/槽位**；之后01实现并做CPU验证、02验收，再申请新授权。本提案尚未冻结或获授权，旧额度不续用。
以下为历史记录。

---

# 01 最新状态：授权 v2 D/L/R 全部完成，交给 02 评价（2026-09-27）

用户“批准通过”后，按 `08698b4` 已验收修复执行新的单次v2队列。**480探针、两臂各200 updates/600 exposures、各60条完整采样全部完成**，退出码0；总2423.55秒（40.39分钟），allocated峰值约0.631GiB，无失败或重试。
CPU交付核验通过：1733份引用文件hash一致、12720个数值数组有限、全部槽位/训练计划/输出身份完整。24项空条件及R静止参照一致性检查均逐位相同。
运行目录 `outputs/itm_rework/method/correction_authorized_20260927_v2/`；交付模型hash、全部索引、预算实耗与复现入口见 [B0_CORRECTION_RUN_HANDOFF_20260927.md](B0_CORRECTION_RUN_HANDOFF_20260927.md)。
下一步由 **02 按原冻结包独立评价 D/L/R 结果**；无需再做相同软件验收。执行完整不代表模型有效性通过，原B0负结果与合成准入边界继续保留。本次授权已执行完毕，01不追加作业。
以下为历史记录。

---

# 01 最新状态：加载器修复已独立验收，申请新单次队列授权（2026-09-27）

已读取 `08698b4`：[02 差异验收](../protocol/CORRECTION_LOADER_ACCEPTANCE_20260927.md)通过，无新增返工项。本轮重新核对新 manifest 及其绑定的127份文件，均一致；未执行模型加载/前向、训练、采样或GPU查询/预检。
新申请已绑定 `a802ef3b03aea7bab7fff3fda5704cf56a2c5cf6c5ed36c693a892fc35d6de13`、GPU `GPU-d37bc466-7ddc-4a40-e1e6-952a58d89004` 和尚不存在的绝对运行目录 `/home/a200/0proj/ITM/outputs/itm_rework/method/correction_authorized_20260927_v2`。
具体申请：[B0_CORRECTION_AUTHORIZATION_REQUEST_20260927_V2.json](B0_CORRECTION_AUTHORIZATION_REQUEST_20260927_V2.json)，状态 `pending_user_approval`、`authorized=false`，不是执行授权。
申请一次 D/L/R：480探针/10分钟；两臂各200 updates/10分钟；各60条完整采样、120条共享90分钟；单GPU、allocated≤24GiB、总2小时从预检前计时。首次超时/OOM/非有限值/身份或计数异常停止，保留产物与完整槽位记录，不自动重试/换卡/CPU fallback。
下一步等待用户明确批准；之后01记录新的实际授权，预检并执行一次，02评价结果。旧授权已消耗，不能复用；无需重复软件验收。
以下为历史记录。

---

# 01 最新状态：授权 D/L/R 加载失败；CPU 修复完成（2026-09-27）

用户批准后按 `d47891f` 已验收实现启动一次队列。GPU 预检通过，D 在 `model.to(device).eval()` 处因上游 MDM 不返回 self 而失败；约 4.05 秒停止。探针前向、训练更新、完整样本均为 0，480/120 分母及失败记录完整保留，无重试。
拆开设备迁移和 eval 调用，新增直接测试加载入口的 CPU 回归；修复前复现同异常、修复后通过，联合 **204 passed / 0 failed / 0 skipped**。
新的 `correction_implementation_20260927_v1` 准备资产已核验，初始权重/输入/环境与已验收 v3 相同，仅绑定 worker 代码 hash 改变。
详情、失败实物、新 manifest 和复现命令见 [B0_CORRECTION_RUN_FAILURE_20260927.md](B0_CORRECTION_RUN_FAILURE_20260927.md)。
下一步由 **02 独立差异验收**，之后再申请新队列授权；旧授权已用于本次失败队列，不复用。真实 D/L/R 结果与修正效果仍未验证。
以下为历史记录。

---

# 01 最新状态：D/L/R 实现与 CPU 验证完成（2026-09-26）

按 `62eed84` 冻结包实现真实长度累积、R 静止参照差分、D hook 证据、共享随机计划和新多阶段受限运行器。
联合 **203 passed / 0 failed / 0 skipped**，含25项新增修正实验 CPU 测试；无预训练MDM前向、真实训练/采样或GPU查询。
共同初始权重和126个代码/输入资产已绑定，最终准备目录为 `outputs/itm_rework/method/correction_implementation_20260926_v3/`。
具体实现、测试、manifest/hash和复现入口见 [B0_CORRECTION_IMPLEMENTATION_20260926.md](B0_CORRECTION_IMPLEMENTATION_20260926.md)。
下一步由 **02 独立软件验收**，再对冻结的2小时整队申请新授权；本轮没有创建真实授权，不执行D/L/R。
以下为历史记录。

---

# 01 最新状态：B0 能力诊断与修正提案（2026-09-26）

已接纳 `dec5e93` 的独立评价：完整性通过，联合控制/实例信息收益尚未证实。
本批 CPU 读取旧权重、history、输入/输出，确认训练动作覆盖狭窄、静止条件仍可产生残差及混合长度训练的注意力上下文差异；因果归属仍需固定探针检验。
提出真实长度训练 L 与静止参照差分 R 两臂，各 200 updates、60 条完整采样，另有 480 次单步 branch-example 探针；新预算上限单 GPU / 24 GiB / 合计 2 小时，尚未授权。
详见 [B0_CORRECTION_PROPOSAL_20260926.md](B0_CORRECTION_PROPOSAL_20260926.md)。本轮模型前向/更新/采样/GPU 查询均为 0，方法代码与旧结果未改。
下一步由 02 冻结具体探针、120 槽位及评价，再由 01 实现并做 CPU 验证；新执行必须重新授权。
以下为历史记录。

---

# 01 最新状态：授权 B0 已完成，交给 02 评价（2026-09-26）

读取 `030c191`，用户已明确批准一次 B0；已通过真实 GPU 预检并执行验收过的运行器。
200/200 updates、36/36 条 DDPM 1000 步样本全部完成，无失败/重试；单 GPU，总约 9.35 分钟，峰值 allocated 约 0.687 GiB。
最终 checkpoint 与训练来源、全部样本 hash/有限值、组内随机计划完整性检查通过。
详见 [B0_RUN_HANDOFF_20260926.md](B0_RUN_HANDOFF_20260926.md)，其中包含运行目录、模型 hash 和全部槽位交付清单。
下一步由 02 独立评价固定 36 条结果；01 不自行追加训练或采样。本次运行完成不等于模型有效性认证。
以下为历史记录，其“未授权/未运行”状态不再代表本批。

---

# 最新方法交接：B0 运行限额（2026-09-26）

读取 `8046f5e`，新增受限运行器，复用 02 的一条训练和 36 条采样模板。
已落实 200 updates / 训练 1 小时、采样共享 3 小时、总 4 小时、单 GPU / 24 GiB allocated，以及失败停止和已完成产物保留。
23 项 CPU 限额/故障测试、166 项联合回归通过。详见 [B0_RUNTIME_LIMITS_HANDOFF_20260926.md](B0_RUNTIME_LIMITS_HANDOFF_20260926.md)。
下一步 02 核验本批执行限制；真实训练/生成、预算授权、GPU 预检均未发生。

---

# 限定合成 B0 输入交接（2026-09-25）

已接入 `c36ea56` 的 `canonical_smpl_paired_v1` 数据，见 [CANONICAL_B0_HANDOFF_20260925.md](CANONICAL_B0_HANDOFF_20260925.md)。
3 train / 3 val 准入仅适用于预先统一坐标的合成基准；新管线 `itm-method-canonical-b0-v1`。
3 个训练源的 398 帧 20 Hz acceleration 已重算并绑定来源，36 个验证条件槽位已完成输入装配。
联合 143 项通过；真实模型训练/前向/采样/GPU 均未执行。下一步由 02 重放输入，然后等待预算授权与 GPU 执行准备。

---

# 缓存与恢复训练交接（2026-09-25）

已读取 02 的 `2de27a6` / `d360e91`。新交接见 [CACHE_MIGRATION_HANDOFF_20260925.md](CACHE_MIGRATION_HANDOFF_20260925.md)。
方法入口拒绝 diagnostic NPZ 被 manifest 改标签；新 checkpoint 绑定实际输入来源与 normalization，恢复时严格核对。
5 份真实缓存的原声明/改标签声明均拒绝；固定 caption 重放与旧报告相同；联合 126 项通过。
没有训练/生成/GPU 作业。旧 checkpoint 采样不变，缺少来源记录时不能从当前训练 CLI 恢复。

---

# caption 方法交接（2026-09-25）

caption 计划接口已接入，见 [CAPTION_HANDOFF_20260925.md](CAPTION_HANDOFF_20260925.md)。
固定 10 来源各两配置：9 通过、000004 阻塞；13 条 SSM 短输入仍拒绝，68 项测试通过。
没有训练/生成/GPU 作业。公共计划与旧产物保持不变；下方仅为历史交接记录。

---

# seconds-v2 独立验收后的记录更正

已读取 02 的 `c9cdc7b` / INPUT_FOLLOWUP_ACCEPTANCE_20260924.md。
方法原检查的 train ID 更正为 **000004/000006/000008**（各 head/wrists），6 条 val 仅 head；
02 单独补查的 000004/000008/000010 不计入方法旧报告。详见 FOLLOWUP_V02.md 的显式勘误。
原 JSON/快照/checkpoint/样本保持不变；83 项联合回归属于 02 的独立验收，本轮未训练或重新生成。
caption、坐标/骨架、相容见证及合格 B0 的阻塞仍在。

---

# 第二批方法入口交接

读取 02 提交 `1bc3a41` 后完成 v0.2 输入适配，详见 [FOLLOWUP_V02.md](FOLLOWUP_V02.md)。
代码测试与输入 dry-run 已通过；尚未启动新训练/生成实验。下文保留上一批历史交接。

---

# 01 → 02 / 03

读取 commit `663414f1d9bffd6a70189846739bc9d9b1d0d873`；契约 `itm-rework-v0.1-draft` 未冻结。
改动文件/hash、命令和执行结果分别见 code_snapshot.json、commands.md、audit.md；阶段状态见 STATUS.md。
本会话只修改方法/采样/训练和测试；实际缓存桥 `src/itm/backbones/mdm.py` 明确由 01 负责。

## 给 02

- B0 正确性已具备：33 项回归、真实 Stage-2b 非零 bias 审计、完整 DDPM 随机流重放。请勿把 smoke checkpoint 当验证过的 B0。
- 已接入你的 `sampling_noise`，每步按 case/seed/backbone/shape 寻址，记录完整地址/hash。该公共文件仍未提交；请在协议提交中带入，方法不复制一份实现。
- 新采样按单条有效长度运行，保留同一条的四路 batched CFG。原始 MDM 自身 mask_frames=False，padding 改变有效帧输出。不要跨长度/随机形状比较时假设噪声仍相同。
- 默认停止全段时间拉伸；使用 crop-local 秒裁切，短 donor 报错。这没有解决 HumanML/IMU 的一帧偏移和世界坐标差，训练仍须显式 diagnostic flag。
- 逐条诊断产物交接：smoke_records.jsonl，原始 normalized HumanML263 与 joints 同源；输入 acceleration/orientation 不能当 generated IMU。记录尚未经过公共 result.schema/正式评价器验收。
- 输出只具 RIC joints；请不要宣称可恢复正确 wrist orientation。C 阻塞不变。
- B 初版只实现 arm_spread_m（水平腕间距均值）方向/有界 margin；请确定它与所需编辑属性是否一致。CLI 使用公共 compatibility 字段并拒绝无见证/非 train/非 compatible 数据。需要你交付 train-only witnesses、attribute 规则/阈值及成本预算。
- 已保留 legacy/null-only/identity-only/combined/no-time/FiLM 开关与 diffusion-only/trajectory-only，尚未做验证选择。
- checkpoint/modelcard 路径与 hash 在 STATUS.md；样本没有经过成功过滤。第一次无缓存 motion 的失败也保留。
- 没有改你的 STATUS、contract 或 evidence_index。请仅把实际完成的正确性审计与 smoke 状态纳入公共事实索引。

## 给 03

实际已实现：有序融合、完整残差 strict gate、秒时间位置、确定的冻结模式、四路 CFG 测试、逐步噪声重放、
训练期单步相对属性响应实验、简单 FiLM 对照接口。身份保持/CFG/零初始化首先是正确性与基线，不包装成新颖机制。
旧 checkpoint 的 legacy null 确实不等于原始 MDM；量级见 audit.md，仅为一个短 CPU 诊断，不可外推成全部旧实验无效。
新 strict null 可以还原原始输出；这不等于解决语义编辑/生成控制。

尚未验证：B 相对 absolute anchor 的生成优势；完整生成的左右归属；B0/FiLM 的预算公平比较；C；B+C；
相容语义成功、严格传感器一致性和最终模型联合能力。论文不得把这些写成结果。
