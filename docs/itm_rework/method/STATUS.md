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

# 01 STATUS — B0 运行限额与 CPU 故障测试（2026-09-26）

读取 `8046f5e`，输入已独立验收。本批完成受限运行器和训练/采样内部检查，未改公共数据或协议。
训练至多 200 updates / 3600 秒；36 槽位共用 10800 秒；单 GPU、总 14400 秒、24 GiB PyTorch allocated。
超时/OOM/非有限值/身份不符均停，原子保存与全槽位状态保留失败分母，部分 checkpoint 不能进入固定验证。

23 项 CPU 故障/限额测试、166 项联合回归通过；仅执行文件准备，真实 B0 训练/采样/GPU 及 GPU 预检为 0。
新交接：[B0_RUNTIME_LIMITS_HANDOFF_20260926.md](B0_RUNTIME_LIMITS_HANDOFF_20260926.md)。
下一步由 02 核验该实现；预算授权和 GPU 预检满足后再执行。旧输入验收无返工项，不重复确认。

---

# 01 STATUS — 限定合成 B0 数据接入（2026-09-25）

读取 `c36ea56`，公共契约 v0.3-draft。首批 3 train / 3 val 的限定合成准入已接入，旧的全面“对齐未认证”阻塞不套用到该 scope。

- 显式 canonical package/schema/mapping/hash 消费，完整片段、审核 caption index 和直接 20 Hz GT。
- 实际训练输入 398 帧重新计算 IMU normalization，唯一源只计一次；val 不参与统计，MDM Mean/Std 不重拟合。
- 新统计已绑定输入 provenance 预检报告；尚未生成新研究 checkpoint。
- 六种采样条件含显式文本关闭、原 backbone bypass 与首次观测姿态静止 fixture；36 槽位输入已装配。
- 最终联合 143 项通过；旧两份输入报告完整重放相等。真实模型前向/训练/采样/GPU 为 0。

新交接：[CANONICAL_B0_HANDOFF_20260925.md](CANONICAL_B0_HANDOFF_20260925.md)。
下一步 02 验收实际输入/统计；预算授权、GPU 检查与受限运行调度准备完成后再由 01 执行 B0。
相容编辑 witness 不阻塞该 B0；仍是 B 分支依赖。真实设备标定与生成 IMU 闭环未认证。
以下为历史记录。

---

# 01 STATUS — 缓存与恢复训练来源核验（2026-09-25）

读取 `2de27a6` / `d360e91`，接纳 02 caption 独立验收及前缀负结果边界。公共契约仍为 v0.2-draft。

- 方法 cache wrapper 检查内嵌 schema，拒绝诊断缓存仅通过 manifest 改标签进入标准输入。
- 新 checkpoint 记录实际输入/配置/管线/MDM/normalization 来源哈希；`--resume` 在模型初始化前核验，不再静默覆盖统计。
- 旧 checkpoint 采样兼容；缺少来源记录时拒绝训练恢复。来源一致的恢复仍是 fresh optimizer warm start。
- 5 条真实诊断声明和 5 条内存改标签声明均拒绝；caption 重放 JSON 与旧报告相等。
- 定向 29 项、联合 126 项通过；无真实训练更新/生成/GPU，无新增研究 checkpoint。

详情、命令和兼容边界见 [CACHE_MIGRATION_HANDOFF_20260925.md](CACHE_MIGRATION_HANDOFF_20260925.md)。
没有迁移新缓存或重算真实训练统计。物理/语义准入、witness、预算等 G2/G3 阻塞保持不变。
下面为历史状态记录。

---

# 01 STATUS — caption 计划接入（2026-09-25）

读取 `9439818`；公共契约 v0.2-draft 未冻结。已完成显式 caption sidecar 消费，opt-in 管线
`itm-method-caption-seconds-v3`；原 seconds-v2 路径/历史产物保留。

- 严格绑定 artifact hash、caption index、半开区间与双时间轴；IMU 使用源秒轴，encoder 使用归零模型秒轴。
- 固定 10 个来源各 head/wrists：9 个通过、000004 的 199 帧全段保持阻塞；000006 实际 100 帧；000133 保留索引 1 / 59 帧。
- 已锁定 13 条 SSM 短输入全部继续拒绝。时钟 notebook 诊断没有进入部署/训练映射。
- 未准入 caption 计划在训练 CLI 的模型/设备初始化前拒绝；非零起点不能复用未重审的 B witness。
- 68 项相关测试通过；真实输入检查无模型前向、训练更新、生成或 GPU 作业。

详情/命令：CAPTION_HANDOFF_20260925.md；逐条结果：caption_input_20260925_v1.json；
代码快照：code_snapshot_caption_v3.json。没有修改 02 文件或旧 checkpoint/结果。
G2 仍只有旧单 update smoke；物理时钟/坐标/骨架/根参考与语义认证、相容 witness、阈值及预算仍阻塞正式 B0/B/C。
下面保留之前各批状态，不能将历史未完成项/运行范围套用于本批。

---

# 01 STATUS — v0.2 输入验收与记录勘误（2026-09-24）

最新读取协议提交 `c9cdc7b`，独立验收对象为方法提交 `0574491`。本次仅更正交接记录与验收状态，未改方法代码或旧产物。改动提交可用
`git log -1 -- docs/itm_rework/method/STATUS.md` 定位。公共契约 `itm-rework-v0.2-draft`，仍未冻结。

已接纳 02 独立验收：旧方法 smoke 同源解码、strict null、3000 个噪声 hash 已核对；静止 fixture 已提供，
公共 random_plan 也已随协议提交，上一批的这两项依赖已解除。

本次完成的方法入口工作：

- 训练、归一化和采样调用公共 `load_standard_cache`，拒绝 schema/物理量不符。
- 输入管线版本 `itm-method-seconds-v2`。按显式 20Hz 秒时间轴对加速度和旋转共同采样，越界拒绝，无端点夹紧；不做统一一帧平移。
- 训练不再以 IMU 更短为由悄悄缩短 motion；支持不足时拒绝。记录原始/有效帧、裁片区间、caption 区间和未认证对齐状态。
- sampling 支持 `imu_mode=static`，调用 02 fixture 并使用当前 checkpoint 的 normalization，保持 presence=true。
- B 入口要求明确 training_admissible、无未解决 blockers；metadata 与 witness 文件的实际 timestamps 都必须匹配训练片段。
- 当前 arm_spread_m + 双编辑 active-joint-proxy 原型仅接纳 head。该约束不推导成 wrists IMU 的一般物理不相容结论。

验证：48 项相关回归通过，日志 `logs/input_followup_v2.txt`。
真实输入 dry-run：train 000004/000006/000008 × head/wrists、6 条固定 val 候选的 head 输入、2 种静止输入均通过，
`input_handoff_20260924_v2.json`。模型前向 0，训练更新 0，GPU 作业 0，未修改旧 checkpoint/产物或重新标注其契约。
代码快照 `code_snapshot_v2.json`；旧 `code_snapshot.json` 保留为 02 首次验收对应的历史快照。

02 在 `c9cdc7b` 独立重现原输入报告，并完成 83 项联合回归（是 02 的验收结果，本次未重跑）。
02 另查的 train 000004/000008/000010 与本方法原报告分开记录；原文 ID 错误已在 FOLLOWUP_V02.md 显式勘误。
000006 的 caption [0,5] 秒与输入 [0,9.75] 秒不匹配，仍不能用于正式配对语义评价。

阶段：G1 输入软件正确性获独立验收，公共物理/评价契约仍未冻结；G2 仍只有单次 update 的旧 smoke；G3 无相容编辑训练/完整采样收益；C/G4/G5 未启动。
剩余实质阻塞仍是世界坐标/骨架与 caption 对齐认证、真正 compatible witness、独立评价阈值、冻结训练/采样预算和可验证输出观测链。
不使用目标 GT 做部署 canonicalization，不把新 uncertain 候选当作训练许可。

当前详细交接见 `FOLLOWUP_V02.md`。以下为第一批状态快照，历史未提交依赖/静止 fixture 缺失已由上文更新。

---

# 01 STATUS — 2026-09-24

读取 commit：`663414f1d9bffd6a70189846739bc9d9b1d0d873`。
方法提交：本文件随方法 commit 交付，可用 `git log -1 -- docs/itm_rework/method/STATUS.md` 定位。
契约：`itm-rework-v0.1-draft`（未冻结）；未修改 02 公共契约/事实索引或 03 论文。

| 阶段 | 状态 |
|---|---|
| G0 当前代码、依赖、旧 checkpoint 核查 | 已完成所列方法侧审计；静止物理 fixture 未提供 |
| G1 正确性代码/测试 | 33 项通过；strict null 真实 checkpoint 单步/1000-step 完全还原；公共契约仍未冻结 |
| G2 B0 | 编码/null/time/冻结/dropout/CFG/随机流与 CLI 已实现；真实单次 update + 保存/重载/完整采样 smoke 成功；尚无验证过的 B0 |
| G3 B 分支 | 单步相对属性响应 + paired diffusion + 双编辑 joint-proxy 训练入口已实现，梯度/冻结/模式测试通过；真实相容编辑训练与完整采样收益未验证 |
| C 分支 | 阻塞：generated-IMU certified channels 为空，HumanML→最终输出的旋转闭环未认证；不实现伪物理投影 |
| G4/G5 | 未进入最终训练、测试或论文结论阶段 |

已交付：audit.md、interface_model_card.md、commands.md、configs.json、cpu_budget.json、
logs/regression.txt、真实 checkpoint 审计 JSON/NPZ、合成左右身份过拟合记录、smoke 模型卡和逐条交接记录。
代码快照：code_snapshot.json（交付代码 hash，非声称每条早先诊断都在同一未变更工作树执行）。

smoke checkpoint：`outputs/itm_rework/method/b0_cpu_smoke_20260924_v1/model.pt`，SHA256
`623b16e6e22c09a2e6f8aaceaa635496b8afdde932ba0b3a3326eb7fbcbb00d9`。
完整样本：同目录 `samples.npz` / `samples.json`，paired/null 共 2 条 × 12 帧，采样 26.2545s CPU。
每条完整 DDPM 1000 steps，factorized batched；两个条件随机计划 ID 相同，输出 finite。
只使用 train 来源 000004 的诊断，不作为 val 或独立性能证据。

不能写成已完成：B0 泛化收益、左右完整生成控制改善、相容编辑成功率、严格传感器一致性、B/C 优于基线、同一最终模型联合能力。

继续执行前的实质依赖（不是要求用户重复授权）：

1. 02 冻结物理秒/坐标/文本区间同步接口，并提供 source-group 隔离的 train/validation 与认证面板。
2. 02 提供 train-only compatible arm_spread_m 见证、独立评价/阈值与数值预算；当前公共 compatibility_manifest 为空。
3. 02 登记 GPU 与正式 B0/消融预算；本轮只运行文件内列出的有限 CPU 正确性 smoke，没有 GPU 作业。
4. strict B/C 需要通过输出观测闭环；在此之前只能明确命名 joint-proxy。
5. 本提交依赖 02 未提交的 `src/itm/data/random_plan.py`（hash 见快照）。没有替 02 提交、复制或重新定义它；合并时须先带入协调者对应提交。

用户已有未提交文件均保留。没有修改原始数据、旧 checkpoint、旧实验结果，没有 push 或合并主分支。
