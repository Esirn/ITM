# 02 → 01：D/L/R 失败保留与加载器修复差异验收

读取 **`03e6d95`**。**本次修复通过独立差异验收，无新增返工项。** 01 可针对新 manifest 申请另一次原上限的队列授权；旧授权已消耗，不能直接重启。冻结设计、上一批输入/训练/限额的软件验收及原 B0 负结果继续保留，不重新冻结或重复验收未变更部分。

本轮只读取失败记录、核对绑定文件，并运行微型 CPU 加载接口测试；无预训练 MDM 加载/前向、真实训练、采样、GPU 查询或重试，也未创建新授权。提交可由 `git log -1 -- docs/itm_rework/protocol/CORRECTION_LOADER_ACCEPTANCE_20260927.md` 定位。

## 失败记录核对

独立读取原运行目录 `outputs/itm_rework/method/correction_authorized_20260927_v1/` 的 status、D context、receipt 和 log，与01交付副本逐字节和hash核对：**4/4 相同**。

- 队列及D阶段均 failed；唯一D子进程记录 `launched=true, returncode=1`，elapsed **4.047569672秒**。
- receipt 的 MDM/encoder examples、updates attempted/completed、denoising/branch calls、exposures 全部为0；completed_slots及新产物列表为空，原目录没有新 `.pt` / `.npz`。
- **480 probe全部not_run_after_stop，120 sample全部not_run_after_stop**；后续四阶段均未运行，完整分母保留。该目录只有本次D子进程的四份runtime文件，没有训练/采样子任务记录。
- 授权、context、status绑定的是旧implementation `449b8d27…`、原唯一目录及同一GPU UUID；历史GPU预检和设备迁移已发生，不能将本次失败说成“从未使用GPU”。峰值allocated记录 **500039680 bytes（约0.466GiB）**；本轮未重新观测GPU。
- traceback明确停在旧 `model.to(device).eval()`，异常为 `AttributeError: 'NoneType' object has no attribute 'eval'`。这不是D数值结果或模型能力负例，不把未运行槽位改记完成。

机器证据：[failure_and_binding.json](correction_loader_acceptance_20260927_v1/failure_and_binding.json)、[audit.log](correction_loader_acceptance_20260927_v1/audit.log)。旧worker文件的hash还通过Git中 `8e4df1e` 的原文件重算核对，未要求当前已修复文件匹配旧hash。

## 修复范围与针对性复现

上游已绑定 `model/mdm.py` 的 `_apply()`、`train()` 调用父实现并处理SMPL，但没有 `return self`。因此 `.to()` / `.eval()` 均不支持链式返回；本次拆成 `model.to(device)`、`model.eval()`、`original.eval()`，仍作用于同一模型对象。未改上游、权重定义、条件、随机计划、数据、评价或预算。

前一批worker fixture整体替换了 `load_model`，确实遗漏了上游加载接口语义；此前CPU通过不能保证真实加载成功。本轮新增测试直接进入实际loader，用原样提取的上游方法接入微型模型，替代重量级工厂/兼容导入，禁止所有 `nn.Module` 前向。

02将这个实际loader fixture扩展到旧函数负例和两种修复模式，**4 passed / 0 failed，9条既有Transformer配置warnings，0.95秒**：

1. 从Git读取旧 `load_model` 函数，仅在内存中替换，复现完全相同的NoneType/eval异常；预期异常被测试捕获，不改01文件。
2. 修复后的 `reference_mode=none` 通过。
3. 修复后的 `reference_mode=static_delta` 通过，并检查bias仍冻结。
4. 01新增的原加载回归测试独立重跑通过。

核对加载前后的模型/骨干对象身份、实际state加载、控制器权重、CPU设备、全部模块及SMPL替身的eval状态。未调用真实CLIP/SMPL/MDM模型前向，未用新预算做GPU加载尝试。证据：[targeted.log](correction_loader_acceptance_20260927_v1/targeted.log)、[JUnit](correction_loader_acceptance_20260927_v1/targeted.xml)、[02扩展测试](/home/a200/0proj/ITM/tests/test_correction_loader_acceptance.py)。

01提供的 **204项联合回归**日志/JUnit及相关文件hash已核对，0 failure/error/skip。**204是01执行、02核对的证据；本轮02独立执行的是上述4项差异测试**，不将两者相加，也不重复约100秒的未变更联合回归。

## 新准备身份

接受的manifest为 `outputs/itm_rework/method/correction_implementation_20260927_v1/implementation.json`，与01文档副本逐字节一致：

| 项目 | 接受身份 |
| --- | --- |
| 新implementation SHA256 | `a802ef3b03aea7bab7fff3fda5704cf56a2c5cf6c5ed36c693a892fc35d6de13` |
| worker SHA256 | `d65e327219ad07e0c83dfe51d764418ce4331dfe629c6a9f937c1b72b5f0a5e1` |
| 共同initial.pt SHA256 | `3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401` |
| 原冻结package SHA256 | `e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a` |

126份绑定文件路径集合相同，**仅worker hash改变**；全部126份现有文件及initial共127份独立hash核对通过。新旧initial文件hash及记录的逐张量hash相同；60份输入metadata、完整环境及其他manifest字段一致（初始化文件路径引用除外）。因此沿用 `d47891f` 对120槽位、398帧统计、共同初始化、R差分和队列限制的验收；本轮不重新准备权重或重算未变更输入。

## 下一步和限制

**下一步由01申请新的单次队列授权**，绑定上述新implementation、冻结package、一个尚不存在的绝对运行目录、具体GPU UUID及实际用户指令。保持已绑定环境（含OMP_NUM_THREADS=2）。原 `B0_CORRECTION_AUTHORIZATION_20260927.json` 已用于失败尝试，不能复用；本报告也不是授权。

预算不变：D≤480 branch-example探针/600秒；L/R各200 updates、600 exposures、最多600秒，训练共享1200秒；各60条完整采样，共120条共享5400秒；单GPU、allocated≤24GiB、全队列最多 **7200秒**。取得新授权后预检并执行一次，首次超时/OOM/非有限值/身份或计数异常停止，不自动重试。

此修复只证明已观察到的加载接口问题在CPU兼容fixture中解决，不承诺真实GPU全队列必定成功。真实D/L/R结果和修正效果仍未产生；02在实际交付后按原冻结协议独立评价。01无需为相同修复再次等待设计确认。
