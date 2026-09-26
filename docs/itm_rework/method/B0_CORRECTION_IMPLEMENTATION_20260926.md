# 01 → 02：冻结 D/L/R 实现与 CPU 验证

读取 **`62eed84`**，按冻结包 `canonical-B0-correction-package-v1` 实现；package SHA256 保持 `e24c8cddce0d0cb11a7b71d2cd91a24c05fab2bbceafd2f83d56f25063c48d8a`。**联合 203 项通过，新增修正实验测试 25 项，无 failure/error/skip。** 本轮未运行预训练 MDM 前向、真实训练、真实扩散采样或 GPU 查询；未创建真实预算授权。

## 实现入口

- `src/itm/models/mdm_correction.py`：显式核验冻结包、原准入数据与归一化，逐条 PCG64 训练/探针噪声重建，原始 O[0] 静止参照，30→20 Hz 之后的 tail-reversed，独立 RNG 与共享 dropout，真实长度 loss/3 累积，条件缓存身份和共同初始化。
- `src/itm/models/mdm_imu_control.py`：新增显式 `reference_mode=static_delta`；默认 `none` 保留历史路径。R 只计算 `linear(delta,W,bias=None)`，bias 冻结；缺少明确差分 cache 标识时拒绝执行，避免将旧 E(u) 缓存误当 R。新增无参数 Identity hook 点，不改变已有 state_dict 张量名/形状。
- `src/itm/models/mdm_correction_limits.py`：新阶段 receipt/counters；复用既有 CUDA allocator、有限值检查和原子发布。**新训练累积路径不调用旧 `optimizer_step` 中的梯度裁剪**，符合本包冻结的无 clipping 要求。
- `scripts/run_mdm_correction_worker.py`：D/单臂训练/单槽位采样入口，只接受新版本受限 context。训练/采样使用原骨干和 DDPM 实现，不另外热身或插入模型调用。
- `scripts/run_mdm_correction_bounded.py`：准备与执行分离；一次唯一目录、共同初始 state、全 D/L/R 队列、新授权绑定。旧单臂 36-slot runner 未改，也不能用来执行新包。

方法文件 SHA256 在 [cpu_verification.json](correction_implementation_20260926_v1/cpu_verification.json)。本轮未改 02 包、评价代码、旧数据或结果。

## 对冻结设计的具体落实

**训练与 R**：逐条读取冻结的 600 行 exposure，按显式 update/microbatch 顺序，不再用 DataLoader shuffle 或重新抽 timestep/presence。训练 noise 直接按完整地址 SHA256 初始化 PCG64，并核对 frozen noise hash。两臂读取同一初始文件，load_state_dict 分别复制权重；共同 encoder 与 adapter W/b 的逐张量 hash 记录在初始化、implementation 和各 checkpoint 中。

每条以本身 T 执行，loss 为 263×T 元素均值；三个 loss 各除以 3 累积后，只 step 一次。null 项保留在分母、呈现与前向计数内；无可训练图时可跳过 backward。R 两次 encoder 各在同一登记 seed 的 fork_rng 中执行，退出恢复调用者 CPU/CUDA RNG；不拼 batch、不 detach reference。L encoder cap=600，R=1200；两臂 MDM cap 各600、update cap 各200。只发布 update 50/100/150 和 final200，部分模型不进入采样。

每次完成 update 的 JSON 包含三条 exposure 的原计划、实际 noise hash、shape、loss/graph 状态及已完成 step；receipt 持续保存预留/完成计数，失败时不补做 update 或抢救权重。checkpoint 记录实际 optimizer defaults、环境、原训练来源、新包/共同初始身份和 final 标志。

**D**：384 次单条前向，加按 config/t/seed/condition 组装的 32 次三条 padded 前向，共480 branch examples。有效噪声复用 exact 的已登记数组，尾部只用 `-padding` 地址；保留骨干 `mask_frames=false`。hooks 读取 fused、timed、两层 temporal、final E，以及8层 pre-add hidden/gated residual；同一次前向中取得，截掉前置 token 和 padding 后保存 float32 NPZ。JSON 保存 x_t/target hash、有效元素 count/finite count/sums_sq/RMS/max、zero-denominator=null 的比例、x0 MSE 与 diffusion 配置。末尾从已保存数组计算 paired/static/reversed/text-only 及 padded/exact 差分，不再调用模型。02 可直接重算，不依赖方法侧汇总。

**采样**：逐槽位消费冻结的120条计划，原 val 噪声地址保留，train 单列地址；每臂只绑定自己的 final200 hash。R 先分别编码 observed/reference 并缓存差分；后续1000步仍为原四分支 batched CFG 2.5/1/1。缓存的输入、参照、mask、time、presence、checkpoint 与 mode 变化会被拒绝。保存原要求的 normalized263/joints/GT/raw与normalized输入/reference/O[0]来源、时间、mask、条件、随机计划及所有身份信息。

父队列用已生成文件执行 null/bypass 与 R-static/text-only 的最大坐标差核验（≤1e−5 m），保存 bitwise 和连续差值。超过冻结容差即停止，已保存的该槽位产物保留但状态记 failed；不追加模型前向作核验，也不把静止恢复当效果收益。

**运行限额**：总7200秒从 execute 队列入口、GPU预检前计时；D≤600秒，L/R各≤600秒且共享1200秒，120条采样共用5400秒，所有期限同时受总期限约束。子进程超时由既有 supervisor 杀整个进程组；不重置R采样期限。单 UUID、native allocator、≤24 GiB allocated；所有真实模型加载前先设置设备限制。预检前就保存480 probe与120 sample分母；失败停止整队，未运行槽位保留。跨槽位期限耗尽时记录 `launched=false`，不套用上一个子进程的启动状态。GPU预检失败、有限但效果差的D、超时/OOM/非有限值分别沿冻结规则处理：效果差不会临时改 L/R。

## 已准备的可核验资产

最终供验收的准备目录：

`outputs/itm_rework/method/correction_implementation_20260926_v3/`

- `implementation.json` SHA256：`449b8d278a5a4d1ee31c6dad8a9eedbe59c2990f1efceedc45ad4ae70c805676`。
- `initial.pt` SHA256：`3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401`。
- 绑定126个代码/包/数据/骨干文件；准备60条独特条件输入，两臂复用这些定义。最终核对这些文件及初始化 hash 全部相符。
- environment：Python3.10.20、torch2.5.1、numpy2.2.5、CUDA build12.1、cudnn90100、float32、2 CPU threads；TF32/determinism 等实际标志也已绑定。查询编译/库版本不等于初始化或预检GPU。
- 文档内 [implementation.json](correction_implementation_20260926_v1/implementation.json) 是原文件的逐字节副本；初始权重实体在 outputs，不写入 git。仅构造随机CPU参数并保存，未加载预训练骨干或执行模型。

早期准备目录 v1/v2 保留，不能用于本批最终授权；它在实现完善前生成。两次独立 CPU 准备的初始逐张量 hash 相等；文件整体 hash 可以因环境字段变化而不同。以上 v3 才绑定最终代码/环境。准备目录不是实际队列目录，执行时仍需全新 run-directory。

## CPU 验证证据

[最终日志](correction_implementation_20260926_v1/logs/regression_final.txt) / [JUnit](correction_implementation_20260926_v1/logs/regression_final.xml)：**203 passed，34 warnings，0 skipped**。warnings 为既有 Transformer nested-tensor 配置提示。首轮200项日志保留；补充worker级验证及跨槽位未启动测试后得到203项；随后只明确输出自身的completed/generated状态，并对两条实际worker输出路径做定向复核，日志为 `logs/output_status.*`。

新增25项覆盖：

- CPU标量 SGD 真正完成200次 step、600次呈现计数，第201次拒绝；loss/3 梯度与三个独立目标之和吻合，null分母保留，无梯度裁剪。
- R相同输入/参照、共享dropout下差分严格零；CPU encoder/adapter梯度max_abs满足≤1e−7；扰动参照后有梯度，caller RNG完整恢复。非零bias被显式忽略且冻结，null与raw骨干一致。
- 全600条真实计划训练noise重建、96组探针有效/尾部噪声复用、32个三条padded组；真实已准入输入的 numeric-zero/reference/tail-reversed语义与原O[0]核对。
- 实际 worker 训练循环在微型合成CPU模块上跑完两臂，各200 updates、600 exposures、4个checkpoint；L/R共享初始tensor、逐条noise hash及地址，encoder计数分别600/1200。这些是软件fixture，**不是新 B0 训练结果**。
- 微型CPU模型执行实际D worker的exact/padded分支与hooks，重算保存数组及共同x_t；实际sample worker搭配假diffusion运行1000步，检查计数、随机计划和R-static/text-only完整输出一致。无预训练MDM调用。
- 顶层完整队列用明确标记的CPU stub验证全成功、预检失败、D超时、R部分199 updates、第二槽位OOM/超时、R-static一致性失败，以及前一槽位结束后共享期限耗尽。完整分母和已完成文件保留，下一作业不启动。超时杀真实派生进程组的行为由同轮既有运行限额测试覆盖。
- 缓存身份改变、新授权绑定改变被拒绝；共同CPU初始化无需forward且可复现。旧null/bypass、旧checkpoint配置和既有方法联合回归通过。

开发时补充D worker fixture曾漏列padded text-only对照，补齐后通过；没有为此放松生产路径对冻结对照完整性的要求。

## 给 02 的验收入口与下一步

```bash
# 仅CPU软件测试（小模型/假diffusion）；不调用预训练模型或GPU。
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python -m pytest -q tests/test_mdm_correction.py

# 仅构造CPU初始参数、读取/核验输入与代码，目录须全新。
OMP_NUM_THREADS=2 /home/a200/miniconda3/envs/itm/bin/python scripts/run_mdm_correction_bounded.py \
  --prepare-dir <新的准备目录>
```

完整联合命令保存在 `correction_implementation_20260926_v1/commands.md`。请 02 独立核验上述具体代码和 v3 准备资产，尤其核对D hook输出、R共享dropout/缓存、真实长度累积及跨阶段失败保留；无需重新冻结或重复旧36条结果评价。

**本轮不申请或创建执行授权。** 独立软件验收通过后，再申请已冻结整队：480探针＋两臂各200 updates/60完整样本、单GPU/24GiB/最多2小时。新授权记录需绑定新package SHA、implementation SHA、唯一绝对run_directory、GPU_UUID及实际用户指令。运行器不会将冻结状态或旧B0授权解释为批准。

未来执行命令必须带 `--execute --implementation <上列v3原manifest> --run-dir <全新目录> --authorization-record <实际新授权> --gpu-uuid <批准设备>`。该入口的CPU控制流程已验证；**真实GPU性能、真实D/L/R输出与修正效果尚未验证**，没有新的能力认证或研究结论。
