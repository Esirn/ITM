# B0 IMU主项去除两路径：入口、验收与单次运行

科学package `a0a4b0bee0f4d61729357dc7a4592d9c24a1b98cf5d1314211bce0a34de905a1`及全部17842来源未变。[冻结设计](../protocol/b0_imu_main_knockout_20261009_v1/design.json)仅将原factorized(1,1,1)改为(1,0,1)，真实text/IMU输入与交互保留。固定T-prefix x899、F200、原caption/1000DDPM/旧逐步noise、train000506 head/wrists，两路径t899→0。

新实现仅新增文件，不覆盖旧绑定方法/失败产物：`src/itm/models/mdm_imu_main_knockout.py`及worker/parent/evaluator/两个verifier/renderer。原始分支batch4、caption1、static-delta encoder、posterior及noise过程沿用已完成队列；新的组合检查在原1e−6绝对/相对软件界核对原float32组合顺序，独立重算时另保留原float32运算舍入界。没有将代数简化式替换真实组合顺序。

[implementation](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/implementation.json) SHA `7f6eb510b1c3fad38fa64af2a94b00dbf5db5566d8772b1c42acaa9a2903855d`，17860绑定依赖。CPU准备构造真实模型/前向/GPU查询/噪声/解码0；加载绑定checkpoint供CPU身份核对，不训练。

[新增CPU验收](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/CPU_acceptance.json)：7 tests通过，52.21秒。独立1000步合成text-prefix→(1,0,1)采样与新1800步worker最终features bitwise；非零text×IMU交互fixture确保新结果不是错误退化成text-only。完整1800calls/7200examples/1812checks/3626产物/6traces/2finals及预算通过；首锚点故障第1调用即停止且raw保存、journal reservation/fsync/tail、batch/deadline、父preflight时限不启动worker、失败完整分母和无模型/GPU prepare、cuda:1 UUID绑定通过。吞吐fixture只在测试内暂抑制fsync，独立故障/journal tests与真实worker实际fsync不变。原9项资源政策验收实现未改，沿用旧记录；不再要求利用率≤50%。

[科学1812 check IDs及最终来源证明](../../../outputs/itm_rework/method/imu_main_knockout_implementation_20261009_v1/implementation_delivery.json)通过。1812为2原起点＋10首步paired x_t/f00/f10/f01/f11锚点＋1800(1,0,1)组合；原先历史内shared-start自比较在新路径已改为与真实绑定source start比较。raw先保存后核验，完整失败分母/日志计数保留。

[唯一执行记录](B0_IMU_MAIN_KNOCKOUT_EXECUTION_20261009.json)按用户长期GPU自主授权及cuda:1偏好建立，不要求再次手动授权。新run `outputs/itm_rework/method/imu_main_knockout_authorized_20261009_v1`；1200秒whole-attempt绝对期限、24GiB allocated cap（不是预留）、按GPU UUID项目锁/native allocator、首故障停止，不重试或恢复。2路径/1800MDM/E4/caption2、旧noise1798/decode2、无新seed/initial noise/训练。其他任务与全局GPU设置未干预。

运行和完整评价结果见[阶段报告](../protocol/B0_IMU_MAIN_KNOCKOUT_EVALUATION_20261009.md)。运行成功及软件检查不代表动作修复，原F与论文结论保持；完整行为粗审必须覆盖全部136帧，沿用原弓步初始恢复/换侧/中段回收/末段下沉规则，不要求片外起始步或最终站直。
