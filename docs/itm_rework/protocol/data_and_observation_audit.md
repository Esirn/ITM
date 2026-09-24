# 数据与观测审计

契约：`../contract.yaml`，版本 `itm-rework-v0.1-draft`。本批未改模型、训练或采样文件。

## 物理定义与缓存

现有合成使用 SMPL 顶点世界坐标二阶差分乘 FPS²，然后五帧均值平滑；姿态为身体段 local-to-world 的全局旋转。
不加重力，也不是机体系比力。源坐标通过 `[x,y,z] → [x,z,-y]` 转为右手 Y-up；未做每段初始朝向 canonicalization。
30 FPS 下五帧窗口的样本数/FPS 为 1/6 秒；模型通常在 20 FPS 使用线性加速度插值与姿态 SLERP。
当前没有经过认证的防混叠处理。

全量 8776 个缓存通过旧审计器的完整数组检查，报告 `full_cache_audit.json`。
源文件在新挂载根全部可访问；sidecar 仅重定向旧 source_path。
三个源片段（每 split 首条）的独立 CPU 重合成与缓存比较：位置/旋转最大差约 10⁻⁷，translation/shape 相同；
加速度最大差为 0.000171661 / 0.000171661 / 0.000128746 m/s²。
预先采用的 `atol=rtol=1e-4` allclose 对加速度三条均未通过，保留这一事实；可能与数值误差经二阶差分放大有关，未证明其唯一原因。
没有据此改缓存或放宽门槛。三条核验也不是全量合成认证，更不是 HumanML 解码闭环。

安装位置已查实际模型权重：左/右腕顶点 1961/5424 对段 18/19 权重约 0.837；对 20/21 约 0.152/0.155。
因此腕部 orientation 采用前臂段 18/19 有安装依据；评价 joint-proxy 20/21 是另一个位置定义，不能只因编号不同判 bug。
模型文件 hash、各槽前三个 skinning 权重见 `source_audit_20260924_v1/cache_synthesis.json`。

## 输入管线问题

| 项目 | 状态 | 证据/影响 |
| --- | --- | --- |
| legacy 方向向量与 standard 12D 隔离 | 新读取接口通过；旧方法入口未接入 | `load_standard_cache` 要求 schema；拒绝物理元数据矛盾，不依靠 12D 猜测 |
| ID、FPS、形状、有限值、SO(3) | 全部缓存通过 | 不等于物理来源或输入/GT 同步已认证 |
| train-only normalization | 部分核实 | checkpoint 指向 train manifest，使用 resume 的统计；已 hash 统计本身，未全量重新拟合，也未认证 HumanML Mean/Std 来源 |
| caption 裁剪 | 尚未通过端到端核验 | dataset 保留 caption start/end，但 `_PairedDataset` 未据此同步裁剪；panel 仅选 full-clip caption |
| HumanML 与 source-world 对齐 | 未认证 | 训练动作来自 HumanML263，IMU 保留源朝向；不能暗中读 GT pelvis 来完成部署输入 canonicalization |
| 物理时间保持 | 旧 sampler 失败 | `_fit_length_linear/orientation` 用 linspace(0,1) 拉伸全段；1880 个已核验输出中 1792 个在加载阶段触发长度拟合，包含后续可能关闭 IMU 的 Text-only 条件 |
| 采样 GT 表示 | 已定位差异 | 旧 sampler 使用缓存 joints 重采样为 GT；如 002149 为 160 帧，HumanML263 为 159 帧。不能拿一个长度冒充另一个 |
| mask 隔离 | 新算子已测；模型端待 01 | inactive/丢包用显式有效掩码；不把 NaN 当静止 |
| 镜像 | 部分测试，实际数据链未测 | 反射 det=-1 被判无效姿态；左右交换/安装坐标/normalization 仍需集成检查 |

## 新观察接口

`src/itm/data/observation.py` 提供：

- `observe_mesh(vertices, global_rotations, timestamps, sensor_mask)`：读取实际渲染的同一 mesh/全局旋转。
- `observe_sensor_positions(...)`：显式世界系米、均匀秒时间轴；差分与平滑完整 stencil 才有效。
- `Observation` 分别保存 acceleration/orientation 的有效掩码和 operator metadata；边界为 NaN + invalid。
- `device_acceleration_to_world(...)`：body-specific-force 必须给定 gravity_world；body-linear-acceleration 禁止再次加重力。

新算子版本为 `smpl-vertex-observation-v1`。旧端点零值/edge padding 语义不被无声替换；新旧内区定义一致，端点有效性不同。
这个接口不生成 SMPL，不拟合安装角，不使用独立预测头，也没有自动获得 HumanML 输出的传感器认证。

已测：静止/匀速/恒加速度；偏心点单轴旋转的切向/向心项；全局刚体变换；平滑差分边界；丢包 stencil；
反射与非法时间轴；重力桥接；时间裁剪内区一致性；直接低频抽样导致的 aliasing 示例。
未测：设备实录校准/时钟漂移、完整 mesh 镜像、模型 masked-input 不变性、HumanML 旋转解码认证、JVP/VJP。
后两项是 01 新表示或 C 分支的前置依赖，本批不假定已通过。

## 输出闭环裁定

从历史 5-case 每样本 fitting 结果重算分布，见 `statistics_20260924_v3/roundtrip_distributions.json`。
没有重新 fitting；没有改变历史 0.5 m/s²、0.2 rad 门槛，也没有把它们用作新协议的应用容差。

| 配置 | acceleration mean / median / p95 (m/s²) | orientation mean / median / p95 (rad) | 加速度/姿态超限率 |
| --- | --- | --- | --- |
| head | 0.8851 / 0.5040 / 1.8423 | 0.3201 / 0.3170 / 0.4030 | 60% / 100% |
| wrists | 2.5635 / 2.7246 / 4.4268 | 1.0325 / 1.2298 / 1.2633 | 100% / 100% |

fitting 运行失败率为 0，物理一致性门槛仍失败。这两者必须分开。
本阶段生成输出认证的 IMU 通道为空；只能报告 joint-proxy 诊断和将来冻结阈值后的 `JSR_joint_proxy`。
不允许以此开启“严格传感器约束”的方法 C 分支。
