# Nymeria 时间对齐宿主机交接文档（2026-07-16）

## 1. 任务目标

请在数据所在的宿主机本地文件系统上验证 Nymeria（不是 NymeriaPlus）的时间对齐协议，重点回答：

1. `narration/*.csv` 的 `start_time/end_time` 是否统一使用头戴 Aria 的 `DEVICE_TIME`（秒）。
2. 是否应通过 Head VRS 的官方接口将该时间转换为公共 `TIME_CODE`，再与 Xsens、Head/Wrist VRS 对齐。
3. 当前通过 SSHFS 读取 `data.vrs` 时出现的错误，是源文件损坏还是 SSHFS 随机读取问题。
4. 哪些 Nymeria 序列满足协议，哪些需要排除或单独修正。

正式对比实验计划使用 Nymeria：

```text
/home/a200/mount/a40/datasets-HDD/Nymeria/
```

宿主机上的真实路径可能不同，请替换为未经过 SSHFS 的本地路径。

## 2. 相关文件及时间域

### 2.1 Xsens 动作与 IMU

```text
body/xdata.npz
```

主要字段：

```text
timestamps_us                Xsens 帧时间，微秒
segment_tXYZ                 23 个身体 segment 的位置
segment_qWXYZ                23 个身体 segment 的方向
sensor_freeAcceleration      17 个 Xsens IMU 的自由加速度
sensor_qWXYZ                 17 个 Xsens IMU 的融合方向
frameRate                    通常为 240 Hz
```

已有证据表明：

```python
xsens_timecode_ns = xdata["timestamps_us"] * 1000
```

可作为 VRS `TIME_CODE` 使用。

### 2.2 Aria VRS

```text
recording_head/data/motion.vrs
recording_head/data/data.vrs
```

- `data.vrs` 包含图像和 IMU 等完整流，通常十几 GB。
- `motion.vrs` 是较小的运动传感器版本。
- 两者中的同名 IMU stream 在已验证样例中时间戳和数值完全一致。
- VRS 内同时有设备自己的 `DEVICE_TIME` 和跨设备同步的 `TIME_CODE`。

### 2.3 Narration

```text
narration/motion_narration.csv
narration/atomic_action.csv
narration/activity_summarization.csv
```

共同字段：

```text
start_time
end_time
```

公开 Nymeria/NymeriaPlus 仓库没有正式实现 narration loader：最新 `NarrationLoader` 标记为 `Not implemented yet`，legacy `NarrationProvider` 也是空实现。因此需要用数据本身和 Project Aria API验证时间域。

## 3. 当前最强假设

正常序列的 CSV 时间应解释为：

```text
CSV start_time/end_time（Head Aria DEVICE_TIME，秒）
    ↓ Head VRS 官方时钟转换
公共 TIME_CODE（纳秒）
    ↓
Xsens / Head VRS / Wrist VRS / SMPL
```

正确转换形式：

```python
csv_device_ns = round(csv_time_seconds * 1e9)
timecode_ns = provider.convert_from_device_time_to_timecode_ns(csv_device_ns)
```

随后使用 `timecode_ns` 查询：

```python
xsens_time_ns = xdata["timestamps_us"] * 1000
```

手腕 VRS 有自己的 `DEVICE_TIME`，不能直接用 Head `DEVICE_TIME` 查询；应统一转换到 `TIME_CODE`。

## 4. 已有实测证据

### 4.1 NymeriaPlus 两条重叠序列

Nymeria 与 NymeriaPlus 中以下文件 SHA256 完全相同：

```text
20230921_s1_alec_meza_act0_8ytqbv/body/xdata.npz
20230921_s1_alec_meza_act1_ag2eub/body/xdata.npz
两条序列的全部 narration/*.csv
```

因此这些原始 Xsens/CSV 文件在两个版本间没有变化，NymeriaPlus 主要增加了 SMPL/MHR 等处理结果。

直接使用 NymeriaPlus Head `motion.vrs` 转换得到：

| Sequence | CSV DEVICE_TIME start | 转换后 TIME_CODE | Xsens TIME_CODE start | 差值 |
| --- | ---: | ---: | ---: | ---: |
| act0 `8ytqbv` | 267.561519 s | 1134.607419 s | 1134.397800 s | 0.209619 s |
| act1 `ag2eub` | 1448.349961 s | 2508.717838 s | 2508.505700 s | 0.212138 s |

约 0.21 秒差值在两个动作中一致，符合人工标注从动作开始稍晚处起标的特征。

### 4.2 Nymeria 旧版直接 VRS 验证

在另一个 SSHFS 路径下，以下两条 Nymeria `motion.vrs` 可正常读取：

| Sequence | CSV start - Xsens start 转成的 Head DEVICE_TIME |
| --- | ---: |
| `20230927_s0_zachary_price_act4_casc9i` | 0.224625 s |
| `20230927_s1_samantha_may_act0_kb2wve` | 0.213315 s |

这同样支持 CSV 使用 Head `DEVICE_TIME`。

### 4.3 MP4 内嵌时间戳审计

Nymeria 的 `video_main_rgb.mp4` metadata `description` 内嵌每一帧 Head `DEVICE_TIME`（纳秒）。例如 Alec act0：

```text
video frame DEVICE_TIME: 100.454923 ... 1236.539780 s
CSV range:              267.561519 ... 1227.507899 s
```

对 HDD 路径中可检查的序列扫描结果：

```text
32 条：全部 CSV 时间位于 MP4 的 Head DEVICE_TIME 范围内
2 条：Grace act0/act2 位于范围外
1 条：Grace act3 没有 narration rows
1 条：Samantha 的 HDD 副本缺 MP4；已在另一副本用 VRS 验证正常
```

这说明 Head `DEVICE_TIME` 是 Nymeria 的主协议，但存在异常数据。

## 5. 已知异常

重点调查：

```text
20230928_s0_grace_randolph_act0_rsm00j
20230928_s0_grace_randolph_act2_0xxd51
20230928_s0_grace_randolph_act3_090k3i
```

Grace act0 在当前可读 VRS 副本中的结果：

```text
Xsens start TIME_CODE -> Head DEVICE_TIME: 31505.642887 s
CSV global start:                       29816.520682 s
差值:                                  -1689.122205 s
```

该序列的 CSV 开始/结束都近似存在固定的大偏移。需要判断：

- CSV 是否错误关联到另一段 recording。
- annotation 导出时是否使用了另一设备/视频的时钟。
- 是否存在缺失的 session offset。
- 本地文件是否来自不匹配的数据版本。

在明确原因前，建议将 Grace 相关序列标为 invalid，不使用“最早 CSV 对齐动作起点”的启发式修复作为论文协议。

## 6. SSHFS/VRS 读取问题

在 SSHFS 路径读取第一条 HDD Nymeria `data.vrs` 时出现大量：

```text
Record size too small. Expected: 32 Actual: 0
TimeSyncMapper: Fail to read record ... streamId 285-2
```

当前不能据此断定源文件损坏，可能原因包括：

- SSHFS 对大型文件随机读取异常。
- 网络挂载短读或缓存问题。
- 源文件本身不完整。
- `projectaria-tools` 版本与文件存在兼容问题（可能性相对较低，因为其他同版本 VRS 可读）。

宿主机必须直接读取源文件复测。建议同时记录：

```bash
stat <data.vrs>
sha256sum <data.vrs>
```

若宿主机可读而 SSHFS 不可读，应保留源文件，不要重新下载或删除。

## 7. 已提供脚本

项目中已有：

```text
scripts/audit_nymeria_time_alignment.py
scripts/compare_nymeria_head_imus.py
scripts/inspect_nymeria_xdata.py
```

依赖：

```bash
pip install projectaria-tools==2.1.4
```

项目 `environment.yml` 已新增：

```text
projectaria-tools>=2.1.4
```

## 8. 宿主机执行步骤

### 8.1 先验证单条 VRS

使用宿主机本地路径：

```bash
python - <<'PY'
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain

path = "/LOCAL/Nymeria/20230921_s1_alec_meza_act0_8ytqbv/recording_head/data/data.vrs"
provider = data_provider.create_vrs_data_provider(path)
print(provider.get_first_time_ns_all_streams(TimeDomain.DEVICE_TIME) / 1e9)
print(provider.get_last_time_ns_all_streams(TimeDomain.DEVICE_TIME) / 1e9)
print(provider.get_first_time_ns_all_streams(TimeDomain.TIME_CODE) / 1e9)
print(provider.get_last_time_ns_all_streams(TimeDomain.TIME_CODE) / 1e9)
PY
```

确认无 `Record size too small` 后再跑全量。

### 8.2 全量正式审计

```bash
python scripts/audit_nymeria_time_alignment.py \
  /LOCAL/Nymeria \
  --clock-source vrs
```

脚本优先读取：

```text
recording_head/data/motion.vrs
```

缺失时退回：

```text
recording_head/data/data.vrs
```

默认判定：

```text
abs(CSV global start - Xsens start 转换后的 Head DEVICE_TIME) <= 1.0 s
```

记为 `direct`，否则为 `offset`。

### 8.3 不读取 VRS 的辅助审计

```bash
python scripts/audit_nymeria_time_alignment.py \
  /LOCAL/Nymeria \
  --clock-source video
```

该模式只判断 CSV 是否落在 MP4 内嵌的 Head `DEVICE_TIME` 范围，不能完成 `DEVICE_TIME -> TIME_CODE` 转换。

## 9. 需要宿主机 Codex 输出的结果

请生成一份 Markdown 或 JSON，至少包含：

```text
sequence_id
VRS path and type (motion/data)
VRS readable
Xsens TIME_CODE start/end
Xsens timestamps monotonic / invalid count
CSV global start/end in Head DEVICE_TIME
CSV start/end converted to TIME_CODE
start delta to Xsens
end delta to Xsens
status: valid / timing_offset / missing_narration / corrupt_or_unreadable
```

并汇总：

```text
总序列数
有效序列数和比例
时钟偏移异常序列
缺失 narration 序列
VRS 不可读序列
Xsens 时间戳异常序列
```

## 10. 论文实验建议判定

在宿主机审计完成前：

- 不宣称 Nymeria 所有序列均已精确对齐。
- 不使用序列起点平移作为官方协议。
- 不使用 Grace 异常序列。

若宿主机全量结果支持主假设，正式预处理采用：

```text
CSV Head DEVICE_TIME -> Head VRS -> TIME_CODE -> Xsens/各设备 VRS
```

并自动排除转换后不落在 Xsens 有效区间或起点误差超过阈值的样本。
