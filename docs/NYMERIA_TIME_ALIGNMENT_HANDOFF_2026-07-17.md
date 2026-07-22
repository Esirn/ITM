# Nymeria 时间对齐交接文档（2026-07-17）

## 1. 结论

Nymeria 中建议使用统一的公共时间轴 `TIME_CODE` 做多模态对齐。

核心协议：

```text
narration/*.csv start_time/end_time
    = Head Aria DEVICE_TIME，单位秒

video_main_rgb.mp4 每帧时间戳
    = Head Aria DEVICE_TIME，单位纳秒，存放在 MP4 metadata description 中

body/xdata.npz timestamps_us
    = 公共 TIME_CODE，单位微秒

recording_head/data/{motion.vrs,data.vrs}
    = Head DEVICE_TIME <-> 公共 TIME_CODE 的官方转换器
```

因此：

```text
CSV <-> MP4:
    可以直接按 Head DEVICE_TIME 对齐。

CSV/MP4 <-> xdata.npz / Xsens IMU / 全身 GT / wrist VRS:
    必须先通过 Head VRS 转成 TIME_CODE。
```

不要把 CSV 秒数直接和 `xdata["timestamps_us"] / 1e6` 比较；它们不是同一个时间域。

## 2. 必需文件

若目标是对齐文本描述、IMU 和全身 GT，至少需要：

```text
narration/*.csv
recording_head/data/data.vrs 或 recording_head/data/motion.vrs
body/xdata.npz
```

各自作用：

```text
narration/*.csv
    文本描述与 start_time/end_time。

Head VRS
    把 CSV/MP4 的 Head DEVICE_TIME 转成公共 TIME_CODE。

body/xdata.npz
    Xsens 全身 GT、Xsens IMU，以及 TIME_CODE 时间戳。
```

如果还需要 Aria 头戴或手腕 IMU，需要对应设备 VRS：

```text
recording_head/data/data.vrs 或 motion.vrs
recording_lwrist/data/data.vrs 或 motion.vrs
recording_rwrist/data/data.vrs 或 motion.vrs
```

注意：Head `DEVICE_TIME` 不能直接用于 wrist VRS。应先转成公共 `TIME_CODE`，再用 wrist VRS 查询或转换到 wrist 自己的 `DEVICE_TIME`。

## 3. Python 对齐模板

### 3.1 CSV 时间转 TIME_CODE

```python
from pathlib import Path
import numpy as np
from projectaria_tools.core import data_provider

seq = Path("/home/20T-2/group_motion/datasets/Nymeria/20230921_s1_alec_meza_act0_8ytqbv")

head_vrs = seq / "recording_head/data/motion.vrs"
if not head_vrs.is_file():
    head_vrs = seq / "recording_head/data/data.vrs"

provider = data_provider.create_vrs_data_provider(str(head_vrs))

csv_start_s = 317.553510
csv_end_s = 322.552711

start_device_ns = round(csv_start_s * 1e9)
end_device_ns = round(csv_end_s * 1e9)

start_timecode_ns = provider.convert_from_device_time_to_timecode_ns(start_device_ns)
end_timecode_ns = provider.convert_from_device_time_to_timecode_ns(end_device_ns)
```

### 3.2 对齐到 xdata.npz

```python
xdata = np.load(seq / "body/xdata.npz", allow_pickle=False)
xsens_timecode_ns = xdata["timestamps_us"].astype(np.int64) * 1000

i0 = int(np.searchsorted(xsens_timecode_ns, start_timecode_ns, side="left"))
i1_excl = int(np.searchsorted(xsens_timecode_ns, end_timecode_ns, side="right"))

segment_tXYZ = xdata["segment_tXYZ"][i0:i1_excl]
segment_qWXYZ = xdata["segment_qWXYZ"][i0:i1_excl]
xsens_acc = xdata["sensor_freeAcceleration"][i0:i1_excl]
xsens_q = xdata["sensor_qWXYZ"][i0:i1_excl]
```

## 4. MP4 时间戳读取与对齐

Nymeria 的 `video_main_rgb.mp4` 在 metadata 的 `description` 字段中保存了每帧 Head `DEVICE_TIME`，单位纳秒。

读取方式：

```python
import re
import subprocess
import numpy as np

video = seq / "video_main_rgb.mp4"

p = subprocess.run(
    [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format_tags=description",
        "-of", "default=nw=1:nk=1",
        str(video),
    ],
    capture_output=True,
    check=True,
    text=True,
)

frame_device_ns = np.asarray(
    [int(x) for x in re.findall(r"\d+", p.stdout)],
    dtype=np.int64,
)
```

CSV 与 MP4 都是 Head `DEVICE_TIME`，因此可以直接选帧：

```python
mask = (frame_device_ns >= start_device_ns) & (frame_device_ns <= end_device_ns)
frame_indices = np.flatnonzero(mask)
```

MP4 播放秒数：

```python
playback_s = (frame_device_ns[frame_indices] - frame_device_ns[0]) / 1e9
```

若要把 MP4 帧对齐到 `.npz` 或 VRS，应把每帧转成 `TIME_CODE`：

```python
frame_timecode_ns = np.asarray(
    [provider.convert_from_device_time_to_timecode_ns(int(t)) for t in frame_device_ns],
    dtype=np.int64,
)
```

## 5. 具体例子：Alec act0

序列：

```text
20230921_s1_alec_meza_act0_8ytqbv
```

`atomic_action.csv` 某段：

```text
start_time = 317.553510 s
end_time   = 322.552711 s
```

这是 Head `DEVICE_TIME`。

MP4 第一帧 Head `DEVICE_TIME` 来自 metadata：

```bash
ffprobe -v error \
  -show_entries format_tags=description \
  -of default=nw=1:nk=1 \
  /home/20T-2/group_motion/datasets/Nymeria/20230921_s1_alec_meza_act0_8ytqbv/video_main_rgb.mp4
```

输出开头：

```text
[100454923050, 100488250387, 100521577725, ...]
```

所以 MP4 第一帧：

```text
100454923050 ns = 100.454923050 s Head DEVICE_TIME
```

该 action 对应：

```text
MP4 frame index:
    6513 到 6662，共 150 帧

MP4 播放时间:
    217.098588 s 到 222.064461 s

TIME_CODE:
    1184.599529239 s 到 1189.598743502 s

xdata.npz frame index:
    12049 到 13248，共 1200 帧

xdata.npz 实际首尾 TIME_CODE:
    1184.601500 s 到 1189.597200 s
```

其中 MP4 播放时间计算方式：

```text
MP4 playback time = frame_device_time - first_frame_device_time
```

## 6. 如何证明 VRS 能转换时间

Project Aria provider 中有官方转换接口：

```python
provider.convert_from_device_time_to_timecode_ns(device_ns)
provider.convert_from_timecode_to_device_time_ns(timecode_ns)
```

检查脚本：

```bash
/home/syx0011/miniconda3/envs/nymeriaplus/bin/python \
  scripts/inspect_vrs_time_domains.py \
  /home/20T-2/group_motion/datasets/Nymeria/20230921_s1_alec_meza_act0_8ytqbv
```

关键输出：

```text
time_sync_mode: MetadataTimeSyncMode.Timecode

All-stream time ranges:
  DEVICE     100.404679 .. 1236.580925 s
  TIME_CODE  967.450345 .. 2103.630002 s

Conversion proof:
  CSV start_time          267.561519000 s
  as DEVICE_TIME          267561519000 ns
  converted TIME_CODE     1134607419343 ns
  roundtrip DEVICE_TIME   267561518999 ns
  Xsens first TIME_CODE   1134397800000 ns
  delta CSV-Xsens start   0.209619343 s
```

这证明：

```text
同一个 Head VRS 同时知道 DEVICE_TIME 和 TIME_CODE，
并可通过官方 API 做双向转换。
```

## 7. 当前已验证的数据状态

数据根目录：

```text
/home/20T-2/group_motion/datasets/Nymeria/
```

异常目录：

```text
/home/20T-2/group_motion/datasets/Nymeria-error/
```

截至 2026-07-17 最新复查，两个分钟级 offset 异常序列已经移出正式 Nymeria 根目录：

```text
/home/20T-2/group_motion/datasets/Nymeria-error/
  20230928_s0_grace_randolph_act0_rsm00j
  20230928_s0_grace_randolph_act2_0xxd51
```

当前 `/home/20T-2/group_motion/datasets/Nymeria/` 下剩余 34 条正式实验序列。合并 VRS 官方转换审计后：

```text
当前序列数: 34
大时间偏移序列: 0

CSV start -> TIME_CODE 相对 Xsens start 偏移:
    min:  0.200597130 s
    max:  0.231683317 s
    mean: 0.215021085 s
```

这与正常序列的形态一致：标注起点通常比 Xsens 录制起点晚约 0.20 到 0.23 秒。

MP4 metadata 辅助审计：

```text
当前 34 条中：
    33 条 CSV 完整落在 MP4 Head DEVICE_TIME 范围内。
    1 条 20230927_s1_samantha_may_act0_kb2wve 无可用 MP4 metadata，
        但该序列 motion.vrs 官方转换为 valid。
```

Samantha 的 VRS 结果：

```text
20230927_s1_samantha_may_act0_kb2wve
    status: valid
    CSV start -> TIME_CODE 相对 Xsens start: +0.213315480 s
```

相关结果文件：

```text
最终当前目录汇总:
  nymeria_current_combined_vrs_time_alignment_20260717.json
  nymeria_current_video_device_time_audit_20260717.json

四条重下序列复查:
nymeria_recheck_4seq_vrs_time_alignment_20260717.json
nymeria_recheck_4seq_video_device_time_20260717.json

历史全量审计:
nymeria_vrs_time_alignment_audit_timeout_20260716.jsonl
nymeria_video_device_time_audit_20260716.json
```

注意：不要使用 `nymeria_current_vrs_time_alignment_audit_20260717.jsonl` 作为最终判断。该文件来自一次 20 秒逐条超时的中途审计，阈值太紧，会把部分读取较慢的 `data.vrs` 误记为 `timeout_or_error`。当前最终判断使用 `nymeria_current_combined_vrs_time_alignment_20260717.json`。

## 8. 已提供脚本

```text
scripts/inspect_vrs_time_domains.py
    打印 VRS stream、DEVICE_TIME/TIME_CODE 范围、转换示例和一条 IMU 记录。

scripts/verify_nymeria_vrs_time_alignment.py
    对序列执行 CSV -> Head VRS -> TIME_CODE -> xdata.npz 的验证。

scripts/verify_nymeria_video_device_time.py
    用 MP4 metadata 检查 CSV 是否落在 Head RGB DEVICE_TIME 范围内。

scripts/audit_nymeria_time_alignment.py
    早期审计脚本。注意它依赖 imageio_ffmpeg；若环境缺包，优先用上面两个 verify 脚本。
```

推荐环境：

```bash
/home/syx0011/miniconda3/envs/nymeriaplus/bin/python
```

需要：

```text
projectaria_tools
numpy
ffprobe
```

## 9. 推荐实验流程

对每个 action row：

1. 读取 CSV：

```text
start_time/end_time: Head DEVICE_TIME 秒
text label: action/narration 字段
```

2. 转换时间：

```text
start_device_ns = round(start_time * 1e9)
end_device_ns   = round(end_time * 1e9)

start_timecode_ns = Head VRS DEVICE_TIME -> TIME_CODE
end_timecode_ns   = Head VRS DEVICE_TIME -> TIME_CODE
```

3. MP4：

```text
用 frame_device_ns 直接筛选 [start_device_ns, end_device_ns]
```

4. xdata.npz：

```text
用 xdata["timestamps_us"] * 1000 筛选 [start_timecode_ns, end_timecode_ns]
```

5. Head/Wrist VRS IMU：

```text
优先用 TIME_CODE 查询；
若 API 需要 DEVICE_TIME，则用对应设备自己的 VRS 做 TIME_CODE -> DEVICE_TIME。
```

6. 过滤异常：

```text
若 CSV 转 TIME_CODE 后明显不落在 Xsens 时间范围内，排除。
若起点相对 Xsens start 偏移为分钟级，排除。
正常序列起点通常约晚于 Xsens start 0.20 到 0.23 秒。
```

## 10. 常见坑

### 坑 1：把 CSV 秒数直接和 xdata 时间比

错误：

```python
csv_start_s = 317.553510
xsens_s = xdata["timestamps_us"] / 1e6
```

这会错，因为 CSV 是 Head `DEVICE_TIME`，xdata 是 `TIME_CODE`。

### 坑 2：把 Head DEVICE_TIME 直接用于 wrist VRS

错误：

```text
CSV Head DEVICE_TIME -> 直接查 wrist VRS
```

正确：

```text
CSV Head DEVICE_TIME -> Head VRS 转 TIME_CODE -> wrist VRS 查询/转换
```

### 坑 3：MP4 播放秒数不等于 Head DEVICE_TIME 秒

MP4 metadata 中的每帧时间是 Head `DEVICE_TIME`。若要得到播放器里的秒数，需要减去第一帧的 Head `DEVICE_TIME`：

```text
playback_s = (frame_device_ns - frame_device_ns[0]) / 1e9
```

### 坑 4：VRS record index 不一定适合作为数据切片编号

对 `.npz` 和 MP4 可以直接用数组 index 切片。

对 VRS，推荐按时间查询：

```python
get_imu_data_by_time_ns(..., TimeDomain.TIME_CODE 或 DEVICE_TIME, ...)
get_image_data_by_time_ns(..., TimeDomain.TIME_CODE 或 DEVICE_TIME, ...)
```

不要假设 VRS 内部 index 与 MP4 frame index 完全等价。
