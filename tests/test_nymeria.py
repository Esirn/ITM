import numpy as np

from itm.data.nymeria import (
    convert_clip,
    device_seconds_to_timecode_us,
    humanml22_from_nymeria23,
    nearest_time_slice,
)


def test_device_time_mapping_uses_audited_offset():
    audit = {"csv_start_device_s": 10.0, "csv_start_timecode_ns": 12_000_000_000}
    assert device_seconds_to_timecode_us(11.5, audit) == 13_500_000


def test_nymeria_joint_mapping_matches_humanml_order():
    source = np.arange(23, dtype=np.float32)[None, :, None] * np.ones((1, 1, 3))
    mapped = humanml22_from_nymeria23(source)
    assert mapped.shape == (1, 22, 3)
    assert mapped[0, 0, 0] == 0
    assert mapped[0, 1, 0] == 19
    assert mapped[0, 21, 0] == 10


def test_nearest_time_slice_is_end_inclusive():
    result = nearest_time_slice(np.array([10, 20, 30, 40, 50]), 20, 40)
    assert (result.start, result.stop) == (1, 4)


def test_convert_clip_accepts_flattened_xdata_arrays():
    frames = 4
    joints = np.zeros((frames, 23, 3), dtype=np.float32)
    joints[:, 16, 0] = 0.2
    joints[:, 20, 0] = -0.2
    joints[:, 8, 0] = 0.3
    joints[:, 12, 0] = -0.3
    acceleration = np.zeros((frames, 17, 3), dtype=np.float32)
    orientation = np.zeros((frames, 17, 4), dtype=np.float32)
    orientation[..., 0] = 1.0
    output = convert_clip(
        joints.reshape(frames, -1),
        acceleration.reshape(frames, -1),
        orientation.reshape(frames, -1),
    )
    assert output[0].shape == (frames, 22, 3)
    assert output[1].shape == (frames, 6, 3)
    assert output[2].shape == (frames, 6, 3, 3)
