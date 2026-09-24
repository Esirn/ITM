"""Method-side adapters to owner-02's versioned cache/time interfaces.

Sampling on cache-local seconds does not certify HumanML/world alignment.
"""
from pathlib import Path

import numpy as np

from itm.data.observation import load_standard_cache
from itm.data.physical_time import resample_positions_at_times

INPUT_PIPELINE_VERSION = "itm-method-seconds-v2"
INPUT_CONTRACT_VERSION = "itm-rework-v0.2-draft"


def load_method_cache(record, *, workspace_root):
    item = dict(record)
    path = Path(item["imu_path"])
    item["imu_path"] = str(path if path.is_absolute() else Path(workspace_root) / path)
    return load_standard_cache(item)


def supported_frame_times(source_times, *, fps=20.0):
    source = np.asarray(source_times, dtype=np.float64)
    if (source.ndim != 1 or len(source) < 2 or not np.isfinite(source).all()
            or (np.diff(source) <= 0).any() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Finite increasing source times and positive FPS required")
    # Compare actual timestamps, not rounded durations or a clamped endpoint.
    times = source[0] + np.arange(int(np.ceil((source[-1] - source[0]) * fps)) + 1) / fps
    return times[times <= source[-1]]


def sample_imu_at_times(cache, target_times):
    from scipy.spatial.transform import Rotation, Slerp

    source = cache["timestamps"]
    acceleration = resample_positions_at_times(cache["acceleration"], source, target_times)
    # Use exactly the same target times for both modalities; no endpoint clamping.
    orientation = np.stack([
        Slerp(source, Rotation.from_matrix(cache["orientation"][:, slot]))(target_times).as_matrix()
        for slot in range(6)
    ], axis=1)
    return acceleration.astype(np.float32), orientation.astype(np.float32)


def normalize_imu(acceleration, orientation, normalization):
    mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    if (mean.shape != (6, 3) or std.shape != (6, 3) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Finite per-slot normalization and positive std required")
    return np.concatenate(((acceleration - mean[None]) / std[None],
                           orientation.reshape(len(orientation), 6, 9)), axis=-1).astype(np.float32)
