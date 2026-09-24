"""Physical static fixtures in the declared world-linear-acceleration schema."""
import hashlib
import json

import numpy as np

from itm.data.observation import SENSOR_CONFIGS, rotation_valid


def static_imu_fixture(*, frames, fps, sensor_config, normalization, orientations=None):
    """Return real static values and normalized model input, never a null token.

    Constant rotations are declared calibration, not fitted from a target motion.
    All sensor arrays are present; the configuration mask determines active slots.
    """
    if not isinstance(frames, int) or frames < 3 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("at least three frames and positive finite FPS required")
    if sensor_config not in SENSOR_CONFIGS:
        raise ValueError("only head/wrists configurations are supported")
    mean = np.asarray(normalization["acceleration_mean"], dtype=np.float32)
    std = np.asarray(normalization["acceleration_std"], dtype=np.float32)
    if mean.shape != (6,3) or std.shape != (6,3) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("finite (6,3) normalization with positive std required")
    rotations = np.broadcast_to(np.eye(3), (6,3,3)) if orientations is None else np.asarray(orientations)
    if rotations.shape != (6,3,3) or not rotation_valid(rotations).all():
        raise ValueError("six proper constant calibration rotations required")
    acceleration = np.zeros((frames,6,3), dtype=np.float32)
    orientation = np.broadcast_to(rotations, (frames,6,3,3)).astype(np.float32).copy()
    features = np.concatenate(((acceleration-mean[None])/std[None], orientation.reshape(frames,6,9)), -1)
    mask = np.zeros(6,dtype=bool); mask[list(SENSOR_CONFIGS[sensor_config])] = True
    digest = hashlib.sha256(json.dumps(normalization,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return {"acceleration":acceleration,"orientation":orientation,"imu":features,"sensor_mask":mask,
        "timestamps_seconds":np.arange(frames)/fps,
        "metadata":{"schema":"standard_imu_v1", "fixture_version":"physical-static-v1", "sensor_config":sensor_config,
          "quantity":"world_linear_acceleration_and_global_orientation", "gravity_convention":"not_added",
          "calibration":"declared_constant_segment_to_world", "normalization_sha256":digest,
          "imu_condition_present":True,"not_a_null_or_numerical_zero_condition":True}}
