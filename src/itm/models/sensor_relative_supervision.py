"""Training-only auxiliary supervision of instrumented joint motion.

Targets enter the loss only. This is a joint proxy, not sensor-signal closure.
"""
import numpy as np

from itm.metrics.sensor_relative import SENSOR_JOINTS
from itm.data.humanml_torch import recover_from_ric_torch


def fit_sensor_relative_scales(records, load_gt):
    """Equal-clip GT statistics over admissible train sources only, once each."""
    train = [r for r in records if r['split'] == 'train']
    if not train or any(not r['training_admissible'] for r in train):
        raise ValueError('Admissible training clips required')
    if len({r['source_group_id'] for r in train}) != len(train):
        raise ValueError('Each training source must occur once')
    if {r['source_group_id'] for r in train} & {r['source_group_id'] for r in records if r['split'] != 'train'}:
        raise ValueError('Train/held-out source overlap')
    moments = {sensor: [] for sensor in SENSOR_JOINTS}
    for r in train:
        g = np.asarray(load_gt(r), dtype=np.float64)
        if g.shape != (r['frames'], 22, 3) or len(g) < 4 or not np.isfinite(g).all():
            raise ValueError('Complete finite training GT required')
        for sensor, ids in SENSOR_JOINTS.items():
            x = g[:, list(ids)] - g[:, :1]
            moments[sensor].append((x.mean(0), (x*x).mean(0), float(np.mean((np.diff(x, axis=0)*20)**2))))
    result = {}
    for sensor, values in moments.items():
        mean = np.mean([v[0] for v in values], axis=0)
        variance = float(np.mean(np.mean([v[1] for v in values], axis=0) - mean*mean))
        speed_square = float(np.mean([v[2] for v in values]))
        result[sensor] = dict(position_variance_m2=max(variance, 1e-6),
                              velocity_mean_square_m2_s2=max(speed_square, 1e-6))
    return dict(train_motion_ids=[r['motion_id'] for r in train], train_groups=len(train),
                unique_train_frames=sum(r['frames'] for r in train), fps=20,
                reduction='Equal weight per whole training clip; joint/coordinate means; no val/test.',
                floor_square=1e-6, configurations=result)


def sensor_relative_auxiliary_loss(prediction, target, mean, std, config, scales,
                                   imu_present, position_weight=.1, velocity_weight=.1):
    """Accept true-length normalized (T,263) tensors; no padding or GT inputs."""
    import torch
    if config not in SENSOR_JOINTS or prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 263 or len(prediction) < 4:
        raise ValueError('Expected complete equal normalized (T>=4,263) tensors')
    if target.requires_grad: raise ValueError('Supervision target must be detached')
    if position_weight < 0 or velocity_weight < 0:
        raise ValueError('Nonnegative weights required')
    if not imu_present:
        zero = prediction.sum()*0
        return zero, dict(position=zero, velocity=zero)
    scale = scales['configurations'][config]
    if any(not np.isfinite(v) or v <= 0 for v in scale.values()):
        raise ValueError('Finite positive train-only scales required')
    norm_mean = torch.as_tensor(mean, dtype=prediction.dtype, device=prediction.device)
    norm_std = torch.as_tensor(std, dtype=prediction.dtype, device=prediction.device)
    if norm_mean.shape != (263,) or norm_std.shape != (263,):
        raise ValueError('Expected HumanML263 normalization vectors')
    p = recover_from_ric_torch(prediction*norm_std + norm_mean)
    with torch.no_grad():
        g = recover_from_ric_torch(target*norm_std + norm_mean)
    ids = list(SENSOR_JOINTS[config])
    e = (p[:, ids]-p[:, :1]) - (g[:, ids]-g[:, :1])
    position = e.square().mean()/scale['position_variance_m2']
    velocity = ((e[1:]-e[:-1])*20).square().mean()/scale['velocity_mean_square_m2_s2']
    return position_weight*position + velocity_weight*velocity, dict(position=position, velocity=velocity)
