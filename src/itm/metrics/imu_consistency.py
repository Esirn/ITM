"""IMU consistency metrics for motion generation.

The functions here intentionally use only the Python standard library so they
can run during early data audits before the final training stack is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


Array = list


@dataclass(frozen=True)
class IMUConsistency:
    """Aggregate IMU consistency errors."""

    acceleration_l2: float
    orientation_l2: Optional[float]


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    return (len(value),) + _shape(value[0])


def _as_float_array(name: str, value: object, ndim: int) -> Array:
    shape = _shape(value)
    if len(shape) != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {len(shape)}")

    def convert(item: object) -> object:
        if isinstance(item, list):
            return [convert(child) for child in item]
        return float(item)

    return convert(value)  # type: ignore[return-value]


def _vector_l2(vector: list[float]) -> float:
    return math.sqrt(sum(component * component for component in vector))


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of an empty list")
    return sum(values) / len(values)


def second_difference_acceleration(joints: Array, dt: float = 1.0) -> Array:
    """Estimate acceleration from joint positions with a second difference.

    Args:
        joints: Joint positions shaped ``[T, S, 3]``.
        dt: Frame interval. The default follows Ego4o-style configs where
            sequence frames are already normalized to the working frame rate.

    Returns:
        Acceleration shaped ``[T - 2, S, 3]``.
    """

    if dt <= 0:
        raise ValueError("dt must be positive")
    joints = _as_float_array("joints", joints, 3)
    if len(joints) < 3:
        raise ValueError("joints must contain at least 3 frames")
    output = []
    for t in range(len(joints) - 2):
        frame = []
        for sensor_idx in range(len(joints[t])):
            frame.append(
                [
                    (
                        joints[t + 2][sensor_idx][axis]
                        - 2.0 * joints[t + 1][sensor_idx][axis]
                        + joints[t][sensor_idx][axis]
                    )
                    / (dt * dt)
                    for axis in range(3)
                ]
            )
        output.append(frame)
    return output


def acceleration_error(pred_joints: Array, target_acc: Array, dt: float = 1.0) -> float:
    """Mean L2 acceleration error between generated motion and IMU readings."""

    pred_acc = second_difference_acceleration(pred_joints, dt=dt)
    target_acc = _as_float_array("target_acc", target_acc, 3)
    if len(target_acc) == len(pred_acc) + 2:
        target_acc = target_acc[1:-1]
    if _shape(pred_acc) != _shape(target_acc):
        raise ValueError(
            "predicted acceleration and target acceleration shapes differ: "
            f"{_shape(pred_acc)} vs {_shape(target_acc)}"
        )
    errors = []
    for t in range(len(pred_acc)):
        for sensor_idx in range(len(pred_acc[t])):
            errors.append(
                _vector_l2(
                    [
                        pred_acc[t][sensor_idx][axis] - target_acc[t][sensor_idx][axis]
                        for axis in range(3)
                    ]
                )
            )
    return float(_mean(errors))


def limb_orientation_vectors(joints: Array, parents: Array, children: Array) -> Array:
    """Compute unit limb direction vectors for selected sensor-attached limbs."""

    joints = _as_float_array("joints", joints, 3)
    parents = [int(value) for value in parents]
    children = [int(value) for value in children]
    if len(parents) != len(children):
        raise ValueError("parents and children must have the same shape")
    output = []
    for frame in joints:
        frame_vectors = []
        for parent, child in zip(parents, children):
            vector = [frame[child][axis] - frame[parent][axis] for axis in range(3)]
            norm = max(_vector_l2(vector), 1e-8)
            frame_vectors.append([component / norm for component in vector])
        output.append(frame_vectors)
    return output


def orientation_vector_error(
    pred_joints: Array,
    target_vectors: Array,
    parents: Array,
    children: Array,
) -> float:
    """Mean L2 error between generated limb directions and IMU orientations."""

    pred_vectors = limb_orientation_vectors(pred_joints, parents, children)
    target_vectors = _as_float_array("target_vectors", target_vectors, 3)
    if _shape(pred_vectors) != _shape(target_vectors):
        raise ValueError(
            "predicted orientation and target orientation shapes differ: "
            f"{_shape(pred_vectors)} vs {_shape(target_vectors)}"
        )
    errors = []
    for t in range(len(pred_vectors)):
        for sensor_idx in range(len(pred_vectors[t])):
            errors.append(
                _vector_l2(
                    [
                        pred_vectors[t][sensor_idx][axis]
                        - target_vectors[t][sensor_idx][axis]
                        for axis in range(3)
                    ]
                )
            )
    return float(_mean(errors))


def imu_consistency(
    pred_joints: Array,
    target_acc: Array,
    *,
    dt: float = 1.0,
    target_orientation_vectors: Optional[Array] = None,
    parents: Optional[Array] = None,
    children: Optional[Array] = None,
) -> IMUConsistency:
    """Compute acceleration and optional orientation consistency."""

    acc = acceleration_error(pred_joints, target_acc, dt=dt)
    ori = None
    if target_orientation_vectors is not None:
        if parents is None or children is None:
            raise ValueError("parents and children are required for orientation error")
        ori = orientation_vector_error(
            pred_joints,
            target_orientation_vectors,
            parents,
            children,
        )
    return IMUConsistency(acceleration_l2=acc, orientation_l2=ori)


def _smoke_test() -> None:
    joints = []
    for t in range(5):
        joints.append([[float(t), 0.0, 0.0], [float(t), 1.0, 0.0]])
    acc = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in range(3)]
    result = imu_consistency(joints, acc)
    print(result)


if __name__ == "__main__":
    _smoke_test()
