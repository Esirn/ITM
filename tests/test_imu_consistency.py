import unittest

from itm.metrics.imu_consistency import (
    acceleration_error,
    imu_consistency,
    orientation_vector_error,
)


class IMUConsistencyTest(unittest.TestCase):
    def test_zero_acceleration_for_constant_velocity(self):
        joints = [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ]
        target_acc = [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
        self.assertEqual(acceleration_error(joints, target_acc), 0.0)

    def test_orientation_vector_error(self):
        joints = [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
        target_vectors = [
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
        self.assertEqual(
            orientation_vector_error(joints, target_vectors, [0], [1]),
            0.0,
        )

    def test_aggregate_metric(self):
        joints = [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ]
        target_acc = [[[0.0, 0.0, 0.0]]]
        result = imu_consistency(joints, target_acc)
        self.assertEqual(result.acceleration_l2, 0.0)
        self.assertIsNone(result.orientation_l2)


if __name__ == "__main__":
    unittest.main()

