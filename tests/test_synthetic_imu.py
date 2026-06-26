import unittest

from itm.data.synthetic_imu import synthesize_sparse_imu


class SyntheticIMUTest(unittest.TestCase):
    def test_synthesize_acceleration_and_orientation(self):
        joints = [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
        ]
        imu = synthesize_sparse_imu(
            joints,
            [0],
            parent_joint_indices=[0],
            child_joint_indices=[1],
        )
        self.assertEqual(imu.acceleration, [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
        self.assertEqual(
            imu.orientation_vectors,
            [
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ],
        )


if __name__ == "__main__":
    unittest.main()

