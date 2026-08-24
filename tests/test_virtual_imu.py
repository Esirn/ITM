import importlib.util
import unittest

import numpy as np

from itm.metrics.virtual_imu import rotation_geodesic_error, vector_l2_error


class VirtualIMUMetricsTest(unittest.TestCase):
    def test_rotation_geodesic_error(self):
        identity = np.eye(3)[None, None]
        half_turn = np.diag([-1.0, -1.0, 1.0])[None, None]
        self.assertAlmostEqual(rotation_geodesic_error(identity, identity), 0.0)
        self.assertAlmostEqual(rotation_geodesic_error(identity, half_turn), np.pi)

    def test_vector_error_respects_sensor_mask(self):
        predicted = np.zeros((3, 2, 3))
        target = np.zeros_like(predicted)
        target[:, 0, 0] = 2.0
        target[:, 1, 0] = 100.0
        self.assertAlmostEqual(vector_l2_error(predicted, target, [True, False]), 2.0)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
    def test_local_to_global_rotations(self):
        import torch
        from itm.data.smpl_fitting import local_to_global_rotations

        local = torch.eye(3).repeat(2, 3, 1, 1)
        global_rotation = local_to_global_rotations(local, torch.tensor([-1, 0, 1]))
        self.assertEqual(tuple(global_rotation.shape), (2, 3, 3, 3))
        self.assertTrue(torch.allclose(global_rotation, local))


if __name__ == "__main__":
    unittest.main()
