import unittest

import numpy as np

from itm.data.humanml import recover_from_ric


class HumanMLRecoveryTest(unittest.TestCase):
    def test_recovers_identity_facing_local_positions(self):
        features = np.zeros((3, 263), dtype=np.float32)
        features[:, 3] = 1.0
        local = np.arange(63, dtype=np.float32).reshape(21, 3) / 100
        features[:, 4:67] = local.reshape(-1)
        joints = recover_from_ric(features)
        self.assertEqual(joints.shape, (3, 22, 3))
        np.testing.assert_allclose(joints[:, 0, 1], 1.0)
        np.testing.assert_allclose(joints[:, 1:], np.broadcast_to(local, (3, 21, 3)))

    def test_rejects_invalid_width(self):
        with self.assertRaises(ValueError):
            recover_from_ric(np.zeros((3, 10), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
