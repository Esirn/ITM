import unittest

import numpy as np

from itm.backbones import MDMBackbone, MotionBackbone, MotionConditions


class BackboneInterfaceTest(unittest.TestCase):
    def test_mdm_frame_mask_preserves_variable_lengths(self):
        import torch

        lengths = torch.tensor([3, 5])
        mask = MDMBackbone.frame_mask(lengths, 5)
        self.assertEqual(mask.tolist(), [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ])

    def test_protocol_accepts_mdm_adapter(self):
        class Dummy:
            pass

        adapter = MDMBackbone(Dummy(), Dummy(), np.zeros(263), np.ones(263), lambda x, n: x)
        self.assertIsInstance(adapter, MotionBackbone)
        conditions = MotionConditions(["walk"], np.array([5]))
        self.assertEqual(conditions.text, ["walk"])


if __name__ == "__main__":
    unittest.main()
