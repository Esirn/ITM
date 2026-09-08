import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_mdm_joint_scale_sweep.py"
SPEC = importlib.util.spec_from_file_location("prepare_mdm_joint_scale_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MDMJointScaleSweepTest(unittest.TestCase):
    def test_parse_scales_and_keys(self):
        self.assertEqual(MODULE.parse_scales("0,0.25,1"), [0.0, 0.25, 1.0])
        self.assertEqual(MODULE.scale_key(0.25), "joint_0p25")
        with self.assertRaises(ValueError):
            MODULE.parse_scales("1,1")
        with self.assertRaises(ValueError):
            MODULE.parse_scales("-1")

    def test_build_cases_only_changes_joint_scale(self):
        subset = {"motion_ids": ["A"], "captions": ["walk"], "lengths": [42]}
        case = MODULE.build_cases(subset, "head", 0.5)[0]
        self.assertEqual(case["motion_id"], "A")
        self.assertEqual(case["imu_motion_id"], "A")
        self.assertEqual(case["sensor_config"], "head")
        self.assertEqual(case["text_scale"], 2.5)
        self.assertEqual(case["imu_scale"], 1.0)
        self.assertEqual(case["joint_scale"], 0.5)


if __name__ == "__main__":
    unittest.main()
