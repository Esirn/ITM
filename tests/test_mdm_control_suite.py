import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from itm.experiments.mdm_control_suite import (
    SampleRecord,
    arm_swing_proxy,
    compute_run_metrics,
    pairwise_root_relative_distance,
    serialize_browser_result,
    step_frequency_proxy,
)


class MDMControlSuiteTest(unittest.TestCase):
    def test_motion_proxy_metrics_are_finite(self):
        frames = 40
        joints = np.zeros((frames, 22, 3), dtype=np.float32)
        joints[:, 0, 0] = np.linspace(0.0, 1.0, frames)
        joints[:, 10, 0] = np.sin(np.linspace(0.0, 8.0, frames))
        joints[:, 11, 0] = np.cos(np.linspace(0.0, 8.0, frames))
        joints[:, 16, 1] = 1.0
        joints[:, 17, 1] = 1.0
        joints[:, 20, 0] = np.sin(np.linspace(0.0, 4.0, frames))
        joints[:, 21, 0] = np.cos(np.linspace(0.0, 4.0, frames))
        self.assertGreater(arm_swing_proxy(joints), 0.0)
        self.assertGreaterEqual(step_frequency_proxy(joints, fps=20.0), 0.0)
        self.assertGreaterEqual(pairwise_root_relative_distance(np.stack([joints, joints + 0.1])), 0.0)

    def test_compute_and_serialize_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npz"
            motion = np.zeros((2, 12, 22, 3), dtype=np.float32)
            gt = np.zeros_like(motion)
            motion[1, :, 15, 0] = 0.2
            acceleration = np.zeros((2, 12, 6, 3), dtype=np.float32)
            orientation = np.zeros((2, 12, 6, 3, 3), dtype=np.float32)
            sensor_mask = np.zeros((2, 6), dtype=np.float32)
            sensor_mask[:, 4] = 1.0
            metadata = {
                "control_checkpoint": "ckpt.pt",
                "seed": 1234,
                "text_scale": 2.5,
                "imu_scale": 1.0,
                "device": "cpu",
                "frame_count": 12,
                "cases": [
                    {
                        "motion_id": "A",
                        "text": "a person walks",
                        "label": "Text A + IMU A",
                        "sensor_config": "head",
                        "text_source": "A",
                        "imu_source": "A",
                    },
                    {
                        "motion_id": "B",
                        "text": "a person walks",
                        "label": "Text A + IMU B",
                        "sensor_config": "head",
                        "text_source": "A",
                        "imu_source": "B",
                    },
                ],
            }
            np.savez_compressed(
                path,
                motion=motion,
                gt=gt,
                acceleration=acceleration,
                orientation=orientation,
                sensor_mask=sensor_mask,
                metadata=np.array(json.dumps(metadata)),
            )
            metrics = compute_run_metrics(path)
            self.assertEqual(len(metrics["cases"]), 2)
            self.assertIn("pairwise_root_relative_distance_m", metrics)
            result = serialize_browser_result(
                run_id="run",
                request={"experiment": "matrix"},
                result_path=path,
                samples={
                    "A": SampleRecord("A", "text a", ("text a",), 12),
                    "B": SampleRecord("B", "text b", ("text b",), 12),
                },
            )
            self.assertEqual(result["run_id"], "run")
            self.assertGreaterEqual(len(result["panels"]), 4)
            self.assertIn("A", result["imu"])


if __name__ == "__main__":
    unittest.main()
