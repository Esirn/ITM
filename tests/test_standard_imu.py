import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from itm.data.amass_bridge import load_humanml_amass_index, source_crop_at_fps
from itm.data.standard_imu import (
    SENSOR_JOINT_INDICES,
    SENSOR_VERTEX_INDICES,
    humanml22_from_smpl24,
    imuposer_features,
    resample_standard_imu,
    smpl24_from_humanml22,
    synthesize_standard_imu,
)


class StandardIMUTest(unittest.TestCase):
    def test_synthesis_shapes_rotation_and_mask(self):
        frames = 7
        vertices = np.zeros((frames, max(SENSOR_VERTEX_INDICES) + 1, 3), dtype=np.float32)
        time = np.arange(frames, dtype=np.float32)
        vertices[:, SENSOR_VERTEX_INDICES, 0] = time[:, None] ** 2
        rotations = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (frames, max(SENSOR_JOINT_INDICES) + 1, 3, 3),
        ).copy()
        imu = synthesize_standard_imu(
            vertices, rotations, fps=2, sensor_slots=(0, 4), smooth_window=1
        )
        self.assertEqual(imu.acceleration.shape, (frames, 6, 3))
        np.testing.assert_allclose(imu.acceleration[1:-1, :, 0], 8.0)
        np.testing.assert_array_equal(imu.sensor_mask, [True, False, False, False, True, False])
        features = imuposer_features(imu.acceleration, imu.orientation, imu.sensor_mask)
        self.assertEqual(features.shape, (frames, 60))
        np.testing.assert_array_equal(features[:, 3:12], 0.0)

    def test_joint_conversion_preserves_humanml_joints(self):
        joints = np.arange(2 * 22 * 3).reshape(2, 22, 3)
        np.testing.assert_array_equal(humanml22_from_smpl24(smpl24_from_humanml22(joints)), joints)

    def test_resample_identity_returns_independent_arrays(self):
        acceleration = np.zeros((3, 6, 3), dtype=np.float32)
        orientation = np.broadcast_to(np.eye(3), (3, 6, 3, 3)).astype(np.float32)
        output_acceleration, output_orientation = resample_standard_imu(
            acceleration, orientation, source_fps=20, target_fps=20
        )
        np.testing.assert_array_equal(output_acceleration, acceleration)
        np.testing.assert_array_equal(output_orientation, orientation)
        self.assertIsNot(output_acceleration, acceleration)
        self.assertIsNot(output_orientation, orientation)

    def test_humanml_amass_mapping_and_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.csv"
            with index.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("source_path", "start_frame", "end_frame", "new_name"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "source_path": "./pose_data/CMU/80/80_63_poses.npy",
                        "start_frame": 10,
                        "end_frame": 30,
                        "new_name": "000004.npy",
                    }
                )
            records = load_humanml_amass_index(index, "/amass")
            self.assertEqual(records["000004"].source_path, Path("/amass/CMU/80/80_63_poses.npz"))
            self.assertEqual(source_crop_at_fps(records["000004"], 60), slice(30, 90))


if __name__ == "__main__":
    unittest.main()
