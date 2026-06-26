from pathlib import Path
import tempfile
import unittest

import numpy as np

from itm.data.imu_cache import build_imu_cache, write_imu_cache_manifest
from itm.data.manifest import ManifestEntry, read_jsonl, write_jsonl


class IMUCacheTest(unittest.TestCase):
    def test_build_imu_cache_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            joints_path = root / "000001.npy"
            text_path = root / "000001.txt"
            manifest_path = root / "manifest.jsonl"
            cache_manifest_path = root / "imu_cache.jsonl"

            joints = np.zeros((4, 22, 3), dtype=np.float32)
            for t in range(joints.shape[0]):
                joints[t, :, 0] = float(t)
                joints[t, :, 1] = np.arange(joints.shape[1], dtype=np.float32)
            np.save(joints_path, joints)
            text_path.write_text("a person walks#tok#0.0#0.0\n")
            write_jsonl(
                [
                    ManifestEntry(
                        motion_id="000001",
                        split="train",
                        text_path=str(text_path),
                        joints_path=str(joints_path),
                        joint_vec_path=str(root / "000001_vec.npy"),
                        num_texts=1,
                    )
                ],
                manifest_path,
            )

            entries = build_imu_cache(
                manifest_path,
                root / "imu",
                include_orientation=True,
                overwrite=True,
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].acceleration_shape, (2, 6, 3))
            self.assertEqual(entries[0].orientation_shape, (4, 6, 3))

            with np.load(entries[0].imu_path) as cached:
                self.assertEqual(cached["acceleration"].shape, (2, 6, 3))
                self.assertEqual(cached["orientation_vectors"].shape, (4, 6, 3))
                self.assertEqual(
                    cached["sensor_joint_indices"].tolist(),
                    [0, 7, 8, 15, 20, 21],
                )

            self.assertEqual(write_imu_cache_manifest(entries, cache_manifest_path), 1)
            records = list(read_jsonl(cache_manifest_path))
            self.assertEqual(records[0]["motion_id"], "000001")
            self.assertEqual(records[0]["acceleration_shape"], [2, 6, 3])


if __name__ == "__main__":
    unittest.main()
