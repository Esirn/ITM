from pathlib import Path
import tempfile
import unittest

import numpy as np

from itm.data.dataset import (
    TextIMUMotionDataset,
    collate_text_imu_motion,
    parse_caption_line,
)
from itm.data.imu_cache import build_imu_cache, write_imu_cache_manifest
from itm.data.manifest import ManifestEntry, write_jsonl


class DatasetTest(unittest.TestCase):
    def test_parse_caption_line_tolerates_missing_fields(self):
        parsed = parse_caption_line("a person walks#tok tok#0.0#1.5")
        self.assertEqual(parsed.caption, "a person walks")
        self.assertEqual(parsed.tokens, "tok tok")
        self.assertEqual(parsed.end, 1.5)

        minimal = parse_caption_line("stand still")
        self.assertEqual(minimal.caption, "stand still")
        self.assertEqual(minimal.start, 0.0)

    def test_dataset_loads_cache_and_collates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            cache_manifest_path = root / "imu_cache.jsonl"

            entries = []
            for motion_id, frames in [("000001", 4), ("000002", 6)]:
                joints_path = root / f"{motion_id}_joints.npy"
                joint_vec_path = root / f"{motion_id}_vec.npy"
                text_path = root / f"{motion_id}.txt"
                joints = np.zeros((frames, 22, 3), dtype=np.float32)
                for t in range(frames):
                    joints[t, :, 0] = float(t)
                    joints[t, :, 1] = np.arange(22, dtype=np.float32)
                joint_vec = np.ones((frames, 263), dtype=np.float32) * frames
                np.save(joints_path, joints)
                np.save(joint_vec_path, joint_vec)
                text_path.write_text(
                    f"caption {motion_id}#token#0.0#0.0\n",
                    encoding="utf-8",
                )
                entries.append(
                    ManifestEntry(
                        motion_id=motion_id,
                        split="train",
                        text_path=str(text_path),
                        joints_path=str(joints_path),
                        joint_vec_path=str(joint_vec_path),
                        num_texts=1,
                    )
                )

            write_jsonl(entries, manifest_path)
            cache_entries = build_imu_cache(manifest_path, root / "imu", overwrite=True)
            write_imu_cache_manifest(cache_entries, cache_manifest_path)

            dataset = TextIMUMotionDataset(
                manifest_path,
                imu_cache_manifest_path=cache_manifest_path,
            )
            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(sample["caption"], "caption 000001")
            self.assertEqual(sample["motion"].shape, (4, 263))
            self.assertEqual(sample["joints"].shape, (4, 22, 3))
            self.assertEqual(sample["imu_acceleration"].shape, (2, 6, 3))
            self.assertEqual(sample["imu_orientation"].shape, (4, 6, 3))

            batch = collate_text_imu_motion([dataset[0], dataset[1]])
            self.assertEqual(batch["motion"].shape, (2, 6, 263))
            self.assertEqual(batch["joints"].shape, (2, 6, 22, 3))
            self.assertEqual(batch["imu_acceleration"].shape, (2, 4, 6, 3))
            self.assertEqual(batch["imu_orientation"].shape, (2, 6, 6, 3))
            self.assertEqual(batch["motion_length"].tolist(), [4, 6])
            self.assertEqual(batch["imu_acceleration_length"].tolist(), [2, 4])
            self.assertEqual(batch["motion_mask"][0].tolist(), [True, True, True, True, False, False])
            self.assertEqual(batch["imu_acceleration_mask"][0].tolist(), [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
