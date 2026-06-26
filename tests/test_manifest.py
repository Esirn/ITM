from pathlib import Path
import tempfile
import unittest

from itm.data.manifest import build_manifest_entries, read_jsonl, write_jsonl


class ManifestTest(unittest.TestCase):
    def test_build_and_roundtrip_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            texts = root / "texts"
            joints = root / "new_joints"
            joint_vecs = root / "new_joint_vecs"
            for path in [texts, joints, joint_vecs]:
                path.mkdir()
            (root / "train.txt").write_text("000001\n")
            (texts / "000001.txt").write_text("a person walks#tok#0.0#0.0\n")
            (joints / "000001.npy").write_bytes(b"placeholder")
            (joint_vecs / "000001.npy").write_bytes(b"placeholder")

            entries = build_manifest_entries(
                split="train",
                split_file=root / "train.txt",
                texts_dir=texts,
                joints_dir=joints,
                joint_vecs_dir=joint_vecs,
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].motion_id, "000001")
            self.assertEqual(entries[0].num_texts, 1)

            output = root / "manifest.jsonl"
            self.assertEqual(write_jsonl(entries, output), 1)
            records = list(read_jsonl(output))
            self.assertEqual(records[0]["motion_id"], "000001")

    def test_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.txt").write_text("000001\n")
            with self.assertRaises(FileNotFoundError):
                build_manifest_entries(
                    split="train",
                    split_file=root / "train.txt",
                    texts_dir=root / "texts",
                    joints_dir=root / "new_joints",
                    joint_vecs_dir=root / "new_joint_vecs",
                )


if __name__ == "__main__":
    unittest.main()

