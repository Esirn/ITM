import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "FastAPI is not installed")
class DemoAppTest(unittest.TestCase):
    def test_four_way_conditions(self):
        from itm.demo.app import GenerateRequest, _build_spec

        request = GenerateRequest(mode="four_way", sensor_config="head")
        sample = {"motion_id": "A", "caption": "text a"}
        spec = _build_spec(request, sample, None)
        self.assertEqual(len(spec), 3)
        self.assertEqual(spec[0]["imu_scale"], 0.0)
        self.assertEqual(spec[1]["text_scale"], 0.0)
        self.assertNotIn("imu_scale", spec[2])

    def test_matrix_contains_all_nonempty_pairs(self):
        from itm.demo.app import GenerateRequest, _build_spec

        request = GenerateRequest(mode="matrix", sensor_config="wrists")
        a = {"motion_id": "A", "caption": "text a"}
        b = {"motion_id": "B", "caption": "text b"}
        spec = _build_spec(request, a, b)
        pairs = {(row["text_source"], row["imu_source"]) for row in spec}
        self.assertEqual(len(spec), 8)
        self.assertEqual(
            pairs,
            {
                (None, "A"), (None, "B"),
                ("A", None), ("A", "A"), ("A", "B"),
                ("B", None), ("B", "A"), ("B", "B"),
            },
        )
        cross_ab = next(row for row in spec if row["text_source"] == "A" and row["imu_source"] == "B")
        cross_ba = next(row for row in spec if row["text_source"] == "B" and row["imu_source"] == "A")
        self.assertEqual((cross_ab["motion_id"], cross_ab["text"]), ("B", "text a"))
        self.assertEqual((cross_ba["motion_id"], cross_ba["text"]), ("A", "text b"))

    def test_serialized_ground_truth_panels_include_sample_captions(self):
        from itm.demo.app import _serialize_result

        request = {
            "mode": "matrix",
            "sample_a": "A",
            "sample_b": "B",
            "sensor_config": "head",
        }
        catalog = {
            "A": {"motion_id": "A", "caption": "text a"},
            "B": {"motion_id": "B", "caption": "text b"},
        }
        cases = [
            {"label": "None + IMU A", "text": "", "text_source": None, "imu_source": "A"},
            {"label": "None + IMU B", "text": "", "text_source": None, "imu_source": "B"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "results.npz"
            np.savez_compressed(
                result_path,
                motion=np.zeros((2, 4, 22, 3), dtype=np.float32),
                gt=np.zeros((2, 4, 22, 3), dtype=np.float32),
                acceleration=np.zeros((2, 4, 6, 3), dtype=np.float32),
                orientation=np.zeros((2, 4, 6, 3, 3), dtype=np.float32),
                metadata=np.array(json.dumps({"cases": cases})),
            )
            result = _serialize_result("run", request, catalog, result_path)
        self.assertEqual(
            [panel["caption"] for panel in result["panels"][:2]],
            ["text a", "text b"],
        )

    def test_run_list_route_precedes_dynamic_run_route(self):
        from itm.demo.app import app

        paths = [route.path for route in app.routes]
        self.assertIn("/api/runs", paths)
        self.assertLess(paths.index("/api/runs"), paths.index("/api/runs/{run_id}"))

    def test_sample_catalog_is_sorted_by_motion_id(self):
        from unittest.mock import patch

        import itm.demo.app as demo_app

        manifest = [
            {"motion_id": "006504", "text_path": "b.txt"},
            {"motion_id": "000123", "text_path": "a.txt"},
        ]
        imu = [
            {"motion_id": "000123", "num_frames": 80},
            {"motion_id": "006504", "num_frames": 90},
        ]

        def fake_jsonl(path):
            return imu if path == demo_app.IMU_MANIFESTS["test"] else manifest

        with patch.object(demo_app, "read_jsonl", side_effect=fake_jsonl), patch.object(
            demo_app,
            "read_caption_records",
            side_effect=lambda path: [SimpleNamespace(caption=f"caption {path}")],
        ):
            rows = demo_app._sample_catalog("test")
        self.assertEqual([row["motion_id"] for row in rows], ["000123", "006504"])

    def test_mdm_root_prefers_candidate_with_smpl_assets(self):
        from itm.demo.app import _resolve_mdm_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = root / "runtime"
            complete = root / "full"
            model = complete / "body_models/smpl/SMPL_NEUTRAL.pkl"
            model.parent.mkdir(parents=True)
            model.touch()
            self.assertEqual(_resolve_mdm_root([incomplete, complete]), complete)

    def test_safe_experiment_paths_reject_traversal(self):
        from fastapi import HTTPException
        from itm.demo.app import _safe_relative_path

        self.assertEqual(_safe_relative_path("matrix_test_head/pair_001"), Path("matrix_test_head/pair_001"))
        with self.assertRaises(HTTPException):
            _safe_relative_path("../secret")
        with self.assertRaises(HTTPException):
            _safe_relative_path("/tmp/secret")

    def test_list_experiments_reads_result_directories(self):
        import itm.demo.app as demo_app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "matrix_test_head/pair_001"
            run.mkdir(parents=True)
            (run / "request.json").write_text(json.dumps({
                "experiment": "matrix",
                "split": "test",
                "sample_a": "A",
                "sample_b": "B",
                "sensor_config": "head",
            }))
            (run / "metrics.json").write_text(json.dumps({
                "aggregate": {"all": {"active_sensor_trajectory_error_m": 0.1, "jerk_ratio": 1.2}}
            }))
            (run / "result.json").write_text(json.dumps({"run_id": "x"}))
            old_roots = demo_app.EXPERIMENT_ROOTS
            try:
                demo_app.EXPERIMENT_ROOTS = {"stage1": root}
                rows = demo_app.list_experiments(root="stage1", limit=100)
            finally:
                demo_app.EXPERIMENT_ROOTS = old_roots
            self.assertEqual(rows[0]["run_id"], "matrix_test_head/pair_001")
            self.assertEqual(rows[0]["root_key"], "stage1")
            self.assertEqual(rows[0]["mode"], "matrix")
            self.assertEqual(rows[0]["active_sensor_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
