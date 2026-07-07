import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    def test_run_list_route_precedes_dynamic_run_route(self):
        from itm.demo.app import app

        paths = [route.path for route in app.routes]
        self.assertIn("/api/runs", paths)
        self.assertLess(paths.index("/api/runs"), paths.index("/api/runs/{run_id}"))

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
            old_root = demo_app.EXPERIMENT_ROOT
            try:
                demo_app.EXPERIMENT_ROOT = root
                rows = demo_app.list_experiments(limit=100)
            finally:
                demo_app.EXPERIMENT_ROOT = old_root
            self.assertEqual(rows[0]["run_id"], "matrix_test_head/pair_001")
            self.assertEqual(rows[0]["mode"], "matrix")
            self.assertEqual(rows[0]["active_sensor_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
