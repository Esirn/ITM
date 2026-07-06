import importlib.util
import unittest
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


if __name__ == "__main__":
    unittest.main()
