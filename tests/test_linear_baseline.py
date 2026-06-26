import unittest

import numpy as np

from itm.baselines.linear_reconstruct import (
    LinearBaselineConfig,
    build_frame_features,
    evaluate_linear_baseline,
    fit_linear_baseline,
    load_linear_baseline,
    predict_motion,
    save_linear_baseline,
    text_hash_features,
)


class LinearBaselineTest(unittest.TestCase):
    def test_text_hash_features_are_stable_and_normalized(self):
        first = text_hash_features("A person walks forward.", 16)
        second = text_hash_features("A person walks forward.", 16)
        np.testing.assert_allclose(first, second)
        self.assertLessEqual(float(np.linalg.norm(first)), 1.0 + 1e-6)

    def test_fit_predict_and_evaluate(self):
        config = LinearBaselineConfig(text_dim=8, ridge_alpha=1e-4)
        samples = []
        for frames in [4, 5]:
            motion = np.zeros((frames, 3), dtype=np.float32)
            motion[:, 0] = np.linspace(0.0, 1.0, frames)
            motion[:, 1] = 2.0
            motion[:, 2] = -1.0
            samples.append(
                {
                    "motion_id": str(frames),
                    "caption": "walk forward",
                    "motion": motion,
                    "imu_acceleration": np.zeros((frames - 2, 2, 3), dtype=np.float32),
                    "imu_orientation": np.zeros((frames, 2, 3), dtype=np.float32),
                }
            )

        features = build_frame_features(samples[0], config)
        self.assertEqual(features.shape[0], 4)
        self.assertEqual(features.shape[1], 1 + 3 + 8 + 6 + 6)

        weights, train_stats = fit_linear_baseline(samples, config)
        self.assertEqual(weights.shape[1], 3)
        self.assertEqual(int(train_stats["num_records"]), 2)
        prediction = predict_motion(samples[0], weights, config)
        self.assertEqual(prediction.shape, (4, 3))
        stats = evaluate_linear_baseline(samples, weights, config)
        self.assertLess(stats["mse"], 1e-4)

    def test_feature_ablation_dimensions(self):
        sample = {
            "caption": "walk forward",
            "motion": np.zeros((4, 3), dtype=np.float32),
            "imu_acceleration": np.zeros((2, 2, 3), dtype=np.float32),
            "imu_orientation": np.zeros((4, 2, 3), dtype=np.float32),
        }
        text_only = LinearBaselineConfig(
            text_dim=8,
            include_text=True,
            include_acceleration=False,
            include_orientation=False,
        )
        imu_only = LinearBaselineConfig(
            text_dim=8,
            include_text=False,
            include_acceleration=True,
            include_orientation=True,
        )
        self.assertEqual(build_frame_features(sample, text_only).shape[1], 1 + 3 + 8)
        self.assertEqual(build_frame_features(sample, imu_only).shape[1], 1 + 3 + 6 + 6)

    def test_save_and_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        config = LinearBaselineConfig(text_dim=4, ridge_alpha=0.5)
        weights = np.ones((8, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "linear.npz"
            save_linear_baseline(
                path,
                weights,
                config,
                train_stats={"num_records": 1.0},
                eval_stats={"heldout": {"mse": 2.0}},
            )
            loaded_weights, loaded_config, metadata = load_linear_baseline(path)

        np.testing.assert_allclose(loaded_weights, weights)
        self.assertEqual(loaded_config, config)
        self.assertEqual(metadata["eval_stats"]["heldout"]["mse"], 2.0)


if __name__ == "__main__":
    unittest.main()
