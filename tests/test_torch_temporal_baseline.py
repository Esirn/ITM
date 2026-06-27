import importlib.util
import unittest

import numpy as np

from itm.models.torch_temporal_baseline import (
    TemporalBaselineConfig,
    build_sequence_item,
    collate_sequence_items,
    make_temporal_model,
    masked_mse,
)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch is not installed")
class TorchTemporalBaselineTest(unittest.TestCase):
    def _sample(self, motion_id, length):
        return {
            "motion_id": motion_id,
            "motion": np.ones((length, 5), dtype=np.float32),
            "imu_acceleration": np.zeros((max(length - 2, 0), 2, 3), dtype=np.float32),
            "imu_orientation": np.zeros((length, 2, 3), dtype=np.float32),
        }

    def test_padding_forward_and_masked_loss(self):
        import torch

        config = TemporalBaselineConfig(model_dim=16, num_layers=1, num_heads=4)
        embeddings = {
            "a": np.ones(8, dtype=np.float32),
            "b": np.zeros(8, dtype=np.float32),
        }
        items = [
            build_sequence_item(self._sample("a", 4), embeddings, config),
            build_sequence_item(self._sample("b", 6), embeddings, config),
        ]
        batch = collate_sequence_items(items)
        self.assertEqual(batch["sequence"].shape[:2], (2, 6))
        self.assertEqual(batch["mask"].sum(), 10)
        model = make_temporal_model(
            batch["sequence"].shape[-1], 8, batch["target"].shape[-1], config
        )
        prediction = model(
            torch.from_numpy(batch["sequence"]),
            torch.from_numpy(batch["text"]),
            torch.from_numpy(batch["mask"]),
        )
        loss = masked_mse(
            prediction, torch.from_numpy(batch["target"]), torch.from_numpy(batch["mask"])
        )
        self.assertEqual(tuple(prediction.shape), (2, 6, 5))
        self.assertTrue(torch.isfinite(loss))

    def test_sensor_slot_selection_changes_input_width(self):
        sample = self._sample("a", 5)
        sample["imu_acceleration"] = np.zeros((3, 3, 3), dtype=np.float32)
        sample["imu_orientation"] = np.zeros((5, 3, 3), dtype=np.float32)
        embeddings = {"a": np.ones(8, dtype=np.float32)}
        all_item = build_sequence_item(sample, embeddings, TemporalBaselineConfig())
        sparse_item = build_sequence_item(
            sample,
            embeddings,
            TemporalBaselineConfig(sensor_slots=(0, 2)),
        )
        self.assertEqual(all_item["sequence"].shape[1], 22)
        self.assertEqual(sparse_item["sequence"].shape[1], 16)

    def test_rejects_out_of_range_sensor_slot(self):
        with self.assertRaises(IndexError):
            build_sequence_item(
                self._sample("a", 5),
                {"a": np.ones(8, dtype=np.float32)},
                TemporalBaselineConfig(sensor_slots=(2,)),
            )


if __name__ == "__main__":
    unittest.main()
