import tempfile
import unittest
from pathlib import Path

import numpy as np

from itm.data.text_embeddings import load_text_embedding_cache, save_text_embedding_cache


class TextEmbeddingCacheTest(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.npz"
            save_text_embedding_cache(
                path,
                ["a", "b"],
                ["walk", "turn"],
                np.asarray([[1, 2], [3, 4]], dtype=np.float32),
                metadata={"model": "fixture"},
            )
            cache = load_text_embedding_cache(path)
            np.testing.assert_array_equal(cache.embeddings["b"], [3, 4])
            self.assertEqual(cache.captions["a"], "walk")
            self.assertEqual(cache.metadata["model"], "fixture")

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_text_embedding_cache(
                    Path(directory) / "text.npz",
                    ["a", "a"],
                    ["walk", "turn"],
                    np.zeros((2, 3), dtype=np.float32),
                    metadata={},
                )


if __name__ == "__main__":
    unittest.main()

