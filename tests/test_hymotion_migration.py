import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "sample_hymotion_text.py"
SPEC = importlib.util.spec_from_file_location("sample_hymotion_text", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checkpoint_hash_is_stable(tmp_path):
    checkpoint = tmp_path / "small.ckpt"
    checkpoint.write_bytes(b"itm")
    assert MODULE.sha256(checkpoint) == "e74cd1e4ea601f45dbb8caa21b7ddd2a91ff3be1116b473b7da8dd37c23f989c"


def test_rot6d_and_global_rotation_are_valid():
    identity6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    local = MODULE.rot6d_to_matrix(np.tile(identity6d, (2, 22, 1)))
    global_rotation = MODULE.local_to_global(local)
    expected = np.broadcast_to(np.eye(3, dtype=np.float32), global_rotation.shape)
    np.testing.assert_allclose(global_rotation, expected, atol=1e-6)
