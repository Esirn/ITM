import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "audit_standard_imu_cache",
    Path(__file__).resolve().parents[1] / "scripts/audit_standard_imu_cache.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def cache_record(tmp_path):
    path = tmp_path / "motion.npz"
    source = tmp_path / "source.npz"
    source.touch()
    arrays = {
        "motion_id": np.array("motion"), "fps": np.array(30),
        "acceleration": np.zeros((3, 6, 3)),
        "orientation": np.broadcast_to(np.eye(3), (3, 6, 3, 3)).copy(),
        "sensor_mask": np.ones(6, dtype=bool),
        "local_rotation": np.broadcast_to(np.eye(3), (3, 24, 3, 3)).copy(),
        "joints": np.zeros((3, 24, 3)), "translation": np.zeros((3, 3)),
        "shape": np.zeros(10),
    }
    record = {
        "motion_id": "motion", "imu_path": str(path), "source_path": str(source),
        "schema": "standard_imu_v1", "fps": 30, "num_frames": 3,
    }
    manifest = tmp_path / "cache.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    return path, arrays, manifest


def test_cache_audit_valid_and_coverage(cache_record):
    path, arrays, manifest = cache_record
    np.savez(path, **arrays)
    report = audit.audit_split(manifest, {"motion"}, 30)
    assert report["records"] == 1
    assert report["max_rotation_matrix_error"] == 0
    assert report["within_hymotion_20_360_frames"] == 0
    with pytest.raises(ValueError, match="coverage"):
        audit.audit_split(manifest, {"motion", "missing"}, 30)


@pytest.mark.parametrize("invalid", ["nan", "reflection", "shape"])
def test_cache_audit_rejects_invalid_arrays(cache_record, invalid):
    path, arrays, manifest = cache_record
    if invalid == "nan":
        arrays["acceleration"][0, 0, 0] = np.nan
    elif invalid == "reflection":
        arrays["orientation"][0, 0, 0, 0] = -1
    else:
        arrays["joints"] = arrays["joints"][:2]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="invalid"):
        audit.audit_split(manifest, {"motion"}, 30)
