#!/usr/bin/env python
"""Explain fixed cache acceleration differences without relaxing the recorded tolerance."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

from audit_caption_intervals import artifact, write_json
from build_standard_imu_cache import _enable_legacy_chumpy_compatibility
from check_prefix_cache_rebuild import overlap_indices
from itm.data.standard_imu import SENSOR_VERTEX_INDICES, second_difference, moving_average


def differentiate64(positions, fps=30.):
    positions = np.asarray(positions, dtype=np.float64)
    raw = np.zeros_like(positions)
    raw[1:-1] = (positions[:-2] + positions[2:] - 2 * positions[1:-1]) * fps**2
    padded = np.pad(raw, ((2, 2), (0, 0), (0, 0)), mode="edge")
    return np.stack([padded[i:i + 5].mean(0) for i in range(len(raw))])


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(values))), "rms": float(np.sqrt(np.mean(values**2)))}


def compare(a, b):
    return {**summary(np.asarray(a, dtype=np.float64) - b),
            "allclose_atol_1e-4_rtol_1e-4": bool(np.allclose(a, b, atol=1e-4, rtol=1e-4))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    report_path = Path("docs/itm_rework/protocol/prefix_rebuild_20260925_v1/report.json")
    cases = json.loads(report_path.read_text())["checks"]
    if [r["motion_id"] for r in cases] != ["000123", "000015", "000506", "004728", "000027"]:
        raise ValueError("The fixed five cases must not change")
    budget = {"clips": 5, "CPU_FK_calls": 20, "threads": 2, "max_frames_per_call": 300,
              "dtypes": ["float32", "float64"], "gpu_jobs": 0, "training_updates": 0,
              "tolerance_unchanged": {"atol": 1e-4, "rtol": 1e-4}}
    write_json(out / "budget.json", budget)
    import torch
    torch.set_num_threads(2)
    _enable_legacy_chumpy_compatibility()
    imuposer = Path("/home/a200/0relatedworks/IMUPoser")
    sys.path.insert(0, str(imuposer / "src"))
    from imuposer.smpl.parametricModel import ParametricModel
    model_path = imuposer / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    model32 = ParametricModel(model_path, device=torch.device("cpu"))
    model64 = copy.copy(model32)
    for name, value in vars(model32).items():
        if torch.is_tensor(value):
            setattr(model64, name, value.double())
    results = []
    for case in cases:
        stored, positions, replay32, arithmetic64, replay64 = {}, {}, {}, {}, {}
        for name, field in (("old", "old_cache"), ("new", "diagnostic_cache")):
            source = case[field]
            if artifact(source["path"])["sha256"] != source["sha256"]:
                raise ValueError("Saved cache identity changed")
            with np.load(source["path"], allow_pickle=False) as data:
                stored[name] = {key: data[key].copy() for key in ("local_rotation", "translation", "shape", "acceleration", "joints", "orientation")}
            data = stored[name]
            if len(data["local_rotation"]) > 300:
                raise ValueError("Frame budget exceeded")
            positions[name] = {}
            for dtype, model in ((torch.float32, model32), (torch.float64, model64)):
                with torch.inference_mode():
                    rotation, joints, vertices = model.forward_kinematics(
                        torch.tensor(data["local_rotation"], dtype=dtype), torch.tensor(data["shape"], dtype=dtype),
                        torch.tensor(data["translation"], dtype=dtype), calc_mesh=True)
                key = "f32" if dtype == torch.float32 else "f64"
                positions[name][key] = vertices[:, SENSOR_VERTEX_INDICES].numpy().copy()
                if dtype == torch.float32:
                    replay32[name] = moving_average(second_difference(positions[name][key], fps=30.), 5)
                    arithmetic64[name] = differentiate64(positions[name][key])
                else:
                    replay64[name] = differentiate64(positions[name][key])
                del rotation, joints, vertices
        old, new, interior = overlap_indices(len(stored["old"]["acceleration"]), len(stored["new"]["acceleration"]), int(case["prefix_seconds"] * 30))
        oi, ni = old[interior], new[interior]
        saved_delta = stored["new"]["acceleration"][ni].astype(np.float64) - stored["old"]["acceleration"][oi]
        diff32 = replay32["new"][ni].astype(np.float64) - replay32["old"][oi]
        diff_arithmetic64 = arithmetic64["new"][ni] - arithmetic64["old"][oi]
        diff64 = replay64["new"][ni] - replay64["old"][oi]
        components = {"saved_vs_replayed_cache_gap": saved_delta - diff32,
                      "float32_difference_and_smoothing": diff32 - diff_arithmetic64,
                      "float32_FK_mesh_coordinates": diff_arithmetic64 - diff64,
                      "stored_input_difference_in_float64_FK": diff64}
        reconstruction_error = summary(saved_delta - sum(components.values()))
        arrays = {"saved_delta": saved_delta, **components}
        for name in ("old", "new"):
            arrays[f"{name}_sensor_positions_f32"] = positions[name]["f32"]
            arrays[f"{name}_sensor_positions_f64"] = positions[name]["f64"]
        path = out / f"{case['motion_id']}_components.npz"
        np.savez_compressed(path, **arrays, old_overlap_indices=oi, new_overlap_indices=ni)
        row = {"motion_id": case["motion_id"], "overlap_valid_frames": len(oi),
            "stored_local_rotation_difference": summary(stored["new"]["local_rotation"][new] - stored["old"]["local_rotation"][old]),
            "stored_translation_difference": summary(stored["new"]["translation"][new] - stored["old"]["translation"][old]),
            "stored_shape_difference": summary(stored["new"]["shape"] - stored["old"]["shape"]),
            "saved_overlap_acceleration": compare(stored["new"]["acceleration"][ni], stored["old"]["acceleration"][oi]),
            "same_float32_mesh_with_float64_difference": compare(arithmetic64["new"][ni], arithmetic64["old"][oi]),
            "float64_FK_and_difference": compare(replay64["new"][ni], replay64["old"][oi]),
            "cache_replay": {name: compare(replay32[name], stored[name]["acceleration"]) for name in ("old", "new")},
            "components": {k: summary(v) for k, v in components.items()},
            "component_sum_residual": reconstruction_error, "arrays": artifact(path)}
        results.append(row)
        print(json.dumps({"motion_id": row["motion_id"], "saved": row["saved_overlap_acceleration"],
                          "f64": row["float64_FK_and_difference"], "cache_replay": row["cache_replay"]}), flush=True)
    write_json(out / "report.json", {"kind": "fixed_acceleration_numerical_decomposition", "budget": budget,
        "cases": results, "inputs": [artifact(report_path), artifact(model_path)],
        "code": [artifact(p) for p in (Path(__file__), Path("src/itm/data/standard_imu.py"),
                 imuposer / "src/imuposer/smpl/parametricModel.py")],
        "limitations": ["Float64 FK promotes the same stored float32 model/input values; it is a numerical diagnostic, not new physical truth.",
                        "Component magnitudes can cancel and must not be interpreted as additive percentages.",
                        "Original allclose failures and tolerance remain unchanged; no caches overwritten."]})


if __name__ == "__main__":
    main()
