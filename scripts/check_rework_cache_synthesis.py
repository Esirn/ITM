#!/usr/bin/env python3
"""CPU-only three-clip re-synthesis audit; never writes replacement caches."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from audit_itm_rework import artifact, dump
from build_standard_imu_cache import _enable_legacy_chumpy_compatibility, _process_source
from itm.data.amass_bridge import load_humanml_amass_index
from itm.data.manifest import read_jsonl
from itm.data.standard_imu import SENSOR_VERTEX_INDICES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--imuposer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error("use a new output path")
    import torch
    torch.set_num_threads(2)
    _enable_legacy_chumpy_compatibility()
    sys.path.insert(0, str(args.imuposer_root / "src"))
    from imuposer import math as imu_math
    from imuposer.smpl.parametricModel import ParametricModel
    model_path = args.imuposer_root / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    model = ParametricModel(model_path, device=torch.device("cpu"))
    mapping = load_humanml_amass_index(args.index, args.amass_root)
    coordinate = torch.tensor([[[1.,0,0],[0,0,1],[0,-1,0]]])
    weights = model._skinning_weights[list(SENSOR_VERTEX_INDICES)].cpu().numpy()
    mounts = [{"vertex": vertex, "top_skinning_segments": np.argsort(w)[-3:][::-1].tolist(),
               "top_skinning_weights": np.sort(w)[-3:][::-1].tolist()} for vertex,w in zip(SENSOR_VERTEX_INDICES, weights)]
    checks = []
    for split in ("train", "val", "test"):
        record = next(read_jsonl(f"outputs/manifests_full/{split}_standard_imu.jsonl"))
        source = mapping[record["motion_id"]]
        computed = _process_source(source, model, imu_math, coordinate, float(record["fps"]), "cpu")
        comparison = {}
        with np.load(record["imu_path"], allow_pickle=False) as data:
            for key in ("acceleration", "orientation", "local_rotation", "joints", "translation", "shape"):
                a,b = data[key], computed[key]
                comparison[key] = {"cached_shape": list(a.shape), "computed_shape": list(b.shape),
                    "max_abs_difference": float(np.max(np.abs(a-b))) if a.shape == b.shape else None,
                    "allclose_atol_1e-4_rtol_1e-4": bool(a.shape == b.shape and np.allclose(a,b,atol=1e-4,rtol=1e-4))}
        checks.append({"split": split, "motion_id": record["motion_id"], "source": artifact(source.source_path),
                       "cache": artifact(record["imu_path"]), "comparison": comparison})
        print(json.dumps({"split": split, "motion_id": record["motion_id"], "comparison": comparison}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, {"operator": artifact("src/itm/data/standard_imu.py"),
        "builder": artifact("scripts/build_standard_imu_cache.py"), "model": artifact(model_path),
        "budget": {"clips": 3, "device": "cpu", "torch_threads": 2},
        "mount_skinning": mounts, "checks": checks,
        "limitation": "three source/cache comparisons only; not HumanML representation round-trip certification"})


if __name__ == "__main__":
    main()
