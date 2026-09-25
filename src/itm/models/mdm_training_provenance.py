"""Method checkpoint input identity, independent of research data admission.

Resume remains a fresh-optimizer warm start. Matching provenance does not make
an unadmitted dataset suitable for research training.
"""
import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_training_input_provenance(dataset, normalization, *, artifacts,
                                    pipeline_version, contract_version):
    """Hash selected inputs plus manifests, backbone, options and normalization.

    Paths are included deliberately: relocating a run requires an explicit
    migration, rather than silently inheriting the old run's normalization.
    """
    hashes = {}

    def artifact(path):
        path = Path(path)
        path = (path if path.is_absolute() else dataset.workspace_root / path).resolve()
        key = str(path)
        if key not in hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[key] = digest.hexdigest()
        return {"path": key, "sha256": hashes[key]}

    selected = {}
    configurations = []
    for sample, record, name, slots in dataset.items:
        mid = str(sample["motion_id"])
        if mid not in selected:
            source = dataset.motion_records[mid]
            selected[mid] = {
                "motion_id": mid,
                "motion": artifact(source["joint_vec_path"]),
                "text": artifact(source["text_path"]),
                "imu": artifact(record["imu_path"]),
            }
        configurations.append({"motion_id": mid, "sensor_config": name, "slots": list(slots)})
    payload = {
        "schema": "itm-training-input-provenance-v1",
        "input_pipeline_version": pipeline_version,
        "contract_version": contract_version,
        "artifacts": {name: artifact(path) for name, path in sorted(artifacts.items()) if path is not None},
        "selected_inputs": list(selected.values()),
        "item_order": configurations,
        "imu_normalization_sha256": _digest(normalization),
    }
    return {"payload": payload, "sha256": _digest(payload)}


def validate_training_resume(checkpoint, provenance, normalization):
    """Reject unbound legacy state or changed inputs before model/device setup."""
    previous = checkpoint.get("training_input_provenance")
    if previous is None:
        raise ValueError("Resume checkpoint lacks training input provenance; no silent legacy migration. "
                         "It remains usable for sampling; use a new run without --resume for new data.")
    if previous != provenance:
        raise ValueError("Resume training input provenance differs (data, pipeline, backbone or normalization); "
                         "use a new run without --resume. No silent migration.")
    if _digest(checkpoint.get("imu_normalization")) != _digest(normalization):
        raise ValueError("Resume IMU normalization differs from the verified current training inputs")
