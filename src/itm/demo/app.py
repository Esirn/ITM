"""FastAPI application for interactive ITM condition comparisons."""

from __future__ import annotations

import json
import hashlib
import multiprocessing
import shutil
import traceback
import time
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import random
import subprocess
import sys
from threading import Lock
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from itm.data.dataset import read_caption_records
from itm.data.manifest import read_jsonl


ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "src/itm/demo/static"
RUN_ROOT = ROOT / "outputs/demo_runs"
EXPERIMENT_ROOTS = {
    "stage1": ROOT / "outputs/mdm_control/experiments",
    "stage2": ROOT / "outputs/mdm_control/experiments_stage2",
    "stage2b": ROOT / "outputs/mdm_control/experiments_stage2b_balanced_b",
    "stage3": ROOT / "outputs/mdm_control/experiments_stage3_upper_body",
    "stage4": ROOT / "outputs/mdm_control/experiments_stage4_text_anchor",
}
CONTROL_CHECKPOINT = ROOT / "outputs/mdm_control/stage1_full_pilot_v2.pt"
MDM_ASSET_DIR = ROOT / "outputs/mdm/checkpoints_extracted/humanml_trans_enc_512"
MDM_ARGS = MDM_ASSET_DIR / "args.json"
MDM_CHECKPOINT = MDM_ASSET_DIR / "model000475000.pt"


def _resolve_mdm_root(candidates=None):
    configured = os.environ.get("ITM_MDM_ROOT")
    paths = list(candidates or (
        Path(configured) if configured else None,
        Path("/home/a200/0relatedworks/motion-diffusion-model"),
        Path("/home/a200/mount/a40/relatedworks/mdm/motion-diffusion-model"),
        ROOT / "outputs/mdm/runtime_root",
    ))
    paths = [path for path in paths if path is not None]
    required = Path("body_models/smpl/SMPL_NEUTRAL.pkl")
    return next((path for path in paths if (path / required).is_file()), paths[0])


MDM_ROOT = _resolve_mdm_root()
MANIFESTS = {
    split: ROOT / f"outputs/manifests_full/{split}.jsonl"
    for split in ("train", "val", "test")
}
IMU_MANIFESTS = {
    split: ROOT / f"outputs/manifests_full/{split}_standard_imu.jsonl"
    for split in ("train", "val", "test")
}
SENSOR_SLOTS = {"head": 4, "wrists": 0}
GENERATION_LOCK = Lock()
MESH_LOCK = Lock()
MESH_ROOT = ROOT / 'outputs/demo_meshes'
MESH_ACTIVE = set()
MESH_DEVICES = set()
MESH_POOLS = {}
MESH_GPU_FINISHED = {}


class MeshRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=500)
    root: str | None = None
    panel_index: int = Field(ge=0)
    device: Literal['auto', 'gpu_all', 'cpu', 'cuda:0', 'cuda:1'] = 'auto'


class MeshClearRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=500)
    root: str | None = None


class GenerateRequest(BaseModel):
    mode: Literal["four_way", "matrix"] = "four_way"
    split: Literal["train", "val", "test"] = "test"
    sample_a: str | None = None
    sample_b: str | None = None
    sensor_config: Literal["head", "wrists"] = "head"
    seed: int = 1234
    text_scale: float = Field(2.5, ge=0, le=10)
    imu_scale: float = Field(1.0, ge=0, le=10)
    device: str = "cuda:1"


app = FastAPI(title="ITM Condition Lab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    return {
        "checkpoint": str(CONTROL_CHECKPOINT),
        "mdm_checkpoint": str(MDM_CHECKPOINT),
        "default_device": "cuda:1",
        "splits": ["test", "val", "train"],
        "sensor_configs": ["head", "wrists"],
        "experiment_roots": [
            {"key": key, "path": _display_path(path)}
            for key, path in EXPERIMENT_ROOTS.items()
        ],
    }


@app.get("/api/samples")
def samples(
    split: Literal["train", "val", "test"] = "test",
    query: str = Query("", max_length=100),
    limit: int = Query(200, ge=1, le=2000),
):
    values = _sample_catalog(split)
    if query:
        needle = query.lower()
        values = [row for row in values if needle in row["motion_id"].lower() or needle in row["caption"].lower()]
    return values[:limit]


@app.get("/api/runs")
def list_runs(limit: int = Query(50, ge=1, le=200)):
    rows = []
    if not RUN_ROOT.exists():
        return rows
    for directory in RUN_ROOT.iterdir():
        request_path = directory / "request.json"
        result_path = directory / "result.json"
        if not directory.is_dir() or not request_path.exists() or not result_path.exists():
            continue
        try:
            request = json.loads(request_path.read_text())
            rows.append({
                "run_id": directory.name,
                "mode": request.get("mode"),
                "split": request.get("split"),
                "sample_a": request.get("sample_a"),
                "sample_b": request.get("sample_b"),
                "sensor_config": request.get("sensor_config"),
                "updated_at": directory.stat().st_mtime,
            })
        except (OSError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda row: row["updated_at"], reverse=True)
    return rows[:limit]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    if not run_id.replace("-", "").isalnum():
        raise HTTPException(400, "Invalid run id")
    path = RUN_ROOT / run_id / "result.json"
    if not path.exists():
        raise HTTPException(404, "Run not found")
    return json.loads(path.read_text())


@app.get("/api/experiments")
def list_experiments(
    root: str = Query("stage1", max_length=40),
    limit: int = Query(100, ge=1, le=500),
):
    root_key, experiment_root = _experiment_root(root)
    rows = []
    if not experiment_root.exists():
        return rows
    for result_path in experiment_root.glob("**/result.json"):
        run_dir = result_path.parent
        request_path = run_dir / "request.json"
        metrics_path = run_dir / "metrics.json"
        try:
            request = json.loads(request_path.read_text()) if request_path.exists() else {}
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            relative = run_dir.relative_to(experiment_root).as_posix()
            aggregate = metrics.get("aggregate", {}).get("all", {})
            rows.append({
                "run_id": relative,
                "root_key": root_key,
                "root_path": _display_path(experiment_root),
                "mode": request.get("experiment", request.get("mode", "experiment")),
                "split": request.get("split"),
                "sample_a": request.get("sample_a"),
                "sample_b": request.get("sample_b"),
                "sensor_config": request.get("sensor_config"),
                "active_sensor_error": aggregate.get("active_sensor_trajectory_error_m"),
                "jerk_ratio": aggregate.get("jerk_ratio"),
                "updated_at": run_dir.stat().st_mtime,
            })
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    rows.sort(key=lambda row: row["updated_at"], reverse=True)
    return rows[:limit]


@app.get("/api/experiments/{run_path:path}")
def get_experiment(run_path: str, root: str = Query("stage1", max_length=40)):
    root_key, experiment_root = _experiment_root(root)
    relative = _safe_relative_path(run_path)
    path = experiment_root / relative / "result.json"
    if not path.exists():
        raise HTTPException(404, "Experiment result not found")
    result = json.loads(path.read_text())
    result["source"] = {
        "kind": "experiment",
        "root_key": root_key,
        "root_path": _display_path(experiment_root),
        "relative_path": relative.as_posix(),
        "result_path": _display_path(path),
    }
    return result


@app.post("/api/generate")
def generate(request: GenerateRequest):
    if not CONTROL_CHECKPOINT.exists():
        raise HTTPException(503, f"Missing checkpoint: {CONTROL_CHECKPOINT}")
    smpl_model = MDM_ROOT / "body_models/smpl/SMPL_NEUTRAL.pkl"
    if not smpl_model.exists():
        raise HTTPException(503, f"MDM root is missing SMPL assets: {MDM_ROOT}")
    catalog = _sample_catalog(request.split)
    by_id = {row["motion_id"]: row for row in catalog}
    rng = random.Random(request.seed)
    sample_a = request.sample_a or rng.choice(catalog)["motion_id"]
    if sample_a not in by_id:
        raise HTTPException(400, f"Sample A {sample_a} is unavailable in {request.split}")
    sample_b = request.sample_b
    if request.mode == "matrix":
        choices = [row["motion_id"] for row in catalog if row["motion_id"] != sample_a]
        sample_b = sample_b or rng.choice(choices)
        if sample_b not in by_id or sample_b == sample_a:
            raise HTTPException(400, "Matrix mode requires a distinct available sample B")

    run_id = uuid4().hex[:12]
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = _build_spec(request, by_id[sample_a], by_id.get(sample_b) if sample_b else None)
    request_payload = request.model_dump() | {"sample_a": sample_a, "sample_b": sample_b, "run_id": run_id}
    (run_dir / "request.json").write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts/sample_mdm_imu_control.py"),
        "--control-checkpoint", str(CONTROL_CHECKPOINT),
        "--mdm-args", str(MDM_ARGS),
        "--mdm-checkpoint", str(MDM_CHECKPOINT),
        "--mdm-root", str(MDM_ROOT),
        "--standard-imu-manifest", str(IMU_MANIFESTS[request.split]),
        "--spec", str(run_dir / "spec.json"),
        "--output", str(run_dir / "results.npz"),
        "--seed", str(request.seed),
        "--text-scale", str(request.text_scale),
        "--imu-scale", str(request.imu_scale),
        "--device", request.device,
    ]
    with MESH_LOCK:
        if request.device in MESH_DEVICES:
            raise HTTPException(409, 'Selected GPU is fitting SMPL; retry after fitting or choose another device')
        if not GENERATION_LOCK.acquire(blocking=False):
            raise HTTPException(409, "Another generation is already using the model")
    try:
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    finally:
        GENERATION_LOCK.release()
    (run_dir / "generation.log").write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if process.returncode:
        raise HTTPException(500, f"Generation failed; inspect {run_dir / 'generation.log'}")
    result = _serialize_result(run_id, request_payload, by_id, run_dir / "results.npz")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


@app.post('/api/mesh')
def prepare_mesh(request: MeshRequest):
    result = (get_experiment(request.run_id, request.root)
              if request.root is not None else get_run(request.run_id))
    if request.panel_index >= len(result['panels']):
        raise HTTPException(400, 'Unknown panel')
    motion, model, cache_id = _mesh_identity(result['panels'][request.panel_index])
    directory = MESH_ROOT / cache_id
    metadata = directory / 'metadata.json'
    with MESH_LOCK:
        if metadata.is_file():
            return _mesh_response(metadata, cache_id)
        if cache_id in MESH_ACTIVE:
            raise HTTPException(409, 'This SMPL fit is already running')
        device = _mesh_device(request.device)
        if device is None:
            raise HTTPException(409, 'Fitting device busy; waiting for an available slot')
        MESH_ACTIVE.add(cache_id)
        MESH_DEVICES.add(device)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / 'motion.npy', motion)
        _run_mesh_job(directory, model, device)
    except Exception as exc:
        (directory / 'fit.log').write_text(traceback.format_exc())
        raise HTTPException(500, f'SMPL fitting failed on {device}; inspect {directory / "fit.log"}') from exc
    finally:
        with MESH_LOCK:
            MESH_ACTIVE.discard(cache_id)
            MESH_DEVICES.discard(device)
            if device.startswith('cuda:'):
                MESH_GPU_FINISHED[device] = time.monotonic()
    return _mesh_response(metadata, cache_id)


def _mesh_identity(panel):
    motion = np.asarray(panel['motion'], dtype='<f4')
    if motion.ndim != 3 or motion.shape[1:] != (22, 3) or not 3 <= len(motion) <= 196 or not np.isfinite(motion).all():
        raise HTTPException(400, 'SMPL preview requires 3–196 frames of finite 22-joint motion')
    model = MDM_ROOT / 'body_models/smpl/SMPL_NEUTRAL.pkl'
    if not model.is_file():
        raise HTTPException(503, 'SMPL model unavailable')
    identity = f'neutral-preview-v1-200:{model.resolve()}:{model.stat().st_mtime_ns}'
    cache_id = hashlib.sha256(identity.encode() + motion.tobytes()).hexdigest()
    return motion, model, cache_id


def _gpu_load():
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], capture_output=True, text=True, timeout=3, check=True)
        return [tuple(int(value.strip()) for value in line.split(','))
                for line in result.stdout.splitlines()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


@app.get('/api/mesh-devices')
def mesh_devices():
    return {'gpus': [index for index, _, _ in _gpu_load()]}


def _mesh_device(requested):
    # Called under MESH_LOCK so concurrent HTTP requests cannot reserve a GPU twice.
    if requested != 'cpu' and not GENERATION_LOCK.locked():
        for index, free_mib, utilization in sorted(_gpu_load(), key=lambda row: -row[1]):
            device = f'cuda:{index}'
            if requested not in ('auto', 'gpu_all', device) or device in MESH_DEVICES:
                continue
            if free_mib >= 6144 and utilization < 20:
                return device
    if requested == 'auto' and not GENERATION_LOCK.locked():
        # GPU utilization is sampled over a window; let our previous fit's load decay.
        if any(device.startswith('cuda:') for device in MESH_DEVICES) or any(
                time.monotonic() - finished < 3 for finished in MESH_GPU_FINISHED.values()):
            return None
    if requested in ('auto', 'cpu') and 'cpu' not in MESH_DEVICES:
        return 'cpu'
    return None


def _run_mesh_job(directory, model, device):
    from itm.demo.mesh_worker import fit_mesh
    with MESH_LOCK:
        if device not in MESH_POOLS:
            MESH_POOLS[device] = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        pool = MESH_POOLS[device]
    try:
        result = pool.submit(fit_mesh, str(directory / 'motion.npy'), str(model), str(directory), device).result()
        (directory / 'fit.log').write_text(json.dumps(result, indent=2))
    except Exception:
        with MESH_LOCK:
            MESH_POOLS.pop(device, None)
        pool.shutdown(wait=True, cancel_futures=True)
        raise


def _mesh_response(metadata, cache_id):
    return json.loads(metadata.read_text()) | {
        'vertices_url': f'/api/mesh-cache/{cache_id}/vertices.bin',
        'faces_url': f'/api/mesh-cache/{cache_id}/faces.json',
    }


@app.post('/api/mesh/clear')
def clear_mesh(request: MeshClearRequest):
    result = (get_experiment(request.run_id, request.root)
              if request.root is not None else get_run(request.run_id))
    keys = {_mesh_identity(panel)[2] for panel in result['panels']}
    removed = 0
    with MESH_LOCK:
        if keys & MESH_ACTIVE:
            raise HTTPException(409, 'This result is still being fitted; retry after fitting finishes')
        for key in keys:
            directory = MESH_ROOT / key
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory)
                removed += 1
    return {'removed': removed, 'shared_motion_cache': True}


@app.on_event('shutdown')
def close_mesh_workers():
    for pool in MESH_POOLS.values():
        pool.shutdown(wait=True, cancel_futures=True)
    MESH_POOLS.clear()


@app.get('/api/mesh-cache/{cache_id}/{filename}')
def mesh_cache(cache_id: str, filename: str):
    if len(cache_id) != 64 or any(char not in '0123456789abcdef' for char in cache_id) or filename not in ('vertices.bin', 'faces.json'):
        raise HTTPException(400, 'Invalid mesh asset')
    path = MESH_ROOT / cache_id / filename
    if not path.is_file() or not (path.parent / 'metadata.json').is_file():
        raise HTTPException(404, 'Mesh unavailable')
    return FileResponse(path)


def _sample_catalog(split):
    available = {str(row["motion_id"]): row for row in read_jsonl(IMU_MANIFESTS[split])}
    result = []
    for row in read_jsonl(MANIFESTS[split]):
        motion_id = str(row["motion_id"])
        if motion_id not in available:
            continue
        captions = read_caption_records(row["text_path"])
        if not captions:
            continue
        result.append({
            "motion_id": motion_id,
            "caption": captions[0].caption,
            "captions": [value.caption for value in captions],
            "frames": int(available[motion_id]["num_frames"]),
        })
    return sorted(result, key=lambda row: row["motion_id"])


def _build_spec(request, sample_a, sample_b):
    a_id, a_text = sample_a["motion_id"], sample_a["caption"]
    common = {"sensor_config": request.sensor_config, "split": request.split}
    if request.mode == "four_way":
        return [
            {**common, "motion_id": a_id, "text": a_text, "label": "Text A only", "imu_scale": 0.0, "text_source": "A", "imu_source": None},
            {**common, "motion_id": a_id, "text": "", "label": "IMU A only", "text_scale": 0.0, "text_source": None, "imu_source": "A"},
            {**common, "motion_id": a_id, "text": a_text, "label": "Text A + IMU A", "text_source": "A", "imu_source": "A"},
        ]
    b_id, b_text = sample_b["motion_id"], sample_b["caption"]
    return [
        {**common, "motion_id": a_id, "text": "", "label": "None + IMU A", "text_scale": 0.0, "text_source": None, "imu_source": "A"},
        {**common, "motion_id": b_id, "text": "", "label": "None + IMU B", "text_scale": 0.0, "text_source": None, "imu_source": "B"},
        {**common, "motion_id": a_id, "text": a_text, "label": "Text A + None", "imu_scale": 0.0, "text_source": "A", "imu_source": None},
        {**common, "motion_id": a_id, "text": a_text, "label": "Text A + IMU A", "text_source": "A", "imu_source": "A"},
        {**common, "motion_id": b_id, "text": a_text, "label": "Text A + IMU B", "text_source": "A", "imu_source": "B"},
        {**common, "motion_id": a_id, "text": b_text, "label": "Text B + None", "imu_scale": 0.0, "text_source": "B", "imu_source": None},
        {**common, "motion_id": a_id, "text": b_text, "label": "Text B + IMU A", "text_source": "B", "imu_source": "A"},
        {**common, "motion_id": b_id, "text": b_text, "label": "Text B + IMU B", "text_source": "B", "imu_source": "B"},
    ]


def _serialize_result(run_id, request, catalog, result_path):
    with np.load(result_path) as data:
        motion = data["motion"].astype(np.float32)
        gt = data["gt"].astype(np.float32)
        acceleration = data["acceleration"].astype(np.float32)
        orientation = data["orientation"].astype(np.float32)
        metadata = json.loads(str(data["metadata"]))
    panels = []
    if request["mode"] == "four_way":
        panels.append({"label": "Ground truth A", "kind": "ground_truth", "text_source": "A", "imu_source": "A", "caption": catalog[request["sample_a"]]["caption"], "motion": gt[0].tolist()})
    else:
        panels.extend([
            {"label": "Ground truth A", "kind": "ground_truth", "text_source": "A", "imu_source": "A", "caption": catalog[request["sample_a"]]["caption"], "motion": gt[0].tolist()},
            {"label": "Ground truth B", "kind": "ground_truth", "text_source": "B", "imu_source": "B", "caption": catalog[request["sample_b"]]["caption"], "motion": gt[1].tolist()},
        ])
    for index, case in enumerate(metadata["cases"]):
        panels.append({
            "label": case["label"],
            "kind": "generated",
            "text_source": case.get("text_source"),
            "imu_source": case.get("imu_source"),
            "caption": case["text"],
            "motion": motion[index].tolist(),
        })
    slot = SENSOR_SLOTS[request["sensor_config"]]
    imu = {"A": {"acceleration": acceleration[0, :, slot].tolist(), "orientation": orientation[0, :, slot, :, 0].tolist()}}
    if request["mode"] == "matrix":
        imu["B"] = {"acceleration": acceleration[1, :, slot].tolist(), "orientation": orientation[1, :, slot, :, 0].tolist()}
    return {
        "run_id": run_id,
        "request": request,
        "samples": {key: catalog[value] for key, value in (("A", request["sample_a"]), ("B", request.get("sample_b"))) if value},
        "fps": 20,
        "frame_count": int(motion.shape[1]),
        "panels": panels,
        "imu": imu,
    }


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise HTTPException(400, "Invalid experiment path")
    return path


def _experiment_root(value: str) -> tuple[str, Path]:
    if value not in EXPERIMENT_ROOTS:
        raise HTTPException(400, f"Unknown experiment root: {value}")
    return value, EXPERIMENT_ROOTS[value]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)
