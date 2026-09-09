"""Process-local SMPL model reuse for demo fitting."""

import json
from pathlib import Path
import time

import numpy as np

from itm.data.smpl_fitting import fit_humanml_joints_to_smpl, load_neutral_smpl

_models = {}


def fit_mesh(motion_path, model_path, output_path, device):
    import torch

    torch.set_num_threads(4)
    start = time.perf_counter()
    motion = np.load(motion_path, allow_pickle=False)
    key = (str(Path(model_path).resolve()), Path(model_path).stat().st_mtime_ns, device)
    if key not in _models:
        _models.clear()
        _models[key] = load_neutral_smpl(model_path, batch_size=1, device=device)
    model = _models[key]
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    fit_start = time.perf_counter()
    fit = fit_humanml_joints_to_smpl(motion, model, iterations=200)
    fit_seconds = time.perf_counter() - fit_start
    if not np.isfinite(fit.vertices).all():
        raise ValueError('SMPL fit produced non-finite vertices')
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    fit.vertices.astype('<f4').tofile(output / 'vertices.bin')
    (output / 'faces.json').write_text(json.dumps(model.faces.reshape(-1).tolist()))
    metadata = {
        'frames': len(motion), 'vertices': int(fit.vertices.shape[1]),
        'joint_error_m': float(np.linalg.norm(fit.joints[:, :22] - motion, axis=-1).mean()),
        'fit_loss': fit.final_loss, 'representation': 'fitted_neutral_smpl',
        'iterations': 200, 'dtype': 'float32_le', 'scale': fit.scale,
        'device': device, 'fit_seconds': fit_seconds,
        'total_seconds': time.perf_counter() - start,
        'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith('cuda') else None,
    }
    pending = output / 'metadata.tmp'
    pending.write_text(json.dumps(metadata))
    pending.replace(output / 'metadata.json')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return metadata
