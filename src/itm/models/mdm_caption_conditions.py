"""Opt-in method adapter for owner-02 caption plans; not training admission."""
import hashlib
import json
from pathlib import Path

import numpy as np

from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.models.mdm_conditions import load_method_cache, sample_imu_at_times, normalize_imu

CAPTION_PIPELINE_VERSION = "itm-method-caption-seconds-v3"


def read_caption_plans(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        key = str(row['motion_id'])
        if key in result:
            raise ValueError("One explicitly selected caption per motion required")
        result[key] = row
    if not result:
        raise ValueError("Empty caption plan")
    return result


def require_caption_training_admission(plans):
    """Diagnostic override does not promote the current sidecar to training data."""
    if not plans or any(row.get('training_admissible') is not True or row.get('status') == 'blocked'
                        or row.get('world_alignment_certified') is not True
                        or row.get('semantic_alignment_certified') is not True
                        for row in plans.values()):
        raise ValueError("Caption plan is not admitted for training; input checks are diagnostic only")


def _path(value, root):
    value = Path(value)
    return value if value.is_absolute() else Path(root) / value


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_caption_input(plan, motion_record, imu_record, normalization, *, workspace_root,
                          cache_loader=load_method_cache):
    """Verify the selected caption and artifact-bound plan, then load exact slices.

    Source times retain the nominal HumanML clip origin; no SSM clock hypothesis,
    target GT yaw, body retarget or root recanonicalization is applied here.
    """
    if plan.get('status') == 'blocked':
        raise ValueError('Caption plan blocked: ' + str(plan.get('reason', 'unspecified')))
    mid = str(plan['motion_id'])
    if (str(motion_record['motion_id']) != mid or str(imu_record['motion_id']) != mid
            or motion_record['split'] != plan['split'] or imu_record['split'] != plan['split']):
        raise ValueError("Caption plan motion/split mismatch")
    artifacts = {str(_path(a['path'], workspace_root).resolve()): a['sha256'] for a in plan['artifacts']}
    paths = [_path(motion_record['text_path'], workspace_root),
             _path(motion_record['joint_vec_path'], workspace_root),
             _path(imu_record['imu_path'], workspace_root)]
    for path in paths:
        if artifacts.get(str(path.resolve())) != _sha(path):
            raise ValueError(f"Caption plan artifact hash mismatch: {path}")
    lines = [line for line in paths[0].read_text().splitlines() if line.strip()]
    index = plan['caption_index']
    if type(index) is not int or not 0 <= index < len(lines):
        raise ValueError("Explicit caption_index outside nonempty annotation lines")
    caption = parse_timed_caption(lines[index])
    if caption.text != plan['caption']:
        raise ValueError("Caption text differs from selected annotation")
    motion = np.load(paths[1], allow_pickle=False)
    if motion.ndim != 2 or motion.shape[1] != 263:
        raise ValueError("HumanML263 features required")
    cache = cache_loader(imu_record, workspace_root=workspace_root)
    checked = plan_caption_crop(caption, len(motion), cache['timestamps'])
    for key in ('version', 'annotation_kind', 'raw_tags_seconds', 'frame_slice', 'fps', 'frames',
                'clock', 'slice_convention', 'source_sample_interval_seconds', 'source_slice_bounds_seconds',
                'source_time_origin_seconds'):
        if plan.get(key) != checked[key]:
            raise ValueError(f"Caption plan differs from source annotation: {key}")
    for key in ('source_times_seconds', 'model_times_seconds'):
        saved = np.asarray(plan.get(key, []), dtype=np.float64)
        if saved.shape != checked[key].shape or not np.allclose(saved, checked[key], rtol=0, atol=1e-12):
            raise ValueError(f"Caption plan time axis differs: {key}")
    start, stop = checked['frame_slice']
    source_times, model_times = checked['source_times_seconds'], checked['model_times_seconds']
    acceleration, orientation = sample_imu_at_times(cache, source_times)
    normalization = normalization or {'acceleration_mean': np.zeros((6,3)), 'acceleration_std': np.ones((6,3))}
    return dict(motion=motion[start:stop].astype(np.float32), imu=normalize_imu(acceleration, orientation, normalization),
                acceleration=acceleration, orientation=orientation, caption=caption.text,
                source_times_seconds=source_times, model_times_seconds=model_times,
                cache_sensor_mask=cache['sensor_mask'], input_metadata={
                    'input_pipeline_version': CAPTION_PIPELINE_VERSION, 'caption_index': index,
                    'original_motion_frames': len(motion), 'effective_frames': stop-start,
                    'frame_slice': [start, stop], 'slice_convention': checked['slice_convention'],
                    'caption_interval_seconds': [caption.start, caption.end],
                    'time_interval_seconds': checked['source_sample_interval_seconds'],
                    'source_slice_bounds_seconds': checked['source_slice_bounds_seconds'],
                    'source_times_seconds': source_times.tolist(), 'model_times_seconds': model_times.tolist(),
                    'source_time_origin_seconds': checked['source_time_origin_seconds'],
                    'clock_reference': 'HumanML_nominal_clip_seconds_to_existing_cache_seconds_uncertified',
                    'alignment_status': 'uncertified_diagnostic', 'training_admissible': False,
                    'feature_recanonicalization': 'none', 'notebook_clock_correction': 'none',
                    'plan_training_admissible': plan.get('training_admissible', False),
                    'artifacts': plan['artifacts'],
                })
