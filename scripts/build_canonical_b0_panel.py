#!/usr/bin/env python
"""Plan/build twelve fixed offline synthetic B0 pairs; no training or generation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from audit_caption_intervals import artifact, write_json
from build_standard_imu_cache import _enable_legacy_chumpy_compatibility, _process_source
from itm.data.amass_bridge import HumanMLAMASSRecord
from itm.data.canonical_pair import SCHEMA, MAPPING_VERSION, canonical_transform, transform_points, encode_humanml263
from itm.data.humanml_clock import PREFIX_TRIM_SECONDS
from itm.data.manifest import read_jsonl

TRAIN = ["000010", "000123", "000015", "000506", "004728", "000027"]
VAL = ["000133", "000433", "000382", "000510", "000134", "000139"]
ORIGINS = Path("docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl")
CANDIDATES = Path("docs/itm_rework/protocol/caption_audit_20260924_v1/full_clip_candidates.jsonl")
MDM = Path("/home/a200/0relatedworks/motion-diffusion-model")
IMUPOSER = Path("/home/a200/0relatedworks/IMUPoser")
MODEL = IMUPOSER / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def plan(path):
    if path.exists():
        raise FileExistsError(path)
    origins = list(read_jsonl(ORIGINS))
    by_id = {r['motion_id']: r for r in origins}
    candidates = {r['motion_id']: r for r in read_jsonl(CANDIDATES)}
    groups = {}
    for row in origins:
        groups.setdefault(row['source_group_id'], set()).add(row['split'])
    cases = []
    for split, ids in (("train", TRAIN), ("val", VAL)):
        for mid in ids:
            c, o = candidates[mid], by_id[mid]
            if c['split'] != split or groups[o['source_group_id']] != {split}:
                raise ValueError("Source must be exclusive to selected original split")
            with np.load(o['resolved_source_path'], allow_pickle=False) as d:
                fps, frames = float(d['mocap_framerate']), len(d['poses'])
            if fps not in (60., 120.):
                raise ValueError("Only exact 60/120Hz notebook clocks admitted here")
            prefix = sum(seconds for dataset, seconds in PREFIX_TRIM_SECONDS.items() if dataset in o['source_group_id'].split('/'))
            start = int((o['start_frame_20fps'] + prefix * 20) * fps / 20)
            requested_stop = int((o['end_frame_20fps'] + prefix * 20) * fps / 20)
            last_endpoint = start + c['frame_slice'][1] * int(fps / 20)
            if last_endpoint >= frames or requested_stop > frames + int(fps / 20) - 1:
                raise ValueError('A required 20Hz endpoint is absent; no extrapolation')
            cases.append({**c, 'source': o, 'source_fps': fps, 'source_frames': frames,
                          'prefix_seconds': prefix, 'source_artifact': artifact(o['resolved_source_path']),
                          'requested_source_slice': [start, requested_stop], 'available_source_slice': [start, min(requested_stop, frames)],
                          'last_required_20hz_endpoint_source_frame': last_endpoint,
                          'terminal_policy': 'Read through available final frame only when all requested 20Hz samples including the velocity endpoint exist; record the missing sub-20Hz tail explicitly.',
                          'builder_end_frame_20fps': -1 if requested_stop > frames else o['end_frame_20fps'] + int(prefix * 20)})
    path.parent.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), Path('src/itm/data/canonical_pair.py'), Path('scripts/build_standard_imu_cache.py'),
            Path('src/itm/data/standard_imu.py'), Path('src/itm/data/amass_bridge.py'), Path('src/itm/data/humanml_clock.py'),
            MDM / 'data_loaders/humanml/common/skeleton.py', MDM / 'data_loaders/humanml/common/quaternion.py',
            MDM / 'data_loaders/humanml/utils/paramUtil.py', IMUPOSER / 'src/imuposer/smpl/parametricModel.py']
    write_json(path, {'version': MAPPING_VERSION, 'read_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'scope': 'offline_synthetic_precanonicalized_source_SMPL_body_no_uniform_retarget',
        'selection': 'Previously fixed five prefix rebuild IDs plus train 000010; original fixed six val IDs. 000008 excluded before build: nonexact 120.0000458Hz; 000004 overlength; 000006 non-full caption. No replacement of rejected cases.',
        'budget': {'cases': 12, 'CPU_FK_calls': 24, 'max_frames_per_call': 300, 'threads': 2, 'gpu_jobs': 0, 'training_updates': 0, 'model_generations': 0},
        'gates': {'source_fps': [60,120], 'max_feature_frames':196, 'RIC_roundtrip_max_abs_m': 1e-5,
                  'rotation_orthogonality_max_abs': 2e-5, 'caption': 'Full interval plus explicit visual semantic review; uncertainty excluded, no replacement'},
        'cases': cases, 'inputs': [artifact(ORIGINS), artifact(CANDIDATES), artifact(MODEL)], 'code': [artifact(p) for p in code]})


def build(plan_path, out):
    p = json.loads(plan_path.read_text())
    for item in p['inputs'] + p['code']:
        if artifact(item['path'])['sha256'] != item['sha256']:
            raise ValueError('Plan input changed: ' + item['path'])
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from itm.data.humanml_torch import recover_from_ric_torch
    torch.set_num_threads(2)
    _enable_legacy_chumpy_compatibility()
    sys.path.insert(0, str(IMUPOSER / 'src'))
    from imuposer import math as imu_math
    from imuposer.smpl.parametricModel import ParametricModel
    model = ParametricModel(MODEL, device=torch.device('cpu'))
    basis = torch.tensor([[[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]])
    rows = []
    for c in p['cases']:
        mid, source = c['motion_id'], c['source']
        if artifact(c['source_artifact']['path'])['sha256'] != c['source_artifact']['sha256'] or artifact(c['text_artifact']['path'])['sha256'] != c['text_artifact']['sha256']:
            raise ValueError('Source or caption changed')
        shift = int(c['prefix_seconds'] * 20)
        record = HumanMLAMASSRecord(mid, Path(source['resolved_source_path']), source['start_frame_20fps']+shift, c['builder_end_frame_20fps'])
        if c['frame_slice'][1] > 196 or (c['available_source_slice'][1]-c['available_source_slice'][0])/c['source_fps']*30 > 300:
            raise ValueError('Budget exceeded')
        a20 = _process_source(record, model, imu_math, basis, 20., 'cpu')
        a30 = _process_source(record, model, imu_math, basis, 30., 'cpu')
        if len(a30['joints']) > 300 or len(a20['joints'])-1 != c['frame_slice'][1]:
            raise ValueError('Planned length differs; no silent cropping')
        matrix, origin = canonical_transform(a20['joints'][:, :22])
        target = transform_points(a20['joints'][:, :22], matrix, origin)
        features = encode_humanml263(target.copy(), mdm_root=MDM)
        decoded = recover_from_ric_torch(torch.from_numpy(features)).numpy()
        error = float(np.max(np.abs(decoded-target[:-1])))
        orientation = np.matmul(matrix, a30['orientation']).astype(np.float32)
        ortho = float(np.max(np.abs(np.matmul(orientation.transpose(0,1,3,2), orientation)-np.eye(3))))
        if error > p['gates']['RIC_roundtrip_max_abs_m'] or ortho > p['gates']['rotation_orthogonality_max_abs']:
            raise ValueError('Predeclared numeric gate failed')
        dest = out / c['split'] / mid
        dest.mkdir(parents=True)
        np.save(dest / 'motion.npy', features)
        np.save(dest / 'joints.npy', target[:-1].astype(np.float32))
        np.save(dest / 'joints_with_endpoint.npy', target.astype(np.float32))
        np.savez_compressed(dest / 'source_20hz.npz', **a20)
        np.savez_compressed(dest / 'source_30hz.npz', **a30)
        np.savez_compressed(dest / 'imu.npz', schema=np.asarray(SCHEMA), mapping_version=np.asarray(MAPPING_VERSION),
            acceleration=(a30['acceleration'].astype(np.float64) @ matrix.T).astype(np.float32),
            orientation=orientation, fps=np.asarray(30.,dtype=np.float32), sensor_mask=a30['sensor_mask'],
            timestamps=np.arange(len(orientation),dtype=np.float64)/30.,
            joints=transform_points(a30['joints'], matrix, origin).astype(np.float32),
            source_to_canonical_rotation=matrix, source_origin=origin)
        (dest / 'caption.txt').write_text(Path(c['text_artifact']['path']).read_text())
        row = {'motion_id':mid, 'split':c['split'], 'source_group_id':c['source_group_id'],
            'schema': SCHEMA, 'mapping_version':MAPPING_VERSION, 'caption_index':c['caption_index'], 'caption':c['caption'],
            'joint_vec_path':str((dest/'motion.npy').resolve()), 'joints_path':str((dest/'joints.npy').resolve()),
            'text_path':str((dest/'caption.txt').resolve()), 'imu_path':str((dest/'imu.npz').resolve()),
            'num_frames':len(orientation), 'motion_frames':len(features), 'fps':30., 'model_fps':20.,
            'source_fps':c['source_fps'], 'source_frame_slice':c['available_source_slice'],
            'requested_source_slice':c['requested_source_slice'], 'terminal_policy':c['terminal_policy'],
            'prefix_seconds':c['prefix_seconds'], 'source_to_canonical_rotation':matrix.tolist(), 'source_origin':origin.tolist(),
            'RIC_roundtrip_max_abs_m':error, 'rotation_orthogonality_max_abs':ortho,
            'source_shape_beta':a20['shape'].tolist(), 'scale':1., 'retarget':'none_source_SMPL_male_shape_retained',
            'root_reference':'first_pelvis_XZ; whole_20Hz_joint_clip_min_Y; smoothed_anatomical_heading_first_value_to_Z',
            'world_alignment_certified':True, 'world_alignment_scope':'offline_synthetic_only_not_real_device_calibration',
            'semantic_alignment_certified':False, 'training_admissible':False, 'status':'pending_visual_semantic_review',
            'source_artifact':c['source_artifact'], 'original_caption_artifact':c['text_artifact'],
            'artifacts':[artifact(f) for f in sorted(dest.iterdir())]}
        rows.append(row)
        write_rows(out / 'built_pairs.jsonl', rows)
        print(json.dumps({'motion_id':mid, 'frames':len(features), 'roundtrip_m':error}),flush=True)
    write_json(out / 'report.json', {'plan':artifact(plan_path), 'pairs':artifact(out/'built_pairs.jsonl'), 'cases':len(rows),
        'max_RIC_error_m':max(r['RIC_roundtrip_max_abs_m'] for r in rows), 'budget':p['budget'],
        'training_admissible_count':0, 'limitations':['Semantic review pending.', 'No retarget to historical HumanML uniform skeleton; this is a new synthetic dataset.', 'GT canonicalization offline only; no deployable IMU calibration.', 'RIC closure does not certify generated vertex acceleration or segment orientation.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    q=sub.add_parser('plan'); q.add_argument('--output', type=Path, required=True)
    q=sub.add_parser('build'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.action=='plan': plan(args.output)
    else: build(args.plan,args.output)
