#!/usr/bin/env python
"""Freeze the next fixed-state diagnostic; arrays/JSON/hashes only, no execution.

The separate checkpoint audit deserializes existing tensors on CPU. This builder
does not import a neural framework, spawn a process, regenerate noise, or mutate
an old protocol. Rebuild into a different directory to compare file bytes.
"""
import argparse
import builtins
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from itm.metrics.training_sufficiency import temporal_decomposition, VIEWS, PRIMARY

ROOT = Path(__file__).resolve().parents[1]
REL = 'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v2'
AUDIT = ROOT/'outputs/itm_rework/protocol/training_sufficiency_preparation_20261008_v2/checkpoint_audit.json'
FP = ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1'
VP = ROOT/'docs/itm_rework/protocol/b0_state_diagnostic_20260928_v1'
QP = ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1'
F = ROOT/'outputs/itm_rework/method/feature_balance_authorized_20261008_v1'
U = ROOT/'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
SNAPSHOTS = ('I', 'F050', 'F100', 'F150', 'F200', 'U200')
BRANCHES = ('T', 'P', 'S', 'R')
FAMILIES = ('GT_noised_Q', 'U_paired_trace', 'F_paired_trace')
BRANCH_CONTRASTS = (('P', 'S'), ('P', 'R'), ('P', 'T'), ('R', 'S'))
PROGRESS_CONTRASTS = (('F050', 'I'), ('F100', 'F050'), ('F150', 'F100'),
                      ('F200', 'F150'), ('F200', 'I'), ('F200', 'U200'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(array):
    a = np.asarray(array)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files if k != 'metadata'}


def matrix(x):
    a = np.asarray(x)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3 or a.shape[:2] != (263, 1):
        raise ValueError('Expected [1?,263,1,T] saved features')
    return a[:, 0].T


def load_state(row):
    """Return canonical batch1 views; never creates noise or a diffusion state."""
    raw = arrays(ROOT/row['saved_state']['path'])
    target = arrays(ROOT/row['target_artifact']['path'])[row['target_array']]
    xt, mask = raw['x_t'], raw['frame_mask']
    if row['state_family'] == 'GT_noised_Q':
        xt, mask = xt[None], mask[None]
    values = dict(x_t=xt, target=target, frame_mask=mask)
    for name, expected in row['arrays'].items():
        if describe(values[name]) != expected:
            raise ValueError('Fixed saved-state descriptor changed: '+name)
    return values


def make_plan(states):
    jobs, checks = [], []
    seen_caption = set()
    for snapshot in SNAPSHOTS:
        seen_control = set()
        for row in states:
            sid, cid = row['state_id'], row['cache_id']
            job_id = snapshot+'-'+sid
            jobs.append(dict(job_id=job_id, snapshot_id=snapshot, state_id=sid, cache_id=cid,
                             branch_order=list(BRANCHES), branch_ids=[job_id+'-'+b for b in BRANCHES],
                             status='planned', MDM_calls=1, MDM_examples=4,
                             encoder_calls=0 if cid in seen_control else 6,
                             caption_calls=0 if cid in seen_caption else 1,
                             state_sha256=row['arrays']['x_t']['sha256'],
                             target_sha256=row['arrays']['target']['sha256'],
                             input_batch_shape=[4,263,1,row['frames']], scaled_t=[row['timestep']]*4))
            seen_control.add(cid); seen_caption.add(cid)
            specifications = [('S_vs_T', 'S', dict(kind='same_job', branch='T'))]
            if snapshot == 'I':
                specifications.append(('P_vs_T_initial_zero_adapters', 'P', dict(kind='same_job', branch='T')))
            else:
                specifications.append(('T_vs_I_same_state', 'T', dict(kind='new_prediction', job_id='I-'+sid, branch='T')))
            for anchor in row['historical_anchors']:
                if snapshot == anchor['snapshot_id']:
                    specifications.append(('historical_'+anchor['branch'], anchor['branch'], dict(kind='saved_array', **anchor)))
            for name, branch, reference in specifications:
                checks.append(dict(check_id=job_id+'-'+name, job_id=job_id, state_id=sid,
                                   snapshot_id=snapshot, branch=branch, reference=reference,
                                   atol=1e-6, rtol=1e-6, status='planned'))
    return jobs, checks


def build(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Frozen package cannot be overwritten')
    sources = {}

    def bind(path, expected=None):
        path = (ROOT/Path(path)).resolve()
        if path not in sources:
            sources[path] = dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), sha256=sha(path))
        result = sources[path]
        if expected is not None and result['sha256'] != expected:
            raise ValueError('Source hash changed: '+str(path))
        return result

    def verify(entry):
        return bind(entry['path'], entry['sha256'])

    def package(path, expected):
        bind(path, expected)
        for entry in read(path)['artifacts']:
            verify(entry)
        for entry in read(path.parent/'sources.json'):
            verify(entry)

    package(FP/'package.json', 'a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9')
    package(VP/'package.json', '6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012')
    package(QP/'package.json', '537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60')
    status = read(F/'runtime/status.json'); bind(F/'runtime/status.json')
    if status['status'] != 'completed':
        raise ValueError('F full queue is not complete')
    implementation = verify(status['implementation'])
    if implementation['sha256'] != 'db323ef515ede3252898f322f7badda10913aea64d67cb4abb1f3e9fc4605e30':
        raise ValueError('F completed implementation identity changed')
    for entry in read(ROOT/implementation['path'])['expected_artifacts']:
        verify(entry)
    audit = read(AUDIT); bind(AUDIT)
    if (audit['version'] != 'itm-training-snapshot-CPU-audit-v1'
            or [r['snapshot_id'] for r in audit['snapshots']] != list(SNAPSHOTS)
            or audit['unique_checkpoint_files'] != 6 or audit['tensor_deserializations'] != 7
            or audit['model_forwards'] or audit['GPU_queries'] or audit['new_training'] or audit['new_samples']):
        raise ValueError('Wrong CPU checkpoint audit')
    for row in audit['snapshots']:
        verify(row['artifact'])
    for path in (Path(__file__), ROOT/'scripts/audit_b0_training_snapshots.py',
                 ROOT/'src/itm/metrics/training_sufficiency.py', ROOT/'src/itm/metrics/feature_balance.py',
                 ROOT/'tests/test_b0_training_sufficiency.py',
                 ROOT/'tests/test_mdm_training_snapshots.py',
                 ROOT/'docs/itm_rework/protocol/B0_TRAINING_SUFFICIENCY_DESIGN_20261008.md',
                 ROOT/'docs/itm_rework/protocol/B0_FEATURE_BALANCE_EVALUATION_20261008.md',
                 ROOT/'docs/itm_rework/GPU_EXECUTION_POLICY_20261008.json',
                 ROOT/'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2/decision.json'):
        bind(path)

    cases = read(VP/'input_cases.json')
    case_map = {(r['data_split'], r['motion_id'], r['sensor_config']): r for r in cases}
    if len(cases) != 12 or len(case_map) != 12:
        raise ValueError('Twelve unique input cases required')
    mean, std = [np.load(ROOT/verify(a)['path']).astype(np.float32)
                 for a in read(FP/'design.json')['common']['motion_normalization']]
    targets = {}
    for case in cases:
        verify(case['original_sample'])
        path = Path(case['sample_spec']['gt_path']).with_name('motion.npy'); bind(path)
        targets[case['cache_id']] = ((np.load(path)-mean)/std).astype(np.float32).T[:,None]

    states, baseline = [], []

    def append(family, identity, case, timestep, state_path, target_path, metadata_path, anchors, noise_seed=None):
        raw, target_raw = arrays(state_path), arrays(target_path)
        xt, mask, target = raw['x_t'], raw['frame_mask'], target_raw['target']
        if family == 'GT_noised_Q':
            xt, mask = xt[None], mask[None]
        length = case['frames']
        if (xt.shape != (1,263,1,length) or xt.dtype != np.float32 or mask.shape != (1,length)
                or mask.dtype != np.bool_ or not mask.all() or not np.isfinite(xt).all()
                or not np.array_equal(target, targets[case['cache_id']])):
            raise ValueError('Saved state/target/length/mask identity mismatch: '+identity)
        meta = read(metadata_path)
        if meta['scaled_t'] != [timestep] or meta['checkpoint_sha256'] != audit['snapshots'][-1 if family != 'F_paired_trace' else -2]['artifact']['sha256']:
            raise ValueError('Saved state model/timestep lineage mismatch')
        # T branch differs only in IMU presence; all observed inputs are frozen.
        expected = next(r for r in case['branches'] if r['branch'] == ('T' if family == 'GT_noised_Q' else 'P'))['arrays']
        if meta['input_hashes'] != expected:
            raise ValueError('Saved state input/caption/config lineage changed')
        sid = f'S{len(states)+1:03d}'
        row = dict(state_id=sid, state_family=family, original_identity=identity,
                   data_split=case['data_split'], motion_id=case['motion_id'], sensor_config=case['sensor_config'],
                   frames=length, timestep=timestep, noise_seed=noise_seed, cache_id=case['cache_id'],
                   saved_state=bind(state_path), saved_state_metadata=bind(metadata_path),
                   target_artifact=bind(target_path), target_array='target',
                   arrays={k: describe(v) for k,v in dict(x_t=xt, target=target, frame_mask=mask).items()},
                   historical_anchors=anchors, planned_checkpoints=list(SNAPSHOTS))
        states.append(row)
        for anchor in anchors:
            verify(anchor['artifact'])
        if family != 'GT_noised_Q':
            pred = matrix(raw['f11'])
            baseline.append(dict(state_id=sid, state_family=family, data_split=case['data_split'],
                                 motion_id=case['motion_id'], sensor_config=case['sensor_config'], timestep=timestep,
                                 checkpoint=anchors[1]['snapshot_id'], branch='P',
                                 comparison_status='descriptive_different_states_not_causal_F_vs_U',
                                 temporal=temporal_decomposition(pred, matrix(target), mask[0])))

    qgroups = read(QP/'probe_groups.json')
    if len(qgroups) != 96:
        raise ValueError('The complete frozen Q96 population is required')
    for group in qgroups:
        case = case_map[group['data_split'], group['motion_id'], group['sensor_config']]
        text_path = U/'Q'/(group['probe_ids'][1]+'.npz')
        text = arrays(text_path)
        anchors = []
        for branch, pid in zip(BRANCHES, group['probe_ids'][1:]):
            path = U/'Q'/(pid+'.npz'); z = arrays(path); meta = read(path.with_suffix('.json'))
            if any(not np.array_equal(z[k], text[k]) for k in ('x_t', 'target', 'frame_mask')):
                raise ValueError('Historical Q branches use different saved states')
            if meta['noise_seed'] != group['noise_seed'] or meta['timestep'] != group['timestep']:
                raise ValueError('Historical Q noise/timestep identity changed')
            for name, desc in meta['arrays'].items():
                if describe(z[name]) != desc:
                    raise ValueError('Historical Q array descriptor changed')
            bind(path.with_suffix('.json'))
            anchors.append(dict(snapshot_id='U200', branch=branch, artifact=bind(path), array='predicted_x0', descriptor=describe(z['predicted_x0'])))
        append('GT_noised_Q', group['group_id'], case, group['timestep'], text_path,
               text_path, text_path.with_suffix('.json'), anchors, group['noise_seed'])
    vgroups = read(VP/'state_groups.json')
    fsamples = {r['case_id']: r for r in read(FP/'sampling_slots.json')}
    paired = [r for r in read(FP/'trace_slots.json') if fsamples[r['case_id']]['condition'] == 'paired']
    if len(vgroups) != 48 or len(paired) != 48:
        raise ValueError('All U48 and F48 paired trace states are required')
    fmap = {r['historical_trace_id']: r for r in paired}
    for family in FAMILIES[1:]:
        for group in vgroups:
            case = case_map[group['data_split'], group['motion_id'], group['sensor_config']]
            trace = group['trace_id'] if family == 'U_paired_trace' else fmap[group['trace_id']]['trace_id']
            path = (U if family == 'U_paired_trace' else F)/'traces'/(trace+'.npz')
            checkpoint = 'U200' if family == 'U_paired_trace' else 'F200'
            z = arrays(path); meta = read(path.with_suffix('.json'))
            if meta['condition'] != 'paired' or meta['trace_id'] != trace:
                raise ValueError('Paired trace identity changed')
            for name in ('x_t', 'f10', 'f11', 'frame_mask'):
                if describe(z[name]) != meta['arrays'][name]:
                    raise ValueError('Paired trace array descriptor changed')
            anchors = [dict(snapshot_id=checkpoint, branch=b, artifact=bind(path), array=a, descriptor=describe(z[a]))
                       for b,a in (('T','f10'), ('P','f11'))]
            append(family, trace, case, group['timestep'], path, ROOT/group['target_artifact']['path'],
                   path.with_suffix('.json'), anchors)

    jobs, checks = make_plan(states)
    validation = dict(states=len(states), jobs=len(jobs), branch_predictions=len(jobs)*4, software_checks=len(checks),
                      snapshots=len(SNAPSHOTS), input_cases=len(cases),
                      state_families=dict(Counter(r['state_family'] for r in states)),
                      MDM_calls=sum(r['MDM_calls'] for r in jobs), MDM_examples=sum(r['MDM_examples'] for r in jobs),
                      encoder_calls=sum(r['encoder_calls'] for r in jobs), caption_calls=sum(r['caption_calls'] for r in jobs),
                      original_runtime_dependencies_verified=True, source_files=len(sources),
                      current_model_imports=0, current_checkpoint_deserializations=0, current_model_forwards=0,
                      current_GPU_queries=0, current_new_training=0, current_new_samples=0)
    if tuple(validation[k] for k in ('states','jobs','branch_predictions','software_checks','encoder_calls','caption_calls')) != (192,1152,4608,2880,432,12):
        raise ValueError('Frozen diagnostic population/call plan mismatch')
    population = Counter((r['state_family'],r['data_split'],r['motion_id'],r['sensor_config'],r['timestep']) for r in states)
    if len(population) != 144 or any(n != (2 if k[0] == 'GT_noised_Q' else 1) for k,n in population.items()):
        raise ValueError('Missing or duplicate fixed-state source/timestep cells')
    design = dict(version='itm-training-sufficiency-design-v1',
                  supersedes='docs/itm_rework/protocol/b0_training_sufficiency_20261008_v1/package.json',
                  revision='CPU audit accounting: seven loads of six unique checkpoint files. Fixed states, scientific comparisons, model/GPU counts and proposed budget unchanged; v1 was never executed.',
                  scientific_scope='Read-only local optimization and temporal contraction diagnostic; no repair/convergence certification.',
                  original_F_package=bind(FP/'package.json'), original_V_package=bind(VP/'package.json'), original_Q_package=bind(QP/'package.json'),
                  checkpoint_audit=bind(AUDIT), snapshots=list(SNAPSHOTS), state_families=list(FAMILIES), branch_order=list(BRANCHES),
                  common=read(FP/'design.json')['common'], environment=audit['environment'],
                  initial_runtime_override='I uses its existing zero-adapter tensors and static_delta runtime semantics; raw reference_mode=none is preserved in audit. No new initialization/RNG.',
                  inference='Original loader; backbone/CLIP once, swap only audited E/adapters. eval float32 denoiser/control, native CLIP dtype; no AMP/grad/CFG/decode, original identity timestep map.',
                  cache='Caption batch1 once per twelve cases, reused across snapshots. E observed/reference batch1 for P/S/R per snapshot/case, six calls; zero T bypasses E. Clear controls at every snapshot; bind checkpoint/input/hash/config/dtype/eval.',
                  software_bounds='Every check uses abs(new-reference)<=1e-6+1e-6*abs(reference); no relaxation after outcomes. Q is former batch5, new batch4, so bitwise equality is not presumed.',
                  artifact_order='Save independent-readable finite raw batch4 predictions/state identity BEFORE deciding any anchor; preserve diagnostic_only failures; no validation forwards.')
    evaluation = dict(version='itm-training-sufficiency-evaluation-v1', branch_rows=4608,
                      branch_contrasts=[list(p) for p in BRANCH_CONTRASTS], branch_contrast_rows=4608,
                      checkpoint_contrasts=[list(p) for p in PROGRESS_CONTRASTS], checkpoint_contrast_rows=4608,
                      feature_views=VIEWS, primary_partition=list(PRIMARY), feature_units='Bound pretrained normalized HumanML263; not meters or physical IMU closure.',
                      temporal_formula='Per coordinate valid-frame mean: MSE(p,g)=MSE(mean(p),mean(g))+MSE(p-mean(p),g-mean(g)). Temporal error=prediction energy+target energy-2covariance;64eps float64 identity bound. Target mean is a descriptive static baseline, no fitted transform.',
                      differences='Fixed adjacent valid frames only; temporal_energy_ratio NA when target energy=0. Report energies, covariance, mean error and temporal error together; no new semantic-success threshold.',
                      contrast_formula='e=baseline-GT,d=prediction-baseline: delta_MSE=2mean(e*d)+mean(d*d); raw difference RMS/max/count, primary partition weighted dims/263. Balanced=.5D67+.5A196.',
                      aggregation='Keep every source/config/t/family/branch/snapshot row. Within Q average its two seeds per source first; equal three sources per split/config/t/family; mean/min/max/planned3/valid_n/missing. No pooling times/families/configs/splits or treating correlated states as independent motions.',
                      decision_rules=[
                          'F050/100/150/200 on the identical saved bytes shows local optimization progress. Final150->200 signed changes remain descriptive; no predeclared plateau/convergence threshold and no automatic extra training.',
                          'Lower total error together with lower predicted temporal energy and unreduced temporal error is evidence of contraction on those x0 predictions. Agreement with known full-motion collapse strengthens that development diagnosis; it does not establish a unique mechanism.',
                          'Improving GT-noised local predictions but failing trajectory states suggests state-dependent local performance; cross-family contrasts are NOT controlled distribution-shift causal effects.',
                          'Favorable P-S/P-R at these states with failed existing full actions means local conditioning is insufficient for behavior. An absent order benefit is not permission for hard masking or declaring input unused.',
                          'Incomplete/failed software means inconclusive primary diagnostic; finite failed arrays are diagnostic_only, preserve every denominator. Never choose the best intermediate checkpoint or sample it under this protocol.' ],
                      inherited_full_motion_decision=bind(ROOT/'outputs/itm_rework/protocol/feature_balance_evaluation_20261008_v2/decision.json'),
                      preserved_negatives=['train000506 head/wrists incomplete alternating lunge', 'train004728 right leg hold/direction/amplitude', 'val000382 wrists lower-body deformation', 'val000139 ordering regression', 'val000433 kick return', 'all old numeric-zero/IMU-only/text/static/reversed review rows'],
                      limitations='Six exposed sources, one training seed, one paired trajectory per config; no generalization/significance, full convergence, future-training benefit, new motion success rate, real-device, compatible-edit, strict IMU-closure or paper-result claim.')
    budget = dict(version='itm-training-sufficiency-budget-v1', status='frozen_pending_implementation',
                  MDM_calls=1152, MDM_examples=4608, encoder_batch1_calls=432, caption_batch1_calls=12,
                  states=192, jobs=1152, branch_predictions=4608, software_checks=2880,
                  GPUs=1, memory_GiB=24, absolute_seconds=600,
                  prohibited=dict(new_training=0,new_checkpoints=0,full_sampling=0,q_sample=0,new_noise=0,posterior=0,CFG=0,motion_decode=0,warmup=0,retries=0),
                  deadline_rule='One absolute deadline from BEFORE preflight, includes identities/load/encode/forward/raw-save/check/receipt validation; do not reset by snapshot or reuse old budget.',
                  stopping='First identity/timeout/OOM/nonfinite/count/software failure stops entire queue. Poor finite model quality never stops population. Predeclare all192states/1152jobs/4608branches/2880checks; reserve, actual, completed counters separately; preserve completed/failed/not_run_after_stop.',
                  prerequisites=['Implement bounded worker/supervisor and incremental CPU failure/cache/identity tests', 'Create final dependency manifest and concrete unique run/GPU record under standing user delegation', 'Run once after preflight without interfering with unrelated processes'],
                  standing_delegation=bind(ROOT/'docs/itm_rework/GPU_EXECUTION_POLICY_20261008.json'), manual_GPU_approval_required=False)
    files = dict(design=design, evaluation=evaluation, budget=budget, checkpoints=audit['snapshots'],
                 input_cases=cases, states=states, jobs=jobs, software_checks=checks,
                 existing_trace_temporal_baseline=dict(rows=baseline, model_forwards=0, interpretation='Prior saved F/U predictions; state bytes differ across families, not a weight intervention.'),
                 validation=validation, sources=[sources[k] for k in sorted(sources)])
    output.mkdir(parents=True)
    for name, value in files.items():
        (output/(name+'.json')).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'package.json').write_text(json.dumps(dict(version='itm-training-sufficiency-package-v1', status='frozen_pending_implementation',
        artifacts=[dict(path=REL+'/'+p.name, sha256=sha(p)) for p in sorted(output.iterdir())],
        new_model_forwards=0, execution_record=None), indent=2, ensure_ascii=False)+'\n')
    return dict(validation, package_sha256=sha(output/'package.json'))


def guarded_build(output):
    original = builtins.__import__
    def restricted(name, *args, **kwargs):
        if name.split('.')[0] in ('torch','tensorflow','jax'):
            raise AssertionError('Neural imports forbidden in freeze')
        return original(name, *args, **kwargs)
    def reject(*args, **kwargs):
        raise AssertionError('Process/GPU execution forbidden in freeze')
    with ExitStack() as stack:
        stack.enter_context(patch.object(builtins, '__import__', restricted))
        for name in ('Popen','run','call','check_call','check_output'):
            stack.enter_context(patch.object(subprocess, name, reject))
        return build(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/REL)
    args = parser.parse_args(); print(json.dumps(guarded_build(args.output)))
