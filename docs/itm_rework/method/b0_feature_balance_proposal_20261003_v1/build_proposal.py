"""Build a proposal from saved files only; no torch/model/GPU operations."""
from pathlib import Path
import argparse
import hashlib
import json
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / 'docs/itm_rework/protocol'
S1 = ROOT / 'outputs/itm_rework/method/sampling_alignment_authorized_20260927_v1'
V = ROOT / 'outputs/itm_rework/method/state_diagnostic_authorized_20261003_v1/V'
sources = {}


def artifact(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    entry = dict(path=str(path), sha256=h.hexdigest())
    sources[str(path)] = entry
    return entry


def read(path):
    artifact(path)
    return json.loads(Path(path).read_text())


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    partition = read(PROTOCOL / 'b0_state_diagnostic_20260928_v1/feature_partition.json')['blocks']
    assert partition['root'] + partition['ric'] == list(range(67))
    assert partition['rotation6d'] + partition['local_velocity'] + partition['contact'] == list(range(67, 263))
    contrasts = read(PROTOCOL / 'b0_state_diagnostic_evaluation_20261003_v1/V_contrasts.json')
    groups = read(PROTOCOL / 'b0_state_diagnostic_20260928_v1/state_groups.json')
    group_map = {g['group_id']: g for g in groups}
    branch_metrics = []
    errors = {}
    for g in groups:
        path = V / 'states' / (g['group_id'] + '.npz'); artifact(path)
        with np.load(path) as z:
            target = z['target'].astype(np.float64)
        for branch in ('T', 'P', 'S', 'R'):
            path = V / 'branches' / (g['group_id'] + '-' + branch + '.npz'); artifact(path)
            with np.load(path) as z:
                error = z['predicted_x0'].astype(np.float64) - target
            assert np.isfinite(error).all() and error.shape[0] == 263
            d, a = float(np.mean(error[:67]**2)), float(np.mean(error[67:]**2))
            uniform = float(np.mean(error**2)); balanced = .5*(d+a)
            assert np.isclose(uniform, (67*d+196*a)/263, atol=1e-12, rtol=1e-12)
            value = dict(group_id=g['group_id'], branch=branch, uniform=uniform,
                         root_ric=d, remaining=a, proposed_balanced=balanced)
            branch_metrics.append(value); errors[g['group_id'], branch] = value
    reweighted = []
    for row in contrasts:
        lhs, rhs = row['contrast'].split('-')
        p, q = errors[row['group_id'], lhs], errors[row['group_id'], rhs]
        delta = {k:p[k]-q[k] for k in ('uniform', 'root_ric', 'remaining', 'proposed_balanced')}
        assert np.isclose(delta['uniform'], row['blocks']['all']['delta_mse'], atol=1e-12, rtol=1e-12)
        reweighted.append(dict(group_id=row['group_id'], data_split=row['data_split'],
            motion_id=row['motion_id'], sensor_config=row['sensor_config'], timestep=row['timestep'],
            contrast=row['contrast'], delta=delta))
    correction = PROTOCOL / 'b0_correction_20260926_v1'
    schedule = read(correction / 'training_schedule.json')
    design = read(correction / 'design.json')
    implementation = read(ROOT / 'outputs/itm_rework/method/correction_implementation_20260927_v1/implementation.json')
    initial = implementation['initialization']['artifact']; assert artifact(initial['path']) == initial
    train_dir = ROOT / 'outputs/itm_rework/method/correction_authorized_20260927_v2/R_static_reference_delta'
    assert len(schedule['rows']) == 600
    for update in range(1, 201):
        record = read(train_dir / f'update_{update:03d}.json')
        expected = schedule['rows'][(update-1)*3:update*3]
        assert record['optimizer_step_completed'] and record['update'] == update
        assert len(record['examples']) == 3
        for actual, row in zip(record['examples'], expected):
            assert all(actual[k] == v for k, v in row.items())
            assert actual['actual_noise_sha256'] == row['noise']['noise_sha256']
    old_checkpoint = artifact(train_dir / 'model.pt')
    assert old_checkpoint['sha256'] == '91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
    old_slots_path = PROTOCOL / 'b0_sampling_alignment_20260927_v1/sampling_slots.json'
    old_slots = read(old_slots_path); slots = []
    assert len(old_slots) == 60
    for old in old_slots:
        path = S1 / 'samples' / (old['case_id'] + '.json'); meta = read(path)
        assert meta['control_checkpoint_sha256'] == old_checkpoint['sha256']
        assert meta['initialization_sha256'] == initial['sha256']
        for key in ('motion_id','sensor_config','condition','frames','guidance','random_plan_case_id','sampling_seed'):
            assert meta[key] == old[key]
        plans = meta['random_plans']; assert len(plans) == 1 and plans[0]['steps'] == 1000
        slot = {k:old[k] for k in ('data_split','motion_id','sensor_config','condition','frames',
            'text','text_present','imu_present','random_plan_case_id','sampling_seed','guidance')}
        slot.update(slot_id=f'F{len(slots)+1:03d}', arm='F_root_ric_balanced',
            status='proposed_not_frozen_not_authorized', K=1, steps=1000,
            old_case_id=old['case_id'], baseline_output=artifact(path.with_suffix('.npz')),
            baseline_metadata=artifact(path), random_plan_id=plans[0]['random_plan_id'],
            trace_timesteps=[900,500,250,50], extra_trace_model_calls=0)
        slots.append(slot)
    for path in (PROTOCOL/'B0_STATE_DIAGNOSTIC_EVALUATION_20261003.md',
                 PROTOCOL/'B0_SAMPLING_ALIGNMENT_EVALUATION_20260928.md',
                 ROOT/'scripts/run_mdm_correction_worker.py',
                 ROOT/'outputs/mdm/runtime_root/data_loaders/humanml/scripts/motion_process.py'):
        artifact(path)
    weights = np.r_[np.full(67,263/134),np.full(196,263/392)]
    assert np.isclose(weights.sum(),263) and np.all(weights > 0)
    proposal = dict(version='itm-feature-balance-proposal-v1',status='proposed_for_02_freeze_not_authorized',
        intervention='Replace only uniform normalized x0 feature MSE by 0.5*mean(error[0:67]^2)+0.5*mean(error[67:263]^2).',
        weights=dict(root_ric_indices=[0,67],remaining_indices=[67,263],half_open=True,
            root_ric_per_coordinate='263/134',remaining_per_coordinate='263/392',sum=263),
        initial=initial, historical_checkpoint=old_checkpoint,
        common_training=artifact(correction/'design.json'), training_schedule=artifact(correction/'training_schedule.json'),
        common=design['common'], reference_mode='static_delta', updates=200, exposures=600,
        optimizer_state='Fresh AdamW exactly as historical R; no continuation from final200.',
        checkpoint_selection='Only final200; no weight/timestep/guidance/epoch search.',
        historical_control='All 60 already generated S1 samples; same training initialization and 600 addressed exposures.',
        historical_sampling_slots=artifact(old_slots_path), feature_partition=artifact(PROTOCOL/'b0_state_diagnostic_20260928_v1/feature_partition.json'),
        no_effect_claim='Saved predictions are unchanged. Reweighted saved errors are descriptive, not forecasts or validation of retraining.')
    budget=dict(status='proposal_not_authorized',GPUs=1,allocated_memory_bytes=24*1024**3,total_seconds=3600,
        training_seconds=600,sampling_shared_seconds=3000,updates=200,exposures=600,
        training_MDM_calls=600,training_MDM_examples=600,training_encoder_examples=1200,
        sampling_slots=60,sampling_steps_each=1000,sampling_MDM_calls=60000,sampling_MDM_examples=240000,
        sampling_encoder_examples_max=120,caption_calls_max=660,new_probes=0,traces=240,trace_extra_calls=0,
        final_checkpoints=1,intermediate_checkpoint_snapshots_max=3,
        timing='Absolute queue deadline starts before preflight; training phase includes preflight/load/save; sampling 3000s shared, capped by global deadline. No time transfer.',
        stop='First timeout/OOM/nonfinite/identity/numerical-anchor/counter/write failure stops queue, preserves full denominators and artifacts. No retry/warmup/GPU switch/CPU fallback.')
    audit=dict(status='CPU_saved_artifacts_only',raw_branch_reductions=len(branch_metrics),contrasts_recomputed=len(reweighted),
        training_schedule_rows_verified=600,training_update_records_verified=200,sampling_pairs=len(slots),
        split_counts=dict(Counter(s['data_split'] for s in slots)),source_files=len(sources),
        new_model_calls=0,new_training_updates=0,new_samples=0,GPU_queries=0,checkpoint_tensor_loads=0,
        interpretation='Loss allocation is a hypothesis prompted by RIC/all-feature disagreement; no evidence of a gradient conflict or root cause is claimed.')
    results={'design.json':proposal,'budget.json':budget,'sampling_pairs.json':slots,
        'saved_branch_loss_views.json':branch_metrics,'saved_contrast_loss_views.json':reweighted,
        'validation.json':audit,'sources.json':list(sources[k] for k in sorted(sources))}
    for name, value in results.items():
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
