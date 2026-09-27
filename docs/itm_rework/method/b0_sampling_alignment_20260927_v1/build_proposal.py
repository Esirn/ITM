"""CPU reads/NumPy only: evidence summary and unfrozen proposal, never execution."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / 'docs/itm_rework/protocol/b0_correction_20260926_v1'
EVALUATION = ROOT / 'docs/itm_rework/protocol/b0_correction_evaluation_20260927_v1'
RUN = ROOT / 'outputs/itm_rework/method/correction_authorized_20260927_v2'
ARM = 'R_static_reference_delta'
CHECKPOINT_SHA = '91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
SOURCES = set()


def read(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def artifact(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    checkpoint = RUN / ARM / 'model.pt'
    SOURCES.add(checkpoint)
    assert artifact(checkpoint)['sha256'] == CHECKPOINT_SHA
    old_samples = read(FROZEN / 'sampling_slots.json')
    reference_slots = [row for row in old_samples if row['arm'] == ARM]
    assert len(reference_slots) == 60
    evaluation = read(EVALUATION / 'samples.json')['rows']
    lookup = {(r['arm'], r['data_split'], r['motion_id'], r['sensor_config'], r['condition']): r
              for r in evaluation}
    metrics = ('global_MPJPE_m', 'root_relative_MPJPE_m',
               'head_or_wrists_position_joint_proxy_m', 'velocity_joint_proxy_RMSE_m_s')
    anchors = []
    for slot in reference_slots:
        if slot['condition'] != 'paired':
            continue
        key = (ARM, slot['data_split'], slot['motion_id'], slot['sensor_config'])
        path = RUN / 'samples' / (slot['case_id'] + '.npz')
        SOURCES.add(path)
        with np.load(path, allow_pickle=False) as arrays:
            mask = arrays['sensor_mask'][0].astype(bool)
            observed = arrays['normalized_imu'][0][:, mask]
            reference = arrays['normalized_reference_imu'][0][:, mask]
            delta = observed.astype(np.float64) - reference.astype(np.float64)
            assert np.isfinite(delta).all() and delta.size
            input_stats = dict(active_input_rms=float(np.sqrt(np.mean(observed.astype(np.float64) ** 2))),
                active_delta_rms=float(np.sqrt(np.mean(delta ** 2))),
                acceleration_delta_rms=float(np.sqrt(np.mean(delta[..., :3] ** 2))),
                orientation_delta_rms=float(np.sqrt(np.mean(delta[..., 3:] ** 2))),
                active_input_abs_max=float(np.abs(observed).max()))
        conditions = {}
        for condition in ('paired', 'text_only', 'physical_static', 'tail_reversed', 'imu_only'):
            record = lookup.get(key + (condition,))
            conditions[condition] = {k: record['metrics'][k] for k in metrics} if record else None
        anchors.append(dict(data_split=slot['data_split'], motion_id=slot['motion_id'],
            sensor_config=slot['sensor_config'], case_id=slot['case_id'],
            input_statistics_dimensionless=input_stats, owner02_metrics=conditions))
    assert len(anchors) == 12

    schedule = read(FROZEN / 'training_schedule.json')['rows']
    training = []
    for arm in ('L_exact_length', ARM):
        bins = defaultdict(list)
        counts = Counter()
        for index in range(200):
            record = read(RUN / arm / f'update_{index+1:03d}.json')
            assert record['update'] == index + 1 and record['optimizer_step_completed']
            for offset, actual in enumerate(record['examples']):
                planned = schedule[index * 3 + offset]
                assert all(actual[k] == v for k, v in planned.items())
                assert actual['actual_noise_sha256'] == planned['noise']['noise_sha256']
                counts[f"text={int(actual['text_present'])},imu={int(actual['imu_present'])}"] += 1
                bucket = f"{actual['motion_id']}/{actual['sensor_config']}/t{actual['timestep']//250}/imu{int(actual['imu_present'])}"
                bins[bucket].append(actual['loss'])
        training.append(dict(arm=arm, updates=200, exposures=600, condition_counts=dict(counts),
            descriptive_loss_strata=[dict(stratum=k, n=len(v), mean=float(np.mean(v)),
                                          min=float(min(v)), max=float(max(v))) for k,v in sorted(bins.items())],
            interpretation='Descriptive changing-weights/noise population, not convergence or final-model teacher-forced performance'))

    # Algebra only, no neural modules or diffusion calls.
    rng = np.random.default_rng(20260927)
    f00, f10, f01, f11, target = rng.normal(size=(5, 2, 263, 1, 5))
    def cfg(scale):
        return f00 + scale * (f10-f00) + (f01-f00) + (f11-f10-f01+f00)
    addition = 1.5 * (f10-f00)
    error = f11-target
    mse_delta = np.mean((error+addition)**2)-np.mean(error**2)
    decomposition = 2*np.mean(error*addition)+np.mean(addition**2)
    algebra = dict(unit_vs_joint_max_abs=float(np.max(np.abs(cfg(1)-f11))),
                  original_vs_joint_plus_text_max_abs=float(np.max(np.abs(cfg(2.5)-(f11+addition)))),
                  mse_decomposition_abs_error=float(abs(mse_delta-decomposition)))
    assert max(algebra.values()) < 1e-12
    write('existing_evidence.json', dict(read_commit='c6aef58', status='CPU_existing_artifacts_only',
        anchors=anchors, training=training, numerical_algebra_check=algebra,
        new_model_forwards=0, new_training_updates=0, new_samples=0, gpu_queries=0,
        missing_evidence=['No fixed teacher-forced probes of final L/R',
                          'No stored L/R free-trajectory branch predictions',
                          'No evidence that unit guidance resolves either failure']))

    samples = []
    for index, slot in enumerate(reference_slots):
        assert slot['case_id'].startswith(ARM+'--')
        row = dict(slot, arm='S1_R_unit_guidance', case_id='S1_R_unit_guidance--'+slot['case_id'].split('--', 1)[1],
                   slot_index=index, status='proposed_not_frozen_not_authorized',
                   comparison_case_id=slot['case_id'], comparison_output=str(RUN / 'samples' / (slot['case_id']+'.npz')),
                   checkpoint=artifact(checkpoint), checkpoint_arm=ARM,
                   checkpoint_selection='fixed_existing_R_final200_no_selection',
                   guidance=dict(text_scale=1., imu_scale=1., joint_scale=1., mode='factorized', branch_execution='batched'),
                   trace_timesteps=[50,250,500,900], trace_additional_model_forwards=0)
        samples.append(row)
    assert len({r['case_id'] for r in samples}) == 60
    for new, old in zip(samples, reference_slots):
        for key in ('random_plan_case_id','sampling_seed','steps','frames','data_split','motion_id',
                    'sensor_config','condition','text','caption_index','text_present','imu_present','sensor_slots'):
            assert new[key] == old[key], key
    write('proposed_sampling_slots.json', samples)

    old_probes = read(FROZEN / 'probe_slots.json')
    groups = [r for r in old_probes if r['context']=='exact_length' and r['condition']=='paired']
    assert len(groups) == 96 and len({r['noise_key'] for r in groups}) == 96
    probes = []
    branches = [('f00', 'unconditional', False, False), ('f10', 'text_only', True, False),
                ('f11_paired', 'paired', True, True), ('f11_static', 'physical_static', True, True),
                ('f11_reversed', 'tail_reversed', True, True)]
    for group in groups:
        for branch, condition, text_present, imu_present in branches:
            probes.append(dict(group, probe_id=f'Q{len(probes)+1:04d}',
                status='proposed_not_frozen_not_authorized', checkpoint=artifact(checkpoint),
                checkpoint_arm=ARM, branch=branch, condition=condition,
                text_present=text_present, imu_present=imu_present,
                comparison_group=group['noise_key'], maximum_encoder_examples=2 if imu_present else 0))
    assert len(probes)==480 and sum(r['maximum_encoder_examples'] for r in probes)==576
    write('proposed_probe_slots.json', probes)
    write('proposed_budget.json', dict(version='itm-sampling-alignment-proposal-v1',
        status='proposed_not_frozen_not_authorized', authorized=False, read_commit='c6aef58',
        training_updates=0, new_checkpoints=0, existing_checkpoint=artifact(checkpoint),
        diagnostics=dict(branch_examples=480, encoder_examples_max=576, maximum_seconds=600),
        sampling=dict(count=60, K=1, steps=1000, denoising_calls_max=60000,
                      branch_examples_max=240000, encoder_examples_max=120, maximum_seconds_shared=2400,
                      trace_records_max=240, trace_additional_model_forwards=0),
        totals=dict(MDM_branch_examples_max=240480, encoder_examples_max=696,
                    GPUs=1, allocated_memory_bytes=24*2**30, maximum_seconds=3600),
        stage_order=['Q_R_final_fixed_probes','S1_60_samples'],
        stop='First timeout/OOM/nonfinite/identity/counter failure stops queue; retain every slot and completed artifact; no retries, fallback or extra cases',
        interpretation='Poor finite probe scores do not change the predeclared sampler or select a checkpoint',
        gpu_uuid=None, run_directory=None,
        prerequisites=['02 freezes evaluation/slots/noise/tolerances/budget',
                       '01 implements versioned runner and CPU checks', '02 software acceptance',
                       'new explicit user authorization bound to final implementation and run/GPU', 'GPU preflight']))

    SOURCES.update([Path(__file__).resolve(),
        ROOT/'docs/itm_rework/protocol/B0_CORRECTION_EVALUATION_20260927.md',
        ROOT/'scripts/run_mdm_correction_worker.py', ROOT/'src/itm/models/mdm_imu_control.py',
        ROOT/'src/itm/models/mdm_sampling.py', FROZEN/'package.json', FROZEN/'probe_noise.json'])
    write('sources.json', [artifact(p) for p in sorted(SOURCES)])
    print(json.dumps(dict(anchors=12, existing_training_records=400, proposed_probes=len(probes),
        proposed_samples=len(samples), proposed_updates=0, new_model_forwards=0, gpu_queries=0,
        algebra=algebra), indent=2))


if __name__ == '__main__':
    main()
