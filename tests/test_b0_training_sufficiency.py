"""Scientific arithmetic and fixed-state population; no model/GPU fixtures."""
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from itm.metrics.training_sufficiency import temporal_decomposition, same_state_delta, PRIMARY, VIEWS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('freeze_sufficiency', ROOT/'scripts/freeze_b0_training_sufficiency.py')
freeze = importlib.util.module_from_spec(spec); spec.loader.exec_module(freeze)
PACKAGE = ROOT/freeze.REL


def test_mean_pose_improvement_can_hide_total_temporal_collapse():
    # Synthetic arithmetic only; no real result is inferred from this fixture.
    target = np.tile(np.array([-1.,1.,-1.,1.])[:,None], (1,263))
    moving_but_offset = target+2
    static = np.zeros_like(target)
    before, after = (temporal_decomposition(p,target) for p in (moving_but_offset,static))
    assert before['uniform']['mse'] == 4
    assert after['uniform']['mse'] == 1
    assert before['uniform']['temporal_error'] == 0
    assert after['uniform']['temporal_error'] == 1
    assert after['uniform']['prediction_temporal_energy'] == 0
    assert after['uniform']['target_mean_baseline_mse'] == 1
    assert after['uniform']['temporal_energy_ratio'] == 0


def test_decomposition_and_primary_partition_independent_formula():
    # Deterministic, varying coordinates; no RNG and no model state.
    target = np.arange(7*263,dtype=float).reshape(7,263)/101
    prediction = target[::-1]+np.arange(263)[None]/47
    report = temporal_decomposition(prediction,target)
    independent_mean_error = np.mean((prediction.mean(0)-target.mean(0))**2)
    independent_temporal_error = np.mean(((prediction-prediction.mean(0))-(target-target.mean(0)))**2)
    assert report['uniform']['mean_error'] == pytest.approx(independent_mean_error)
    assert report['uniform']['temporal_error'] == pytest.approx(independent_temporal_error)
    for field in ('mse','mean_error','temporal_error','prediction_temporal_energy','target_temporal_energy'):
        assert sum(report[b][field]*len(VIEWS[b])/263 for b in PRIMARY) == pytest.approx(report['uniform'][field])
    assert report['balanced']['mse'] == pytest.approx(.5*report['root_ric67']['mse']+.5*report['remaining196']['mse'])
    assert sorted(sum([VIEWS[b] for b in PRIMARY],[])) == list(range(263))


def test_masked_nan_does_not_bridge_time_gaps():
    p = np.tile(np.array([0.,np.nan,100.,101.])[:,None], (1,263))
    mask = np.array([True,False,True,True])
    report = temporal_decomposition(p,p,mask)['uniform']
    assert report['mse'] == 0
    assert report['adjacent_valid_pairs'] == 1
    assert report['prediction_difference_rms'] == 1


def test_constant_target_ratio_and_single_frame_difference_are_NA():
    r = temporal_decomposition(np.ones((1,263)),np.zeros((1,263)))['uniform']
    assert r['temporal_energy_ratio'] is None
    assert r['prediction_difference_rms'] is None
    assert r['mean_error'] == 1 and r['temporal_error'] == 0


@pytest.mark.parametrize('bad', [np.empty((0,263)), np.zeros((2,262)), np.full((2,263),np.nan), np.full((2,263),np.inf)])
def test_bad_features_rejected(bad):
    with pytest.raises(ValueError):
        temporal_decomposition(bad,np.zeros((2,263)))


@pytest.mark.parametrize('mask', [np.array([False,False]), np.array([1,1]), np.array([True])])
def test_bad_mask_rejected(mask):
    with pytest.raises(ValueError):
        temporal_decomposition(np.ones((2,263)),np.zeros((2,263)),mask)


def test_signed_same_state_intervention_and_balanced_loss():
    g = np.zeros((2,263)); b = np.ones_like(g); p = 2*b
    report = same_state_delta(b,p,g)
    assert report['uniform']['delta_mse'] == 3
    assert report['uniform']['cross'] == 2
    assert report['uniform']['residual_energy'] == 1
    assert report['balanced']['delta_mse'] == 3
    assert report['uniform']['count'] == 526
    with pytest.raises(ValueError):
        same_state_delta(b[:1],p,g)


def test_planned_caches_and_same_state_identity_without_real_arrays():
    # Twelve cases, deterministic fixture; the real population is checked below.
    rows = [dict(state_id=f'S{i:03}',cache_id=f'C{i%12:02}',frames=3,timestep=50,
                 arrays=dict(x_t=dict(sha256=str(i)),target=dict(sha256='target')),
                 historical_anchors=[]) for i in range(192)]
    jobs, checks = freeze.make_plan(rows)
    assert len(jobs) == 1152
    assert sum(r['encoder_calls'] for r in jobs) == 432
    assert sum(r['caption_calls'] for r in jobs) == 12
    assert len(checks) == 2304  # no saved anchors in fixture
    for i,row in enumerate(rows):
        same = [j for j in jobs if j['state_id']==row['state_id']]
        assert len(same) == 6
        assert len({j['state_sha256'] for j in same}) == 1
        assert all(j['branch_order']==['T','P','S','R'] for j in same)


def test_freeze_rejects_overwriting_before_reading_sources(tmp_path):
    with pytest.raises(FileExistsError):
        freeze.guarded_build(tmp_path)


def test_real_frozen_population_and_saved_bytes():
    states = freeze.read(PACKAGE/'states.json')
    jobs = freeze.read(PACKAGE/'jobs.json')
    checks = freeze.read(PACKAGE/'software_checks.json')
    assert len(states)==len({r['state_id'] for r in states})==192
    assert Counter(r['state_family'] for r in states)==dict(GT_noised_Q=96,U_paired_trace=48,F_paired_trace=48)
    assert Counter(r['data_split'] for r in states)==dict(train=96,val=96)
    for row in states:
        values = freeze.load_state(row)
        assert values['x_t'].shape==(1,263,1,row['frames'])
        assert row['planned_checkpoints']==list(freeze.SNAPSHOTS)
    assert freeze.make_plan(states)==(jobs,checks)
    assert len(checks)==2880
    assert len({r['check_id'] for r in checks})==2880
    counts=Counter(r['reference']['kind'] for r in checks)
    assert counts==dict(same_job=1344,new_prediction=960,saved_array=576)
    assert all(r['atol']==r['rtol']==1e-6 for r in checks)


def test_real_protocol_prohibits_selection_training_and_new_noise():
    budget = freeze.read(PACKAGE/'budget.json')
    assert all(v==0 for v in budget['prohibited'].values())
    assert budget['manual_GPU_approval_required'] is False
    assert budget['absolute_seconds']==600
    evaluation = freeze.read(PACKAGE/'evaluation.json')
    assert evaluation['branch_rows']==evaluation['branch_contrast_rows']==evaluation['checkpoint_contrast_rows']==4608
    assert ['F200','F150'] in evaluation['checkpoint_contrasts']
    assert 'Never choose the best intermediate checkpoint' in evaluation['decision_rules'][-1]
