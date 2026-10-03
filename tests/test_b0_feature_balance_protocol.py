"""Owner02 analytic F objective, equal-example reduction and frozen population."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('freeze_F',ROOT/'scripts/freeze_b0_feature_balance.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

def read(name):return f.read(ROOT/f.REL/(name+'.json'))


def test_positive_normalized_weights_and_half_masses():
    weights=np.asarray(read('loss')['weights'])
    assert len(weights)==263 and (weights>0).all()
    assert weights[:67].sum()/263==pytest.approx(.5)
    assert weights[67:].sum()/263==pytest.approx(.5)
    assert weights.sum()==pytest.approx(263)
    e=np.ones((263,3));e[:67]*=2
    losses,grad=f.loss_reference(e)
    assert losses['proposed_balanced']==2.5
    assert losses['uniform']==pytest.approx((67*4+196)/263)
    assert np.mean(e**2*weights[:,None])==pytest.approx(2.5)
    assert grad[0,0]==pytest.approx(2/(67*3))
    assert grad[100,0]==pytest.approx(1/(196*3))


def test_analytic_gradient_independent_finite_difference_and_invalid_frames():
    error=np.linspace(-2,2,263*4).reshape(263,4)
    mask=np.array([True,False,True,True])
    loss,grad=f.loss_reference(error,mask)
    for c,t in [(0,0),(66,2),(67,3),(262,0),(4,1)]:
        plus=error.copy();minus=error.copy();plus[c,t]+=1e-6;minus[c,t]-=1e-6
        numeric=(f.loss_reference(plus,mask)[0]['proposed_balanced']-f.loss_reference(minus,mask)[0]['proposed_balanced'])/2e-6
        assert numeric==pytest.approx(grad[c,t],abs=2e-10)
    error[:,1]=np.nan
    assert f.loss_reference(error,mask)[0]==loss
    assert (grad[:,1]==0).all()
    with pytest.raises(ValueError):f.loss_reference(error)
    with pytest.raises(ValueError):f.loss_reference(error,np.zeros(4,bool))
    with pytest.raises(ValueError):f.loss_reference(error,np.ones(4))


def test_equal_example_mean_is_not_frame_weighted():
    pairs=[f.loss_reference(np.full((263,t),a))[0] for t,a in [(104,1),(136,2),(158,3)]]
    equal=np.mean([p['proposed_balanced'] for p in pairs])
    assert equal==pytest.approx(14/3)
    assert equal!=pytest.approx((104+4*136+9*158)/398)
    # An absent-condition/no-gradient loss remains in the three-example denominator.
    assert sum(p['proposed_balanced'] for p in pairs[:2])/3!=sum(p['proposed_balanced'] for p in pairs[:2])/2


def test_conservative_quality_gate_and_uncertain_not_equivalence():
    assert f.metric_gate(1,1)['nonworse'] and not f.metric_gate(1,1)['improved']
    assert not f.metric_gate(1.001,1)['nonworse']
    assert f.metric_gate(.999,1)['improved']
    assert f.metric_gate(np.nan,1)['status']=='not_comparable'
    assert read('evaluation')['primary']['guardrail']['degradation_margin']==0


def test_new_population_matches_exact_historical_U_not_old_guidance():
    slots=read('sampling_slots');history=read('historical_samples')
    assert len(slots)==60 and len({r['case_id'] for r in slots})==60
    assert sum(r['data_split']=='train' for r in slots)==18
    for s,h in zip(slots,history):
        assert s['historical_case_id']==h['case_id']==h['spec']['case_id']
        assert 'S1_R_unit_guidance' in h['case_id']
        assert 'sampling_alignment_authorized' in s['historical_output']['path']
        assert s['arm']=='F_root_ric_balanced' and s['checkpoint_sha256'] is None
        assert s['loss_version']==f.LOSS_VERSION
        for k in ('frames','text','caption_index','text_present','imu_present','sensor_config','random_plan_case_id','sampling_seed','guidance'):
            assert s[k]==h['spec'][k]
    assert len(read('trace_slots'))==240
    assert len(read('within_contrasts'))==36


def test_null_checks_do_not_assume_IMU_only_cross_model_invariance():
    checks=read('software_checks');slots={s['case_id']:s for s in read('sampling_slots')}
    assert len(checks)==48
    assert sum(c['reference_arm']=='U' for c in checks)==30
    assert sum(c['kind']=='F_static_vs_F_text' for c in checks)==12
    assert sum(c['kind']=='F_bypass_vs_F_text' for c in checks)==6
    for c in checks:
        assert c['max_abs_m']==1e-5
        assert slots[c['new_case_id']]['condition'] in ('text_only','physical_static','backbone_bypass')
    assert len({r['case_id'] for r in read('review_order')['rows']})==120


def test_schedule_budget_and_schema_cannot_resume_R():
    schedule=read('training_schedule');old=f.read(f.CORRECTION/'training_schedule.json')
    assert schedule==old and len(schedule['rows'])==600
    b=read('budget');d=read('design')
    assert b['authorized'] is False and b['execution_authorization'] is None
    assert b['run_directory'] is None and b['GPU_UUID'] is None
    assert (b['updates'],b['training_seconds'],b['sampling_shared_seconds'],b['total_seconds'])==(200,600,3000,3600)
    assert b['MDM_calls_total']==60600 and b['MDM_examples_total']==240600
    assert b['encoder_examples_total_max']==1320 and b['caption_calls_max']==660
    assert b['new_probes']==b['trace_extra_calls']==0
    assert d['initial']['sha256']==f.INITIAL_SHA
    assert d['historical_checkpoint']['sha256']==f.R_SHA
    assert d['checkpoint_contract']['schema']=='itm-feature-balance-control-v1'


def test_source_identity_and_immutable_outputs(tmp_path):
    p=read('package')
    assert len(p['artifacts'])==15 and p['execution_authorization'] is None
    for item in p['artifacts']+read('sources'):
        assert f.sha(ROOT/item['path'])==item['sha256']
    with pytest.raises(FileExistsError):f.guarded_build(tmp_path)


def test_lunge_review_respects_observed_clip_boundaries():
    target=read('target_review')
    assert target['frames']==136
    assert target['pelvis_y_m'][0]<target['pelvis_y_m'][30]
    assert target['pelvis_y_m'][-1]<target['pelvis_y_m'][100]
    assert target['left_ankle_relative_z_m'][0]>target['right_ankle_relative_z_m'][0]
    assert target['right_ankle_relative_z_m'][50]>target['left_ankle_relative_z_m'][50]
    assert target['windows'][0]['frames_half_open']==[0,22]
    assert target['windows'][-1]['frames_half_open']==[106,136]
