"""02 frozen-plan tests: pure NumPy/files, no pretrained model or GPU work."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sampling_freeze',ROOT/'scripts/freeze_b0_sampling_alignment.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
P=ROOT/f.REL


def test_manifest_sources_and_no_authorization():
    package=f.read(P/'package.json');assert package['execution_authorization'] is package['GPU_UUID'] is package['run_directory'] is None
    for a in package['artifacts']+f.read(P/'sources.json'):assert f.sha(ROOT/a['path'])==a['sha256']
    d=f.read(P/'design.json');assert d['checkpoint']['sha256']==f.CHECKPOINT_SHA
    assert d['training_updates']==d['new_checkpoints']==0
    assert d['common']['imu_normalization']==f.read(f.OLD/'design.json')['common']['imu_normalization']


def test_all_historical_slots_inputs_and_noise_preserved():
    new=f.read(P/'sampling_slots.json');old=[r for r in f.read(f.OLD/'sampling_slots.json') if r['arm']==f.R]
    assert len(new)==len(old)==len({r['case_id'] for r in new})==60
    assert Counter(r['data_split'] for r in new)=={'train':18,'val':42}
    assert Counter(r['condition'] for r in new)=={'paired':12,'text_only':12,'physical_static':12,'imu_only':6,'numeric_zero':6,'backbone_bypass':6,'tail_reversed':6}
    for s,o in zip(new,old):
        assert s['comparison_case_id']==o['case_id'] and s['checkpoint']['sha256']==f.CHECKPOINT_SHA
        for k,v in o.items():
            if k not in ('arm','case_id','status','slot_index','checkpoint_selection'):assert s[k]==v
        assert s['guidance']==dict(text_scale=1.,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched')
        assert len(s['historical_input_hashes'])==11
        m=f.read(ROOT/s['historical_metadata']['path']);assert f.digest(m['random_plans'][0])==s['historical_random_plan_sha256']
    assert len(f.read(P/'sampling_noise_groups.json'))==12


def test_Q_exact_batch_branch_presence_and_same_noise():
    rows=f.read(P/'probe_slots.json');groups=f.read(P/'probe_groups.json');noise=f.read(P/'probe_noise.json')
    assert len(rows)==480 and len(groups)==len(noise)==96
    assert sum(r['maximum_encoder_examples'] for r in rows)==576
    for i,g in enumerate(groups):
        rs=rows[5*i:5*i+5];assert [r['branch'] for r in rs]==f.BRANCHES
        assert [r['probe_id'] for r in rs]==g['probe_ids']
        assert [(r['text_present'],r['imu_present']) for r in rs]==[(False,False),(True,False),(True,True),(True,True),(True,True)]
        assert all(r['noise_key']==g['group_id'] and r['expected_MDM_batch_shape']==[5,263,1,g['frames']] and r['context_frames']==r['frames'] for r in rs)
        assert noise[g['group_id']]['valid_noise']==f.read(f.OLD/'probe_noise.json')[g['group_id']]['valid_noise']
        assert noise[g['group_id']]['padding_noise'] is None
    assert Counter(r['timestep'] for r in groups)=={50:24,250:24,500:24,900:24}
    assert Counter(r['data_split'] for r in groups)=={'train':48,'val':48}


def test_trace_is_complete_no_extra_calls_and_predeclared_order():
    trace=f.read(P/'trace_slots.json');samples=f.read(P/'sampling_slots.json')
    assert len(trace)==len({r['trace_id'] for r in trace})==240
    for i,s in enumerate(samples):
        rs=trace[4*i:4*i+4];assert [r['timestep'] for r in rs]==[900,500,250,50]
        assert all(r['case_id']==s['case_id'] and r['shape']==[1,263,1,s['frames']] for r in rs)
        assert all(r['branch_order']==['f00','f10','f01','f11'] and r['additional_MDM_branch_examples']==0 for r in rs)


def test_budget_denominators_and_frozen_identity_checks():
    b=f.read(P/'budget.json');d=f.read(P/'design.json');e=f.read(P/'evaluation.json')
    assert b['authorized'] is False and b['gpu_uuid'] is b['run_directory'] is None
    assert b['diagnostics']['branch_examples']==480 and b['diagnostics']['maximum_seconds']==600
    assert b['sampling']['count']==60 and b['sampling']['maximum_seconds_shared']==2400
    assert b['totals']['maximum_seconds']==3600 and b['totals']['allocated_memory_bytes']==24*2**30
    assert b['totals']['MDM_branch_examples_max']==480+60*1000*4
    assert b['totals']['encoder_examples_max']==96*3*2+60*2
    assert d['software_tolerances']['full_sample']['max_abs_m']==1e-5
    assert len(d['software_tolerances']['full_sample']['comparisons'])==3
    assert 'ALL train' in e['contact'] and e['sample_denominators']['Q_derived_state_condition_pairs']==288


def test_review_contains_both_scales_and_all_cases_in_frozen_order():
    review=f.read(P/'review_order.json');samples=f.read(P/'sampling_slots.json');history=f.read(P/'historical_samples.json')
    population=[r['case_id'] for r in samples]+[r['comparison_case_id'] for r in history]
    order=np.random.Generator(np.random.PCG64(20260930)).permutation(120)
    assert len({r['review_id'] for r in review['rows']})==120
    assert [r['case_id'] for r in review['rows']]==[population[int(i)] for i in order]


def test_decomposition_exposes_helpful_and_harmful_text_terms():
    z=np.zeros((2,3));one=np.ones((2,3))
    harmful=f.reduced_analysis(z,one,one,z)
    assert harmful['unit_mse']==1 and harmful['legacy_reduced_mse']==6.25
    assert harmful['twice_error_text_inner_product']==3 and harmful['text_energy']==2.25
    helpful=f.reduced_analysis(z,-one,one,z)
    assert helpful['legacy_minus_unit']==-.75 and helpful['twice_error_text_inner_product']==-3
    assert harmful['decomposition_error']<=harmful['decomposition_bound']
    assert helpful['decomposition_error']<=helpful['decomposition_bound']


@pytest.mark.parametrize('scale',[1e-4,1.,1e5])
def test_float32_combination_rounding_bound_including_cancellation(scale):
    rng=np.random.Generator(np.random.PCG64(20260927));a,b,c,d=(rng.standard_normal((4,263,1,5))*scale).astype(np.float32)
    for wt in (1.,2.5):
        actual=a+np.float32(wt)*(b-a)+(c-a)+(d-b-c+a)
        reference=d.astype(float)+(wt-1)*(b.astype(float)-a)
        assert np.all(np.abs(actual-reference)<=f.combination_bound((a,b,c,d),wt))
    # IMU-only text-absent pair; same execution with zero text increment remains exactly invariant.
    b=a.copy()
    x=a+(b-a)+(c-a)+(d-b-c+a);y=a+np.float32(2.5)*(b-a)+(c-a)+(d-b-c+a)
    np.testing.assert_array_equal(x,y)


def test_Q_prediction_difference_cancels_text_but_MSE_contrast_can_change():
    z=np.zeros(2);one=np.ones(2)
    paired=f.reduced_analysis(z,one,one,z);static=f.reduced_analysis(z,one,2*one,z)
    assert (one+1.5*one-(2*one+1.5*one)==one-2*one).all()
    assert paired['legacy_reduced_mse']-static['legacy_reduced_mse'] != paired['unit_mse']-static['unit_mse']


def test_invalid_numeric_inputs_rejected_and_frozen_output_not_overwritten():
    with pytest.raises(ValueError):f.reduced_analysis(np.zeros(2),np.zeros(3),np.zeros(2),np.zeros(2))
    with pytest.raises(ValueError):f.reduced_analysis(np.zeros(2),np.array([np.nan,0]),np.zeros(2),np.zeros(2))
    with pytest.raises(FileExistsError):f.build(P)
