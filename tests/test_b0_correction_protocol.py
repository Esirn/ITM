"""Validate the frozen experiment before implementation; no model or GPU calls."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1'
OLD=ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3'
ARMS=['L_exact_length','R_static_reference_delta']
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def test_hash_bound_package_is_not_authorization():
    p=read(P/'package.json')
    assert p['execution_authorization'] is p['GPU_UUID'] is p['run_directory'] is None
    for a in p['artifacts']+read(P/'sources.json'):
        assert hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256']
    assert read(P/'design.json')['common']['training_input_provenance']==read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')['training_input_provenance']


def test_all_slots_and_original_noises_preserved():
    rows=read(P/'sampling_slots.json');old={r['case_id']:r for r in read(OLD/'sampling_cases.json')}
    assert len(rows)==len({r['case_id'] for r in rows})==120
    by_arm={a:[r for r in rows if r['arm']==a] for a in ARMS}
    for arm,rs in by_arm.items():
        assert Counter(r['split'] for r in rs)=={'development_val':42,'train_diagnostic':18}
        assert Counter(r['condition'] for r in rs)=={'paired':12,'text_only':12,'physical_static':12,'imu_only':6,'backbone_bypass':6,'numeric_zero':6,'tail_reversed':6}
        originals=[r for r in rs if r['baseline_case_id']]
        assert {r['baseline_case_id'] for r in originals}==set(old)
        for r in originals:
            o=old[r['baseline_case_id']]
            for k in ['random_plan_case_id','sampling_seed','length','caption_index','text_present','imu_present','gt_path']:assert r[k]==o[k]
    for l,r in zip(by_arm[ARMS[0]],by_arm[ARMS[1]]):
        for k in ['split','motion_id','sensor_config','condition','random_plan_case_id','sampling_seed','frames','gt_path']:assert l[k]==r[k]
    ids=defaultdict(set)
    for r in rows:ids[r['random_plan_case_id']].add((r['data_split'],r['motion_id'],r['sensor_config']))
    assert len(ids)==12 and all(len(v)==1 for v in ids.values())


def test_probe_denominators_and_matched_noise_addresses():
    rows=read(P/'probe_slots.json');noise=read(P/'probe_noise.json')
    assert len(rows)==len({r['probe_id'] for r in rows})==480
    assert Counter(r['context']=='exact_length' for r in rows)=={True:384,False:96}
    groups=defaultdict(list)
    for r in rows:groups[r['noise_key']].append(r)
    assert set(groups)==set(noise) and len(groups)==96
    for k,rs in groups.items():
        assert len(rs)==(6 if rs[0]['split']=='train' else 4)
        assert {r['condition'] for r in rs if r['context']=='exact_length'}=={'paired','physical_static','tail_reversed','text_only'}
        assert all(r['text_present'] for r in rs)
        for d in noise[k].values():
            if d is None:continue
            address={q:d[q] for q in ['version','backbone_id','case_id','sample_seed','step','shape']}
            h=digest(address);assert h==d['address_sha256']
            v=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(d['shape']).astype(np.float32)
            assert hashlib.sha256(v.tobytes()).hexdigest()==d['noise_sha256']
        if rs[0]['split']=='train':
            padded=[r for r in rs if r['context']!='exact_length']
            assert len(padded)==2 and all(r['context_frames']==158 for r in padded)
            assert (noise[k]['padding_noise'] is None)==(rs[0]['frames']==158)


def test_training_counts_and_noise_are_reproducible_without_arms():
    s=read(P/'training_schedule.json');rows=s['rows'];assert len(rows)==600
    assert set(Counter(r['update'] for r in rows))==set(range(1,201))
    assert set(Counter(r['update'] for r in rows).values())=={3}
    assert Counter((r['motion_id'],r['sensor_config']) for r in rows)==dict.fromkeys([tuple(x) for x in s['population_order']],100)
    for epoch in range(1,101):
        e=[r for r in rows if r['epoch']==epoch];assert len(e)==6
        assert {(r['motion_id'],r['sensor_config']) for r in e}=={tuple(x) for x in s['population_order']}
    assert Counter(f"{int(r['text_present'])}{int(r['imu_present'])}" for r in rows)==s['condition_counts']
    for r in rows:
        identity={'version':s['version'],'seed':s['seed'],'namespace':'training_noise','epoch':r['epoch'],
                  'position':r['position_in_epoch'],'motion_id':r['motion_id'],'sensor_config':r['sensor_config']}
        h=digest(identity);assert h==r['noise']['address_sha256']
        v=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(r['noise']['shape']).astype(np.float32)
        assert hashlib.sha256(v.tobytes()).hexdigest()==r['noise']['noise_sha256']
        assert r['noise']['shape']==[1,263,1,r['frames']] and 0<=r['timestep']<1000
        assert 0<=r['encoder_dropout_seed']<2**63 and 'arm' not in r


def test_resource_caps_and_no_promotion_to_effectiveness():
    b=read(P/'budget.json');d=read(P/'design.json');e=read(P/'evaluation.json')
    assert b['diagnostic']['maximum_branch_example_forwards']==480
    assert b['training']['total_updates']==400 and b['training']['maximum_MDM_branch_example_forwards']==1200
    assert b['sampling']['total_count']==120 and b['sampling']['maximum_denoising_calls']==120000
    assert b['sampling']['maximum_branch_example_forwards']==480000
    assert b['resource_caps']['total_seconds']==7200 and b['resource_caps']['maximum_allocated_memory_bytes']==24*1024**3
    assert b['diagnostic']['maximum_seconds']+b['training']['maximum_seconds_shared']+b['sampling']['maximum_seconds_shared']==7200
    assert d['common']['checkpoints']==[50,100,150,200] and d['common']['backbone']['mask_frames'] is False
    assert d['common']['optimizer']['foreach'] is d['common']['optimizer']['fused'] is None
    assert 'not empirical IMU advantage' in e['interpretation']['static_restoration']
    assert 'all train N/A' in e['split_rules']['foot_contact']


def test_review_permutation_prespecified_and_complete():
    slots=read(P/'sampling_slots.json');review=read(P/'review_order.json')
    rows=review['rows'];assert len(rows)==120 and len({r['review_id'] for r in rows})==120
    assert {r['case_id'] for r in rows}=={r['case_id'] for r in slots}
    order=np.random.Generator(np.random.PCG64(review['seed'])).permutation(120)
    assert [r['case_id'] for r in rows]==[slots[int(i)]['case_id'] for i in order]
