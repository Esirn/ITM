#!/usr/bin/env python
"""Owner-02 CPU-only evaluation of the one authorized canonical B0 run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from itm.data.humanml_torch import recover_from_ric_torch
from itm.metrics.canonical_b0 import evaluate_motion
from accept_canonical_b0_handoff import reference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3'
RUN = ROOT/'outputs/itm_rework/method/canonical_b0_authorized_20260926_v1'
DEFAULT = ROOT/'docs/itm_rework/protocol/b0_evaluation_20260926_v1'

def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def audit(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.json').exists():raise FileExistsError('Use a new output directory; preserve old evidence')
    cases=read(DATA/'sampling_cases.json'); assert len(cases)==36
    # Persist condition-order randomization before any output metrics are computed.
    order=np.random.Generator(np.random.PCG64(20260926)).permutation(36)
    write(out/'review_key.json',{'seed':20260926,'algorithm':'numpy PCG64 permutation of frozen slot list',
        'rows':[{'review_id':f'R{i+1:02d}','case_id':cases[int(j)]['case_id']} for i,j in enumerate(order)]})
    checked={}
    def check(a):
        p=Path(a['path']);p=p if p.is_absolute() else ROOT/p
        if str(p) not in checked:checked[str(p)]=sha(p)
        assert checked[str(p)]==a['sha256'],str(p)
    status=read(RUN/'runtime/status.json'); assert status['status']=='completed'
    auth=read(ROOT/'docs/itm_rework/method/B0_AUTHORIZATION_20260926.json')
    assert auth==status['authorization'] and auth['authorized'] and auth['run_directory']==str(RUN)
    for a in status['required_artifacts']+[status['plan'],status['training_receipt']]:check(a)
    assert status['plan']['sha256']==auth['execution_plan_sha256']
    assert sha(DATA/'b0_budget.json')==auth['budget_sha256']
    train=read(status['training_receipt']['path'])
    assert train['status']=='completed' and train['updates_completed']==train['updates_attempted']==200
    finalsha=sha(RUN/'model.pt')
    for a in train['artifacts']:check(a)
    state=torch.load(RUN/'model.pt',map_location='cpu',weights_only=True)
    accepted=read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
    provenance=state['training_input_provenance']; assert provenance==accepted['training_input_provenance']
    assert digest(provenance['payload'])==provenance['sha256']
    assert state['execution']['final_update_200'] is True
    norm=state['imu_normalization']; assert norm==accepted['imu_normalization']
    for a in provenance['payload']['artifacts'].values():check(a)
    for r in provenance['payload']['selected_inputs']:
        for k in ('motion','text','imu'):check(r[k])
    refs={}; plans={}
    for split in ('train','val'):
        plans[split]={r['motion_id']:r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
        for r in lines(DATA/f'{split}_imu.jsonl'):
            plan=plans[split][r['motion_id']]
            for a in plan['artifacts']:check(a)
            refs[split,r['motion_id']]=reference(r,plan['frames'])[:3]
    population=np.concatenate([refs['train',mid][0].astype(float) for mid in ['000027','000506','004728']])
    assert population.shape==(398,6,3)
    mean=np.asarray(norm['acceleration_mean'],np.float32);std=np.asarray(norm['acceleration_std'],np.float32)
    np.testing.assert_array_equal(population.mean(0).astype(np.float32),mean)
    np.testing.assert_allclose(np.sqrt(np.maximum(population.var(0),1e-8)).astype(np.float32),std,rtol=1e-7,atol=0)
    motion_mean=np.load(accepted['training_args']['mean']);motion_std=np.load(accepted['training_args']['std'])
    rows=[];groups={};predictions={};receipts=[train];replayed=0
    slots={s['case_id']:s for s in status['slots']};assert set(slots)=={c['case_id'] for c in cases}
    for case in cases:
        cid=case['case_id']; row={k:case[k] for k in ('case_id','motion_id','sensor_config','condition','length')}
        try:
            slot=slots[cid];assert slot['status']=='completed' and slot['returncode']==0
            for a in slot['artifacts']+[slot['receipt']]:check(a)
            receipt=read(slot['receipt']['path']);receipts.append(receipt)
            assert receipt['status']=='completed' and receipt['denoising_calls']==1000 and receipt['branch_examples']==4000
            assert receipt['gpu_uuid']==auth['gpu_uuid']
            meta=read(Path(slot['output']).with_suffix('.json'))
            assert meta['control_checkpoint_sha256']==finalsha and meta['training_input_provenance']==provenance
            check({'path':meta['mdm_checkpoint'],'sha256':meta['mdm_checkpoint_sha256']})
            assert meta['steps']==meta['model_forward_calls']==1000 and meta['sampler']=='ddpm'
            assert meta['seed']==20260926 and meta['training_seed']==20260925
            assert (meta['text_scale'],meta['imu_scale'],meta['joint_scale'])==(2.5,1,1)
            actual=meta['cases'][0]; assert len(meta['cases'])==1
            for k in ('case_id','motion_id','sensor_config','condition','caption_index','text_present','imu_present','length','sampling_seed','random_plan_case_id','gt_path'):assert actual[k]==case[k],k
            assert actual['text']==plans['val'][case['motion_id']]['caption']
            assert actual['backbone_bypass']==(case['condition']=='backbone_bypass')
            for a in actual['input_metadata']['artifacts']:check(a)
            n=case['length'];gt=np.load(case['gt_path']);assert gt.shape==(n,22,3)
            with np.load(slot['output'],allow_pickle=False) as z:
                assert json.loads(str(z['metadata']))==meta
                for k in z.files:
                    if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all(),k
                p=z['motion'][0].copy();assert z['motion'].shape==(1,n,22,3)
                assert z['normalized_humanml263'].shape==(1,263,1,n)
                np.testing.assert_array_equal(z['gt'],gt[None])
                np.testing.assert_array_equal(z['frame_mask'],np.ones((1,n),bool))
                np.testing.assert_array_equal(z['timestamps_seconds'],np.arange(n)/20)
                assert float(z['fps'])==20 and meta['frame_count']==n
                assert meta['original_lengths']==meta['effective_lengths']==[n]
                for k in ('text','imu'):np.testing.assert_array_equal(z[k+'_condition_present'],[case[k+'_present']])
                mask=np.zeros((1,6),bool);mask[0,[4] if case['sensor_config']=='head' else [0,1]]=True
                np.testing.assert_array_equal(z['sensor_mask'],mask)
                a,o,_=refs['val',case['motion_id']]
                if case['condition']=='physical_static':a=np.zeros_like(a);o=np.broadcast_to(o[:1],o.shape)
                expected=np.concatenate([(a-mean)/std,o.reshape(n,6,9)],-1)
                if case['condition']=='numeric_zero':expected=np.zeros_like(expected)
                np.testing.assert_array_equal(z['acceleration'][0],a)
                np.testing.assert_allclose(z['orientation'][0],o,atol=2e-7,rtol=0)
                np.testing.assert_allclose(z['normalized_imu'][0],expected,atol=2e-7,rtol=0)
                features=z['normalized_humanml263'][0,:,0,:].T*motion_std+motion_mean
                decoded=recover_from_ric_torch(torch.from_numpy(features)).numpy()
                decode_error=float(np.max(np.abs(decoded-p)))
                # Arithmetic replay allowance, not a model-quality threshold; retain continuous error.
                np.testing.assert_allclose(decoded,p,rtol=1e-6,atol=2e-6)
            group=case['random_plan_case_id'];plan=meta['random_plans'][0]
            assert len(meta['random_plans'])==1
            if group in groups:assert groups[group]==plan
            else:
                draws=plan['noise_draws'];assert len(draws)==1000
                assert [d['step'] for d in draws]==[-1]+list(range(999,0,-1))
                assert plan['timestep_map']==list(range(1000)) and plan['clip_denoised'] is False
                for d in draws:
                    address={k:d[k] for k in ('version','backbone_id','case_id','sample_seed','step','shape')}
                    assert address['version']=='itm-pcg64-addressed-noise-v1'
                    assert address['backbone_id']==meta['mdm_checkpoint_sha256']
                    assert address['case_id']==group and address['sample_seed']==20260926 and address['shape']==[1,263,1,n]
                    h=digest(address);assert h==d['address_sha256']
                    noise=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(address['shape']).astype(np.float32)
                    assert hashlib.sha256(noise.tobytes()).hexdigest()==d['noise_sha256']
                    replayed+=1
                assert digest(draws)==plan['random_plan_id']
                groups[group]=plan
            contacts=None if case['motion_id']=='000139' else np.load(Path(case['gt_path']).with_name('motion.npy'))[:,259:263]
            row.update(status='valid',decode_max_abs_m=decode_error,metrics=evaluate_motion(p,gt,case['sensor_config'],contacts),
                       artifacts=slot['artifacts'],random_plan_id=plan['random_plan_id'])
            predictions[cid]=p
        except (AssertionError,ValueError,KeyError,OSError) as e:
            row.update(status='failure',reason=type(e).__name__+': '+str(e))
        rows.append(row)
    null=[]
    for mid in ('000433','000382','000139'):
        for config in ('head','wrists'):
            a,b=(f'{mid}-{config}-{c}' for c in ('text_only','backbone_bypass'))
            if a in predictions and b in predictions:
                null.append({'motion_id':mid,'config':config,'max_abs_m':float(np.abs(predictions[a]-predictions[b]).max()),
                             'bitwise_equal':bool(np.array_equal(predictions[a],predictions[b]))})
    aggregate=[]
    for config in ('head','wrists'):
        for condition in ('paired','text_only','imu_only','backbone_bypass','numeric_zero','physical_static'):
            valid=[r for r in rows if r['sensor_config']==config and r['condition']==condition and r['status']=='valid']
            values={}
            if valid:
                for k,v in valid[0]['metrics'].items():
                    if isinstance(v,(int,float)) or v is None:
                        xs=[r['metrics'][k] for r in valid if r['metrics'][k] is not None]
                        values[k]={'source_groups':len(xs),'equal_source_mean':float(np.mean(xs)) if xs else None}
            aggregate.append({'config':config,'condition':condition,'planned':3,'valid':len(valid),'metrics':values})
    assert all(r['peak_allocated_bytes']<=auth['limits']['allocated_memory_bytes'] for r in receipts)
    assert all(r['gpu_uuid']==auth['gpu_uuid'] for r in receipts)
    assert status['denoising_calls']==36000 and status['branch_examples']==144000
    start=status['total_deadline_monotonic']-14400;end=max(r['last_monotonic'] for r in receipts)
    assert end-start<=14400 and train['last_monotonic']-start<=3600
    assert end-(status['sampling_deadline_monotonic']-10800)<=10800
    report={'version':'independent-canonical-B0-evaluation-v1','read_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'protocol':{'path':str(DATA/'metrics.json'),'sha256':sha(DATA/'metrics.json')},
            'execution_scope':'CPU evaluation/reconstruction only; no model forward, training, sampling or GPU job',
            'checkpoint_sha256':finalsha,'training_provenance_sha256':provenance['sha256'],'updates':200,
            'normalization_population_frames':398,'noise_draws_replayed':replayed,'noise_groups':len(groups),
            'planned_slots':36,'valid_slots':sum(r['status']=='valid' for r in rows),
            'failure_rate':sum(r['status']=='failure' for r in rows)/36,
            'max_allocated_bytes':max(r['peak_allocated_bytes'] for r in receipts),'receipt_elapsed_seconds':end-start,
            'strict_null_vs_backbone':null,'rows':rows,'aggregates':aggregate}
    write(out/'evaluation.json',report)
    checked[str(RUN/'model.pt')]=finalsha
    write(out/'artifact_audit.json',{'verified_files':[{'path':p,'sha256':s} for p,s in sorted(checked.items())],
        'code':[{'path':str(p),'sha256':sha(p)} for p in [Path(__file__),ROOT/'src/itm/metrics/canonical_b0.py',ROOT/'src/itm/data/humanml_torch.py',ROOT/'scripts/accept_canonical_b0_handoff.py']],
        'runtime_status_sha256':sha(RUN/'runtime/status.json')})
    print(json.dumps({k:report[k] for k in ('valid_slots','failure_rate','noise_draws_replayed')}))
    if report['valid_slots']!=36:
        print(json.dumps([r for r in rows if r['status']=='failure'],indent=2));raise SystemExit(1)


def compare(out):
    """Descriptive, source-paired differences; no new thresholds or statistical tests."""
    report=read(out/'evaluation.json');rows={r['case_id']:r for r in report['rows']}
    keys=['global_MPJPE_m','root_relative_MPJPE_m','root_path_RMSE_m',
          'head_or_wrists_position_joint_proxy_m','velocity_joint_proxy_RMSE_m_s',
          'jerk_absolute_m_s3','foot_contact_speed_proxy_m_s']
    comparisons=[]
    for mid in ('000433','000382','000139'):
        for config in ('head','wrists'):
            cid=f'{mid}-{config}-paired';row=rows[cid]
            if row['status']!='valid':continue
            with np.load(RUN/'samples'/f'{cid}.npz') as z:a=z['motion'][0].astype(float)
            for condition in ('text_only','numeric_zero','physical_static','imu_only'):
                other=f'{mid}-{config}-{condition}';baseline=rows[other]
                if baseline['status']!='valid':continue
                with np.load(RUN/'samples'/f'{other}.npz') as z:b=z['motion'][0].astype(float)
                comparisons.append({'paired_case':cid,'comparator_case':other,
                    'metric_difference_paired_minus_comparator':{
                        k:row['metrics'][k]-baseline['metrics'][k] if row['metrics'][k] is not None else None for k in keys},
                    'output_to_output_mean_joint_distance_m':float(np.linalg.norm(a-b,axis=-1).mean()),
                    'output_to_output_max_abs_m':float(np.abs(a-b).max())})
    write(out/'paired_comparisons.json',comparisons)
    # A flat per-slot table makes all 36 results directly inspectable, including failures.
    import csv
    with (out/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','status','frames']+keys+['GT_jerk_absolute_m_s3',
            'GT_foot_contact_speed_proxy_m_s','foot_contact_stencils','pelvis_path_length_m','GT_pelvis_path_length_m'])
        writer.writeheader()
        for row in report['rows']:
            flat={'case_id':row['case_id'],'status':row['status'],'frames':row['length']}
            if row['status']=='valid':
                m=row['metrics'];flat.update({k:m[k] for k in keys+['GT_jerk_absolute_m_s3','GT_foot_contact_speed_proxy_m_s','foot_contact_stencils']})
                flat['pelvis_path_length_m']=m['motion_extent_m']['pelvis_path_length_m']
                flat['GT_pelvis_path_length_m']=m['GT_motion_extent_m']['pelvis_path_length_m']
            writer.writerow(flat)
    print(f'Saved {len(comparisons)} source-paired comparisons and all {len(rows)} rows')


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    out_images=ROOT/'outputs/itm_rework/protocol/b0_evaluation_20260926_v1/review'
    out_images.mkdir(parents=True,exist_ok=True)
    key=read(out/'review_key.json')['rows'];cases={c['case_id']:c for c in read(DATA/'sampling_cases.json')}
    chains=[([0,1,4,7,10],'#1479c4'),([0,2,5,8,11],'#d44a38'),([0,3,6,9,12,15],'#555555'),([9,13,16,18,20],'#1479c4'),([9,14,17,19,21],'#d44a38')]
    actions={'000433':'kick then stand','000382':'repeated hand approach/separation then arms down','000139':'stand/sway then forward steps + arm swing'}
    artifacts=[]
    for page in range(6):
        fig=plt.figure(figsize=(24,18));gs=GridSpec(6,13,figure=fig,hspace=.65,wspace=.65)
        for row,item in enumerate(key[page*6:page*6+6]):
            case=cases[item['case_id']];p=np.load(RUN/'samples'/f"{case['case_id']}.npz")['motion'][0];t=np.arange(len(p))/20
            centered=p.copy();centered[:,:,0]-=p[:,0:1,0];centered[:,:,2]-=p[:,0:1,2]
            for col,f in enumerate(np.linspace(0,len(p)-1,10).round().astype(int)):
                ax=fig.add_subplot(gs[row,col]);pose=centered[f]
                # Common oblique projection, fixed axes for all outputs; horizontal root travel on separate plot.
                for ids,color in chains:ax.plot(pose[ids,0]+.55*pose[ids,2],pose[ids,1]+.10*pose[ids,2],color=color,lw=1.4)
                ax.set(xlim=(-1.4,1.4),ylim=(-.35,2.3));ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{f/20:.2f}s',fontsize=8)
                if col==0:ax.text(0,1.25,item['review_id']+' | '+actions[case['motion_id']],transform=ax.transAxes,fontsize=10,weight='bold')
            ax=fig.add_subplot(gs[row,10]);ax.plot(t,np.linalg.norm(p[:,20]-p[:,21],axis=-1),label='wrist sep');ax.plot(t,p[:,20,1]-p[:,0,1],label='L wrist-pelvis');ax.plot(t,p[:,21,1]-p[:,0,1],label='R wrist-pelvis');ax.set_title('Hands (m)',fontsize=9);ax.legend(fontsize=5)
            ax=fig.add_subplot(gs[row,11]);ax.plot(t,p[:,7,1],label='L ankle');ax.plot(t,p[:,8,1],label='R ankle');ax.plot(t,p[:,0,1],label='pelvis');ax.set_title('Height (m)',fontsize=9);ax.legend(fontsize=5)
            ax=fig.add_subplot(gs[row,12]);ax.plot(p[:,0,0],p[:,0,2]);ax.scatter(*p[0,0,[0,2]],s=15,c='green');ax.scatter(*p[-1,0,[0,2]],s=15,c='red');ax.set_title('Root X/Z (m)',fontsize=9);ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle(f'B0 condition-hidden review {page+1}/6 | left blue / right red | poses root-XZ-centered only for display | traces use every frame',fontsize=12)
        path=out_images/f'page_{page+1}.png';fig.savefig(path,dpi=110,bbox_inches='tight');plt.close(fig)
        artifacts.append({'path':str(path),'sha256':sha(path)})
    write(out/'review_rendering.json',{'images':artifacts,'review_key_sha256':sha(out/'review_key.json'),
        'scope':'10 uniform pose snapshots plus all-frame hand/ankle/root traces; condition/config hidden, coarse target action shown. Root XZ centering only for visualization, never metrics. Single AI reviewer; no inter-rater certification.'})
    print('Rendered 6 pages / 36 outputs; mapping remains in review_key.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['audit','render','compare']);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();{'audit':audit,'render':render,'compare':compare}[args.mode](args.output)
