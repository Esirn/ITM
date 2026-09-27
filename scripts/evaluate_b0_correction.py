#!/usr/bin/env python
"""Owner02: saved-array-only evaluation of the frozen D/L/R experiment."""
import argparse
from collections import defaultdict
import csv
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
FROZEN = ROOT/'docs/itm_rework/protocol/b0_correction_20260926_v1'
DATA = ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3'
RUN = ROOT/'outputs/itm_rework/method/correction_authorized_20260927_v2'
OLD = ROOT/'outputs/itm_rework/method/canonical_b0_authorized_20260926_v1'
OUT = ROOT/'docs/itm_rework/protocol/b0_correction_evaluation_20260927_v1'
IMAGES = ROOT/'outputs/itm_rework/protocol/b0_correction_evaluation_20260927_v1/review'
ARMS = ('L_exact_length', 'R_static_reference_delta')
KEYS = ('global_MPJPE_m','root_relative_MPJPE_m','root_path_RMSE_m',
        'head_or_wrists_position_joint_proxy_m','velocity_joint_proxy_RMSE_m_s',
        'jerk_absolute_m_s3','GT_jerk_absolute_m_s3','foot_contact_speed_proxy_m_s',
        'GT_foot_contact_speed_proxy_m_s')

def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
    return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return dict(count=a.size,finite_count=int(np.isfinite(a).sum()),sums_sq=float(np.square(a).sum()),
                rms=float(np.sqrt(np.square(a).mean())),max_abs=float(np.abs(a).max()))
def metric_delta(a,b):return {k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in KEYS}
def distance(a,b):return dict(mean_joint_distance_m=float(np.linalg.norm(a.astype(float)-b,axis=-1).mean()),
                              max_abs_m=float(np.abs(a.astype(float)-b).max()),bitwise_equal=bool(np.array_equal(a,b)))
def averages(rows,group_keys,value_key='metrics'):
    """Each source contributes once; missing/NA never silently becomes zero."""
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in group_keys)].append(row)
    result=[]
    for group,items in sorted(grouped.items()):
        valid=[r for r in items if r.get('status','valid')=='valid']
        metrics={}
        for k in KEYS:
            xs=[r[value_key][k] for r in valid if r[value_key][k] is not None]
            metrics[k]=dict(n=len(xs),mean=float(np.mean(xs)) if xs else None,
                            min=float(min(xs)) if xs else None,max=float(max(xs)) if xs else None)
        result.append(dict(zip(group_keys,group),planned=len(items),valid=len(valid),
                           missing=[r.get('case_id') for r in items if r not in valid],
                           partial=len(valid)!=len(items),metrics=metrics))
    return result

def reconstruct_noise(draw):
    address={k:draw[k] for k in ('version','backbone_id','case_id','sample_seed','step','shape')}
    h=digest(address);assert h==draw['address_sha256']
    noise=np.random.Generator(np.random.PCG64(int(h,16))).standard_normal(address['shape']).astype(np.float32)
    assert ah(noise)==draw['noise_sha256']
    return noise

def render():
    """No metric/condition labels in images; order was frozen before generation."""
    OUT.mkdir(parents=True,exist_ok=True);IMAGES.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    rows=read(FROZEN/'review_order.json')['rows'];cases={r['case_id']:r for r in read(FROZEN/'sampling_slots.json')}
    actions={'000433':'kick, return to standing','000382':'clap, lower arms','000139':'stand/sway, then walk',
             '000506':'alternating forward lunges/squats','004728':'RIGHT leg back lift, hold, lower','000027':'punch, then turn'}
    chains=[([0,1,4,7,10],'#1479c4'),([0,2,5,8,11],'#d44a38'),([0,3,6,9,12,15],'#555555'),([9,13,16,18,20],'#1479c4'),([9,14,17,19,21],'#d44a38')]
    artifacts=[]
    for page in range(20):
        p=IMAGES/f'page_{page+1:02}.png'
        if p.exists():
            artifacts.append(dict(path=str(p),sha256=sha(p)));continue
        fig=plt.figure(figsize=(24,18));gs=GridSpec(6,13,figure=fig,hspace=.65,wspace=.65)
        for row,item in enumerate(rows[page*6:page*6+6]):
            c=cases[item['case_id']]
            with np.load(RUN/'samples'/f"{c['case_id']}.npz") as z:p=z['motion'][0]
            t=np.arange(len(p))/20;centered=p.copy();centered[:,:,0]-=p[:,0:1,0];centered[:,:,2]-=p[:,0:1,2]
            for col,f in enumerate(np.linspace(0,len(p)-1,10).round().astype(int)):
                ax=fig.add_subplot(gs[row,col]);pose=centered[f]
                for ids,color in chains:ax.plot(pose[ids,0]+.55*pose[ids,2],pose[ids,1]+.1*pose[ids,2],color=color,lw=1.5)
                ax.set(xlim=(-1.4,1.4),ylim=(-.35,2.3));ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{f/20:.2f}s',fontsize=8)
                if col==0:ax.text(0,1.25,item['review_id']+' | '+actions[c['motion_id']],transform=ax.transAxes,fontsize=10,weight='bold')
            ax=fig.add_subplot(gs[row,10]);ax.plot(t,np.linalg.norm(p[:,20]-p[:,21],axis=-1),label='wrist sep');ax.plot(t,p[:,20,1]-p[:,0,1],label='L wrist-pelvis');ax.plot(t,p[:,21,1]-p[:,0,1],label='R wrist-pelvis');ax.set_title('Hands (m)',fontsize=9);ax.legend(fontsize=5)
            ax=fig.add_subplot(gs[row,11]);ax.plot(t,p[:,7,1],label='L ankle');ax.plot(t,p[:,8,1],label='R ankle');ax.plot(t,p[:,0,1],label='pelvis');ax.set_title('Height (m)',fontsize=9);ax.legend(fontsize=5)
            ax=fig.add_subplot(gs[row,12]);ax.plot(p[:,0,0],p[:,0,2]);ax.scatter(*p[0,0,[0,2]],s=15,c='green');ax.scatter(*p[-1,0,[0,2]],s=15,c='red');ax.set_title('Root X/Z (m)',fontsize=9);ax.set_aspect('equal',adjustable='datalim')
        fig.suptitle(f'Frozen-order correction review {page+1}/20 | L blue / R red | root XZ centered for display only | full-frame traces',fontsize=12)
        p=IMAGES/f'page_{page+1:02}.png';fig.savefig(p,dpi=100,bbox_inches='tight');plt.close(fig)
        artifacts.append(dict(path=str(p),sha256=sha(p)));print(f'rendered {page+1}/20',flush=True)
    write(OUT/'review_rendering.json',dict(images=artifacts,order_sha256=sha(FROZEN/'review_order.json'),
        scope='Arm/config/condition hidden on figures. Single AI reviewer; snapshots plus all-frame traces. Mapping file partially inspected during schema discovery; not a fully blinded independent human assessment.'))

def samples():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'samples.json').exists(),'Preserve previous results'
    frozen=read(FROZEN/'sampling_slots.json');status=read(RUN/'runtime/status.json')
    slots={r['case_id']:r for r in status['samples']};assert set(slots)=={r['case_id'] for r in frozen}
    implementation=read(status['implementation']['path']);checked={}
    def check(entry):
        p=ROOT/entry['path']
        if str(p) not in checked:checked[str(p)]=sha(p)
        assert checked[str(p)]==entry['sha256'],str(p)
    check(status['implementation'])
    for a in implementation['expected_artifacts']+[implementation['initialization']['artifact']]:check(a)
    design=read(FROZEN/'design.json');norm=design['common']['imu_normalization']
    mean=np.array(norm['acceleration_mean'],np.float32);std=np.array(norm['acceleration_std'],np.float32)
    accepted=read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
    mm=np.load(accepted['training_args']['mean']);ms=np.load(accepted['training_args']['std'])
    refs={};captions={}
    for split in ('train','val'):
        captions[split]={r['motion_id']:r for r in lines(DATA/f'{split}_caption_plans.jsonl')}
        for r in lines(DATA/f'{split}_imu.jsonl'):refs[split,r['motion_id']]=reference(r,captions[split][r['motion_id']]['frames'])[:3]
    population=np.concatenate([refs['train',mid][0].astype(float) for mid in ('000027','000506','004728')]);assert len(population)==398
    np.testing.assert_array_equal(population.mean(0).astype(np.float32),mean)
    np.testing.assert_allclose(np.sqrt(np.maximum(population.var(0),1e-8)).astype(np.float32),std,atol=0,rtol=1e-7)
    checkpoints={};training=[];schedule=read(FROZEN/'training_schedule.json')['rows']
    for arm in ARMS:
        state=torch.load(RUN/arm/'model.pt',map_location='cpu',weights_only=True)
        assert state['correction_arm']==arm and state['execution']['final_update_200'] and state['execution']['updates_completed']==200
        assert state['initialization_sha256']==implementation['initialization']['artifact']['sha256']
        assert state['training_input_provenance']==accepted['training_input_provenance'] and state['imu_normalization']==norm
        assert state['environment']==implementation['environment']
        checkpoints[arm]=sha(RUN/arm/'model.pt')
        updates=[read(p) for p in sorted((RUN/arm).glob('update_*.json'))];assert len(updates)==200
        for i,u in enumerate(updates):
            assert u['update']==i+1 and u['optimizer_step_completed'] and len(u['examples'])==3
            for j,ex in enumerate(u['examples']):
                plan=schedule[i*3+j];assert all(ex[k]==v for k,v in plan.items())
                draw=plan['noise']
                address=dict(version='itm-B0-correction-address-v1',seed=20260925,namespace='training_noise',epoch=plan['epoch'],position=plan['position_in_epoch'],motion_id=plan['motion_id'],sensor_config=plan['sensor_config'])
                assert digest(address)==draw['address_sha256']
                eps=np.random.Generator(np.random.PCG64(int(draw['address_sha256'],16))).standard_normal(draw['shape']).astype(np.float32)
                assert ex['actual_noise_sha256']==draw['noise_sha256']==ah(eps)
                assert ex['actual_shape']==[1,263,1,plan['frames']]
            training.append(dict(arm=arm,update=i+1,losses=u['losses'],mean_loss=float(np.mean(u['losses']))))
    rows=[];predictions={};plans={};replayed=0;historical=[]
    for c in frozen:
        cid=c['case_id'];row={k:c[k] for k in ('case_id','arm','data_split','motion_id','sensor_config','condition','frames')}
        try:
            slot=slots[cid];assert slot['spec']==c and slot['status']=='completed'
            for a in slot['artifacts']+[slot['receipt']]:check(a)
            m=read(RUN/slot['output'].replace('.npz','.json'))
            for k,v in c.items():
                if k!='status':assert m[k]==v,k
            assert m['control_checkpoint_sha256']==checkpoints[c['arm']]
            assert m['training_input_provenance']==accepted['training_input_provenance']
            assert m['initialization_sha256']==implementation['initialization']['artifact']['sha256']
            assert m['package_sha256']==sha(FROZEN/'package.json')
            assert m['guidance']==dict(text_scale=2.5,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched')
            assert m['sampler']=='ddpm' and m['model_forward_calls']==1000 and m['MDM_branch_examples']==4000
            assert m['reference_mode']==('none' if c['arm']==ARMS[0] else 'static_delta')
            n=c['frames'];gt=np.load(c['gt_path']);assert gt.shape==(n,22,3)
            a,o,t=refs[c['data_split'],c['motion_id']];so=np.broadcast_to(o[:1],o.shape)
            ref=np.concatenate([(np.zeros_like(a)-mean)/std,so.reshape(n,6,9)],-1)
            ea,eo=a,o
            if c['condition']=='physical_static':ea,eo=np.zeros_like(a),so
            if c['condition']=='tail_reversed':index=np.r_[0,np.arange(n-1,0,-1)];ea,eo=a[index],o[index]
            u=np.concatenate([(ea-mean)/std,eo.reshape(n,6,9)],-1)
            if c['condition']=='numeric_zero':u=np.zeros_like(u)
            with np.load(RUN/slot['output'],allow_pickle=False) as z:
                assert json.loads(str(z['metadata']))==m
                for k in z.files:
                    if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all(),k
                p=z['motion'][0];assert z['motion'].shape==(1,n,22,3) and z['normalized_humanml263'].shape==(1,263,1,n)
                np.testing.assert_array_equal(z['gt'],gt[None]);np.testing.assert_array_equal(z['frame_mask'],np.ones((1,n),bool))
                np.testing.assert_array_equal(z['timestamps_seconds'],t);assert float(z['fps'])==20
                assert m['frame_count']==n and m['original_lengths']==m['effective_lengths']==[n]
                mask=np.zeros((1,6),bool);mask[0,c['sensor_slots']]=True;np.testing.assert_array_equal(z['sensor_mask'],mask)
                for k in ('text','imu'):np.testing.assert_array_equal(z[k+'_condition_present'],[c[k+'_present']])
                np.testing.assert_array_equal(z['acceleration'][0],ea)
                for k,v in [('orientation',eo),('normalized_imu',u),('normalized_reference_imu',ref)]:np.testing.assert_allclose(z[k][0],v,atol=2e-7,rtol=0)
                np.testing.assert_allclose(z['original_orientation0'],o[0],atol=2e-7,rtol=0)
                assert m['text']==captions[c['data_split']][c['motion_id']]['caption']
                decoded=recover_from_ric_torch(torch.from_numpy(z['normalized_humanml263'][0,:,0,:].T*ms+mm)).numpy()
                decode=float(np.abs(decoded-p).max());np.testing.assert_allclose(decoded,p,atol=2e-6,rtol=1e-6)
            group=c['random_plan_case_id'];plan=m['random_plans'][0];assert len(m['random_plans'])==1
            if group in plans:assert plans[group]==plan
            else:
                draws=plan['noise_draws'];assert len(draws)==1000 and [d['step'] for d in draws]==[-1]+list(range(999,0,-1))
                assert plan['timestep_map']==list(range(1000)) and plan['clip_denoised'] is False
                for d in draws:
                    assert d['case_id']==group and d['shape']==[1,263,1,n] and d['sample_seed']==20260926
                    assert d['backbone_id']==m['mdm_checkpoint_sha256']
                    reconstruct_noise(d);replayed+=1
                assert digest(draws)==plan['random_plan_id'];plans[group]=plan
            contacts=np.load(Path(c['gt_path']).with_name('motion.npy'))[:,259:263] if c['data_split']=='val' and c['motion_id'] in ('000433','000382') else None
            metrics=evaluate_motion(p,gt,c['sensor_config'],contacts)
            if c['data_split']=='train':metrics['foot_status']='not_applicable_train_floor_admission_not_frozen'
            row.update(status='valid',decode_max_abs_m=decode,metrics=metrics,random_plan_id=plan['random_plan_id'])
            predictions[cid]=p
            if c['baseline_case_id']:
                oldfile=OLD/'samples'/f"{c['baseline_case_id']}.npz"
                with np.load(oldfile) as z:oldp=z['motion'][0];oldmeta=json.loads(str(z['metadata']))
                assert oldmeta['random_plans'][0]==plan
                historical.append(dict(case_id=cid,old_file=str(oldfile),old_sha256=sha(oldfile),
                    metrics=metric_delta(metrics,evaluate_motion(oldp,gt,c['sensor_config'],contacts)),**distance(p,oldp)))
        except (AssertionError,ValueError,KeyError,OSError) as e:row.update(status='failure',reason=type(e).__name__+': '+str(e))
        rows.append(row)
    write(OUT/'samples.json',dict(planned=120,valid=sum(r['status']=='valid' for r in rows),rows=rows,
        aggregates=averages(rows,('arm','data_split','sensor_config','condition')),checkpoints=checkpoints,
        denominators=[dict(arm=arm,data_split=split,planned=sum(r['arm']==arm and r['data_split']==split for r in rows),valid=sum(r['arm']==arm and r['data_split']==split and r['status']=='valid' for r in rows)) for arm in ARMS for split in ('train','val')],
        normalization_population=398,noise_groups=len(plans),noise_draws_replayed=replayed,
        max_decode_error_m=max((r.get('decode_max_abs_m',0) for r in rows),default=None)))
    write(OUT/'training.json',training);write(OUT/'historical_comparisons.json',historical)
    write(OUT/'sample_artifact_audit.json',dict(files=[dict(path=p,sha256=h) for p,h in sorted(checked.items())],
        read_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256=sha(__file__)))
    comparisons(rows,predictions)
    print(json.dumps(dict(valid=sum(r['status']=='valid' for r in rows),planned=120,noise_draws=replayed,
                          failures=[r for r in rows if r['status']=='failure'])),flush=True)

def comparisons(rows,predictions):
    index={(r['arm'],r['data_split'],r['motion_id'],r['sensor_config'],r['condition']):r for r in rows}
    within=[];between=[];did=[];identities=[]
    def pair(a,b):
        result=dict(case_id=a['case_id'],comparator_case_id=b['case_id'])
        if a['status']!='valid' or b['status']!='valid':return dict(result,status='failure',reason='missing valid member')
        return dict(result,status='valid',metrics=metric_delta(a['metrics'],b['metrics']),**distance(predictions[a['case_id']],predictions[b['case_id']]))
    for key,a in index.items():
        arm,split,mid,config,condition=key
        group=dict(arm=arm,data_split=split,motion_id=mid,sensor_config=config)
        if condition=='paired':
            for base in ['text_only','physical_static']+(['tail_reversed','numeric_zero'] if split=='val' else []):
                b=index[arm,split,mid,config,base];within.append(dict(group,comparator=base,**pair(a,b)))
        if arm==ARMS[1]:
            b=index[ARMS[0],split,mid,config,condition];between.append(dict(data_split=split,motion_id=mid,sensor_config=config,condition=condition,**pair(a,b)))
        if condition=='backbone_bypass' or (arm==ARMS[1] and condition=='physical_static'):
            b=index[arm,split,mid,config,'text_only'];identities.append(dict(group,condition=condition,**pair(a,b)))
    w={(r['arm'],r['data_split'],r['motion_id'],r['sensor_config'],r['comparator']):r for r in within}
    for k,r in w.items():
        if k[0]!=ARMS[1] or r['comparator']=='numeric_zero':continue
        l=w[(ARMS[0],*k[1:])]
        did.append(dict(data_split=k[1],motion_id=k[2],sensor_config=k[3],comparator=k[4],
            status='valid' if r['status']==l['status']=='valid' else 'failure',
            metrics=metric_delta(r['metrics'],l['metrics']) if r['status']==l['status']=='valid' else {},
            output_distance_difference_m=r.get('mean_joint_distance_m',0)-l.get('mean_joint_distance_m',0)))
    write(OUT/'contrasts.json',dict(within=within,R_minus_L=between,difference_of_contrasts=did,identities=identities,
        within_aggregates=averages(within,('arm','data_split','sensor_config','comparator')),
        between_aggregates=averages(between,('data_split','sensor_config','condition')),
        difference_aggregates=averages(did,('data_split','sensor_config','comparator'))))
    with (OUT/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case_id','status',*KEYS,'pelvis_path_length_m','GT_pelvis_path_length_m'])
        writer.writeheader()
        for r in rows:
            d={k:r[k] for k in ('case_id','status')}
            if r['status']=='valid':
                m=r['metrics'];d.update({k:m[k] for k in KEYS});d.update(pelvis_path_length_m=m['motion_extent_m']['pelvis_path_length_m'],GT_pelvis_path_length_m=m['GT_motion_extent_m']['pelvis_path_length_m'])
            writer.writerow(d)

def probe_aggregate(rows, group_keys):
    # Two replicates first, then equal sources. Each timestep/hook statistic remains separate.
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:grouped[tuple(r[k] for k in group_keys)][r['motion_id']].append(r)
    result=[]
    for key,sources in sorted(grouped.items()):
        names=next(iter(sources.values()))[0]['values'].keys();values={}
        for name in names:
            source_values={mid:float(np.mean([r['values'][name] for r in rs if r['values'][name] is not None]))
                           for mid,rs in sources.items() if any(r['values'][name] is not None for r in rs)}
            x=list(source_values.values())
            values[name]=dict(source_means=source_values,n=len(x),mean=float(np.mean(x)) if x else None,
                              min=min(x) if x else None,max=max(x) if x else None)
        result.append(dict(zip(group_keys,key),replicates_per_source={mid:len(rs) for mid,rs in sources.items()},values=values))
    return result

def probes():
    assert not (OUT/'probes.json').exists(),'Preserve previous results'
    rows=read(FROZEN/'probe_slots.json');noise=read(FROZEN/'probe_noise.json')
    accepted=read(ROOT/'docs/itm_rework/method/canonical_training_input_20260925_v1.json')
    mm=np.load(accepted['training_args']['mean']);ms=np.load(accepted['training_args']['std'])
    report=[];byid={};q_errors=[];noise_checked={};files=[]
    for slot in rows:
        pid=slot['probe_id'];path=RUN/'D'/f'{pid}.npz';meta=read(path.with_suffix('.json'));n=slot['frames']
        for k,v in slot.items():
            if k!='status':assert meta[k]==v,(pid,k)
        assert meta['status']=='completed' and meta['noise']==noise[slot['noise_key']]
        assert meta['scaled_t']==slot['timestep'] and meta['diffusion']['num_timesteps']==1000
        assert meta['diffusion']['timestep_map']==list(range(1000))
        with np.load(path) as z:
            arrays={k:z[k] for k in z.files};np.testing.assert_array_equal(arrays['frame_mask'],np.ones(n,bool))
        expected={'predicted_x0','target','x_t','frame_mask','encoder_fused','encoder_timed','encoder_layer_0','encoder_layer_1','encoder_final'}|{f'{k}_{j}' for k in ('hidden','residual') for j in range(8)}
        assert set(arrays)==expected
        for k,v in arrays.items():
            if k=='frame_mask':continue
            assert v.shape==((263,1,n) if k in ('predicted_x0','target','x_t') else (n,512)) and v.dtype==np.float32
            s=stats(v);assert s==meta['reductions'][k],(pid,k)
        target=((np.load(slot['target_feature_path'])-mm)/ms).astype(np.float32).T[:,None,:]
        np.testing.assert_array_equal(arrays['target'],target);assert ah(target)==meta['target_sha256'] and ah(arrays['x_t'])==meta['x_t_sha256']
        for draw in meta['noise'].values():
            if draw and draw['address_sha256'] not in noise_checked:noise_checked[draw['address_sha256']]=reconstruct_noise(draw)
        eps=noise_checked[meta['noise']['valid_noise']['address_sha256']][0]
        betas=np.array(meta['diffusion']['betas']);alpha=np.cumprod(1-betas)[slot['timestep']]
        # Same q_sample scalar casts; no diffusion/model invocation.
        qt=np.float32(np.sqrt(alpha))*target+np.float32(np.sqrt(1-alpha))*eps
        qerr=float(np.max(np.abs(qt-arrays['x_t'])));q_errors.append(qerr)
        np.testing.assert_allclose(qt,arrays['x_t'],atol=1e-6,rtol=1e-6)
        mse=float(np.square(arrays['predicted_x0'].astype(float)-target).mean());assert mse==meta['x0_masked_mse']
        values={'x0_mse':mse}
        for k,s in meta['reductions'].items():
            values[k+'.rms']=s['rms'];values[k+'.max_abs']=s['max_abs']
        for j in range(8):
            h=meta['reductions'][f'hidden_{j}']['rms'];r=meta['reductions'][f'residual_{j}']['rms'];ratio=r/h if h else None
            assert ratio==meta['residual_hidden_ratios'][str(j)]
            values[f'residual_hidden_ratio_{j}']=ratio
        item={k:slot[k] for k in ('probe_id','split','motion_id','sensor_config','frames','timestep','noise_seed','condition','context')}
        item.update(status='valid',values=values,reductions=meta['reductions'],
                    ratio_zero_denominator_layers=[j for j in range(8) if values[f'residual_hidden_ratio_{j}'] is None],q_sample_max_abs=qerr)
        report.append(item);byid[pid]=item;files.extend([dict(path=str(p),sha256=sha(p)) for p in (path,path.with_suffix('.json'))])
    index={(r['split'],r['motion_id'],r['sensor_config'],r['timestep'],r['noise_seed'],r['context'],r['condition']):r for r in rows}
    supplied={(r['probe_id'],r['reference_probe_id']):r for r in read(RUN/'D/paired_differences.json')};deltas=[]
    for key,slot in index.items():
        comparisons=[]
        if slot['condition']!='text_only':comparisons.append((key[:-1]+('text_only',),slot['condition']+'-text_only'))
        if slot['condition']=='paired':
            for condition in ('physical_static','tail_reversed'):
                other=key[:-1]+(condition,)
                if other in index:comparisons.append((other,'paired-'+condition))
        if slot['context']!='exact_length':comparisons.append((key[:-2]+('exact_length',key[-1]),'padded-exact'))
        with np.load(RUN/'D'/f"{slot['probe_id']}.npz") as z:a={k:z[k] for k in z.files if k!='frame_mask'}
        for other,contrast in comparisons:
            bslot=index[other]
            with np.load(RUN/'D'/f"{bslot['probe_id']}.npz") as z:b={k:z[k] for k in z.files if k!='frame_mask'}
            np.testing.assert_array_equal(a['target'],b['target']);np.testing.assert_array_equal(a['x_t'],b['x_t'])
            reductions={k:stats(v.astype(float)-b[k]) for k,v in a.items()}
            stored=supplied[slot['probe_id'],bslot['probe_id']];assert reductions==stored['differences']
            dm=byid[slot['probe_id']]['values']['x0_mse']-byid[bslot['probe_id']]['values']['x0_mse'];assert dm==stored['x0_mse_difference']
            values={'x0_mse_difference':dm}
            for k,s in reductions.items():values[k+'.rms']=s['rms'];values[k+'.max_abs']=s['max_abs']
            item={k:slot[k] for k in ('probe_id','split','motion_id','sensor_config','frames','timestep','noise_seed','condition','context')}
            item.update(reference_probe_id=bslot['probe_id'],contrast=contrast,values=values,reductions=reductions,
                padding_treatment=('no_added_padding' if slot['frames']==158 else 'actual_padding') if contrast=='padded-exact' else 'not_padding_contrast')
            deltas.append(item)
    assert len(deltas)==len(supplied)==624
    padding=[r for r in deltas if r['contrast']=='padded-exact'];assert len(padding)==96
    # Preserve source-level padding comparisons, preventing the 158-frame control from entering treatment means.
    write(OUT/'probes.json',dict(planned=480,valid=len(report),rows=report,noise_addresses_replayed=len(noise_checked),
        max_q_sample_replay_error=max(q_errors),aggregates=probe_aggregate(report,('split','sensor_config','timestep','context','condition'))))
    write(OUT/'probe_contrasts.json',dict(rows=deltas,planned=624,recomputed=len(deltas),padding_count=len(padding),
        aggregates=probe_aggregate([r for r in deltas if r['contrast']!='padded-exact'],('split','sensor_config','timestep','context','contrast')),
        padding_by_source=probe_aggregate(padding,('motion_id','sensor_config','timestep','condition','padding_treatment'))))
    write(OUT/'probe_artifact_audit.json',dict(files=files,code_sha256=sha(__file__)))
    print(json.dumps(dict(probes=len(report),contrasts=len(deltas),padding=len(padding),q_sample_max_abs=max(q_errors))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['render','samples','probes'])
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--images',type=Path,default=IMAGES)
    args=parser.parse_args();OUT=args.output;IMAGES=args.images
    {'render':render,'samples':samples,'probes':probes}[args.mode]()
