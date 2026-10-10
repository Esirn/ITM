#!/usr/bin/env python
"""One frozen same-history suffix queue; save raw before every software check."""
import argparse
from pathlib import Path

import numpy as np
import torch

from itm.data.random_plan import sampling_noise
from itm.models.mdm_correction import read,verify,artifact,encode_control,environment
from itm.models.mdm_imu_control import make_text_imu_guidance_model
from itm.models.mdm_suffix_switch import SuffixPackage,SuffixGuard,PACKAGE_SHA,BRANCHES,step_id,replay_suffix
from itm.models.mdm_sampling_alignment import numpy,describe,difference,combination_check,input_arrays,save_npz,save_json
from itm.models.mdm_state_diagnostic import CountedEncoder
from run_mdm_correction_worker import load_model,batch_inputs
from run_mdm_prefix_mechanism_worker import CountedRaw,Observe,forbidden


def run(package,args,guard):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    weight=package.state('F200')
    model,diffusion,controlled,encoder,original,handles,backbone=load_model(package,weight,args.device)
    try:
        if (diffusion.num_timesteps!=1000 or list(diffusion.timestep_map)!=list(range(1000))
            or getattr(diffusion,'rescale_timesteps',False)):raise ValueError('Original schedule required')
        for n,p in model.named_parameters():
            p.requires_grad_(False)
            if p.is_floating_point() and not n.startswith('clip_model.') and p.dtype!=torch.float32:raise ValueError('Native precision required')
        for n in ('q_sample','p_sample','p_sample_loop','ddim_sample_loop','training_losses'):
            if hasattr(diffusion,n):setattr(diffusion,n,forbidden)
        counted=CountedRaw(model,guard)
        guided=make_text_imu_guidance_model(counted,mode='factorized',branch_execution='batched',bypass_encoder=original).eval()
        if any(m.training for m in model.modules()):raise ValueError('Frozen eval required')
        cases={};texts={};controls={};starts={};current={}
        def condition(trajectory):
            cid=trajectory['panel_case_id']
            if cid not in cases:
                item,row=package.inputs(trajectory);cases[cid]=(item,row)
                ia=input_arrays(item,row);ia['target']=item['motion'].T[None,:,None,:]
                save_npz(out/'inputs'/(cid+'.npz'),ia,guard)
                save_json(out/'inputs'/(cid+'.json'),dict(slot=row,arrays={k:describe(v) for k,v in ia.items()}),guard)
            item,row=cases[cid];_,y=batch_inputs([item],[row],args.device)
            if cid not in texts:
                texts[cid]=guard.invoke('caption',1,lambda:model.encode_text([row['text']]))
                if texts[cid].dtype!=torch.float32 or texts[cid].shape[1]!=1:raise ValueError('Native caption batch1')
                save_npz(out/'cache'/('text-'+cid+'.npz'),dict(text=numpy(texts[cid])),guard)
            if cid not in controls:
                value,present=encode_control(CountedEncoder(encoder,guard),y,mode='static_delta')
                if value.dtype!=torch.float32 or not present.all():raise ValueError('Paired encoder presence/precision')
                controls[cid]=value
                cp=out/'cache'/('F200-'+cid)
                save_npz(cp.with_suffix('.npz'),dict(control=numpy(value)),guard)
                save_json(cp.with_suffix('.json'),dict(checkpoint=package.snapshots['F200']['artifact'],case_id=cid,
                    input_arrays={k:describe(v) for k,v in input_arrays(item,row).items()},control=describe(value),text=describe(texts[cid])),guard)
            y.update(text_embed=texts[cid],imu_control=controls[cid],imu_control_mode='static_delta',text_scale=1.,imu_scale=1.,joint_scale=1.,
                backbone_bypass=torch.tensor([False],device=args.device));y.pop('imu_reference')
            if trajectory['condition']=='switch_imu_off':
                y.update(imu_control=torch.zeros_like(y['imu_control']),imu_condition_present=torch.zeros_like(y['imu_condition_present']))
            if not y['text_condition_present'].all() or bool(y['imu_condition_present'].all())!=trajectory['imu_present']:
                raise ValueError('Frozen strict condition flags')
            return y
        def observe(t,raw):
            trajectory=current['trajectory'];tid=trajectory['trajectory_id'];key=step_id(tid,t)
            path=out/'steps'/(key+'.npz');save_npz(path,raw,guard)
            guard.update('steps',key,raw_saved=True,raw_artifact=artifact(path),diagnostic_only=True);guard.persist()
            if t in trajectory['trace_timesteps']:
                tp=out/'traces'/(key+'.npz');save_npz(tp,raw,guard)
                guard.update('traces',key,status='running',raw_saved=True,raw_artifact=artifact(tp),diagnostic_only=True);guard.persist()
            checks=combination_check([raw[b] for b in BRANCHES],raw['actual_combined_prediction'],enforce=False)
            guard.equation('CFG-'+key,dict(passed=all(v['passed'] for v in checks.values()),reports=checks))
            if trajectory['condition']=='switch_imu_off':
                for a,b in [('f00','f01'),('f10','f11')]:guard.equation('ABS-'+key+'-'+a+'='+b,difference(raw[a],raw[b],enforce=False))
            for anchor in (a for a in package.anchors if a['trajectory_id']==tid and a.get('timestep')==t):
                with np.load(verify(anchor['raw']),allow_pickle=False) as old:
                    for b in anchor['arrays']:guard.equation('ANCH-'+key+'-'+b,difference(raw[b],old[b],enforce=False))
            current['raw']=raw
        observer=Observe(guided,counted,observe).eval()
        with torch.no_grad():
            for trajectory in package.trajectories:
                tid=trajectory['trajectory_id'];cid=trajectory['panel_case_id'];current.clear();current['trajectory']=trajectory
                guard.update('trajectories',tid,status='running');guard.state['active']=dict(table='trajectories',id=tid);guard.persist()
                y=condition(trajectory)
                with np.load(verify(trajectory['start']),allow_pickle=False) as z:
                    x=z['x_899'].copy();mask=z['frame_mask'].copy()
                if x.dtype!=np.float32 or x.shape!=(1,263,1,trajectory['frames']) or not np.isfinite(x).all():raise ValueError('Native saved x899')
                if not np.array_equal(mask,numpy(y['imu_frame_mask'])):raise ValueError('Start frame mask')
                save_npz(out/'starts'/(tid+'.npz'),dict(x_899=x,frame_mask=mask),guard)
                prior=starts.setdefault(cid,x.copy())
                guard.equation('START-'+cid,dict(passed=np.array_equal(prior,x),bitwise=True))
                specs={r['step']:r for r in package.noise[trajectory['noise_id']]['draws']};drawn=[]
                def draw(step):
                    spec=specs[step]
                    value,meta=guard.invoke('noise',1,lambda:sampling_noise(tuple(spec['shape']),backbone_id=spec['backbone_id'],
                        case_id=spec['case_id'],sample_seed=spec['sample_seed'],step=step))
                    if meta!=spec or value.dtype!=np.float32:raise ValueError('Frozen suffix addressed noise identity')
                    drawn.append(meta);return torch.from_numpy(value).to(args.device)
                def before(t,x):
                    key=step_id(tid,t);guard.state['active']=dict(table='steps',id=key)
                    guard.update('steps',key,status='running');guard.persist();guard.check()
                def after(t,x,posterior):
                    guard.ensure_finite(posterior,'suffix posterior');guard.check();key=step_id(tid,t)
                    save_json(out/'steps'/(key+'.json'),dict(status='completed',package_sha256=PACKAGE_SHA,
                        implementation_sha256=guard.context['implementation_sha256'],trajectory_id=tid,timestep=t,
                        checkpoint=package.snapshots['F200']['artifact'],raw_artifact=guard.maps['steps'][key]['raw_artifact'],
                        arrays={k:describe(v) for k,v in current['raw'].items()},next_x=describe(x),noise=drawn[-1] if t else None),guard)
                    guard.update('steps',key,status='completed',diagnostic_only=False)
                    if t in trajectory['trace_timesteps']:guard.update('traces',key,status='completed',diagnostic_only=False)
                    guard.persist()
                features=replay_suffix(observer,diffusion,y,torch.from_numpy(x).to(args.device),draw,before=before,after=after)
                guard.state['active']=dict(table='finals',id=tid);guard.update('finals',tid,status='running');guard.persist()
                save_npz(out/'raw_finals'/(tid+'.npz'),dict(normalized_humanml263=numpy(features)),guard)
                joints=guard.invoke('decode',1,lambda:backbone.decode_motion(features))
                if joints.dtype!=torch.float32 or joints.shape!=(1,trajectory['frames'],22,3):raise ValueError('Original final joint layout')
                raw=dict(normalized_humanml263=numpy(features),joints=numpy(joints),frame_mask=numpy(y['imu_frame_mask']))
                fp=out/'finals'/(tid+'.npz');save_npz(fp,raw,guard)
                guard.update('finals',tid,raw_saved=True,raw_artifact=artifact(fp),diagnostic_only=True);guard.persist()
                for anchor in (a for a in package.anchors if a['trajectory_id']==tid and a.get('kind')=='final'):
                    with np.load(verify(anchor['raw']),allow_pickle=False) as old:
                        for b in anchor['arrays']:
                            reference=old['motion' if b=='joints' else b]
                            report=dict(passed=np.array_equal(raw[b],reference),bitwise=True) if b=='frame_mask' else difference(raw[b],reference,enforce=False)
                            guard.equation('FINAL-'+tid+'-'+b,report)
                save_json(fp.with_suffix('.json'),dict(status='completed',trajectory=trajectory,package_sha256=PACKAGE_SHA,
                    implementation_sha256=guard.context['implementation_sha256'],checkpoint=package.snapshots['F200']['artifact'],
                    raw_artifact=artifact(fp),arrays={k:describe(v) for k,v in raw.items()},joint_field_mapping='joints equals original motion layout'),guard)
                save_json(out/'trajectories'/(tid+'.json'),dict(trajectory=trajectory,status='completed',noise_draws=drawn,
                    checkpoint=package.snapshots['F200']['artifact'],final=artifact(fp)),guard)
                guard.update('finals',tid,status='completed',diagnostic_only=False);guard.update('trajectories',tid,status='completed');guard.persist()
                print('suffix',tid,trajectory['sensor_config'],trajectory['condition'],guard.state['counts']['mdm']['completed_calls'],flush=True)
        guard.state['active']=None;guard.persist()
    finally:handles.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('package','run-context','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.device='cuda:0';package=SuffixPackage(a.package);context=read(a.run_context);guard=SuffixGuard(context,package)
    try:
        for e in context['expected_artifacts']:verify(e);guard.check()
        if environment()!=package.common['environment']:raise ValueError('Frozen environment')
        guard.configure_cuda(torch,torch.device(a.device));run(package,a,guard);guard.finish()
    except BaseException as e:guard.fail(e);raise


if __name__=='__main__':main()
