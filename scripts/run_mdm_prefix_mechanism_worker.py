#!/usr/bin/env python
"""One immutable prefix queue; raw arrays precede all numerical checks."""
import argparse
from pathlib import Path

import numpy as np
import torch

from itm.data.random_plan import sampling_noise
from itm.models.mdm_correction import read,verify,artifact,encode_control,environment
from itm.models.mdm_imu_control import make_text_imu_guidance_model
from itm.models.mdm_prefix_mechanism import PrefixPackage,PrefixGuard,PACKAGE_SHA,BRANCHES,step_id,replay_prefix
from itm.models.mdm_sampling_alignment import numpy,describe,difference,combination_check,input_arrays,save_npz,save_json
from itm.models.mdm_state_diagnostic import CountedEncoder
from run_mdm_correction_worker import load_model,batch_inputs


def forbidden(*a,**k):raise RuntimeError('Prefix forbids training/full sampling/decoding')


class CountedRaw(torch.nn.Module):
    def __init__(self,raw,guard):super().__init__();self.raw=raw;self.guard=guard;self.last=None
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('raw'),name)
    def forward(self,x,t,y=None):
        if len(x)!=4:raise ValueError('Native factorized MDM batch4 required')
        result=self.guard.invoke('mdm',4,lambda:self.raw(x,t,y=y))
        if result.dtype!=torch.float32 or result.shape!=x.shape:raise ValueError('Native float32 shape')
        self.last=tuple(result.chunk(4));return result


class Observe(torch.nn.Module):
    def __init__(self,guided,counted,callback):
        super().__init__();self.guided=guided;self.counted=counted;self.callback=callback
        self.njoints=guided.njoints;self.nfeats=guided.nfeats
    def forward(self,x,t,y=None):
        self.counted.last=None;result=self.guided(x,t,y=y)
        if self.counted.last is None:raise ValueError('Missing native batch4 result')
        arrays={b:numpy(v) for b,v in zip(BRANCHES,self.counted.last)}
        arrays.update(x_t=numpy(x),actual_combined_prediction=numpy(result),frame_mask=numpy(y['imu_frame_mask']))
        self.callback(int(t[0]),arrays);return result


def run(package,args,guard):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    weights={sid:package.state(sid) for sid in ('F200','U200')}
    model,diffusion,controlled,encoder,original,handles,backbone=load_model(package,weights['F200'],args.device)
    try:
        if (diffusion.num_timesteps!=1000 or list(diffusion.timestep_map)!=list(range(1000))
                or getattr(diffusion,'rescale_timesteps',False)):raise ValueError('Original schedule required')
        for n,p in model.named_parameters():
            p.requires_grad_(False)
            if p.is_floating_point() and not n.startswith('clip_model.') and p.dtype!=torch.float32:raise ValueError('Native precision required')
        for n in ('q_sample','p_sample','p_sample_loop','ddim_sample_loop','training_losses'):
            if hasattr(diffusion,n):setattr(diffusion,n,forbidden)
        backbone.decode_motion=forbidden
        counted=CountedRaw(model,guard)
        guided=make_text_imu_guidance_model(counted,mode='factorized',branch_execution='batched',bypass_encoder=original)
        guided.eval()
        state_specs={(s['trajectory_id'],s['timestep']):s for s in package.states}
        trajectory_specs={r['trajectory_id']:r for r in package.trajectories}
        probe_lookup={(p['state_id'],p['checkpoint']):p for p in package.probes}
        cases={};texts={};controls={};initials={};current={};active_weight=None
        def set_weight(sid):
            nonlocal active_weight
            if sid!=active_weight:
                encoder.load_state_dict(weights[sid]['imu_encoder']);controlled.adapters.load_state_dict(weights[sid]['adapters'])
                model.eval();original.eval();guided.eval();active_weight=sid
                if any(m.training for m in model.modules()):raise ValueError('Frozen eval required')
        def condition(cid,sid):
            if cid not in cases:
                trajectory=next(r for r in package.trajectories if r['panel_case_id']==cid and r['path_arm']=='F')
                item,row=package.inputs(trajectory);cases[cid]=(item,row)
                ia=input_arrays(item,row);ia['target']=item['motion'].T[None,:,None,:]
                save_npz(out/'inputs'/(cid+'.npz'),ia,guard)
                save_json(out/'inputs'/(cid+'.json'),dict(slot=row,arrays={k:describe(v) for k,v in ia.items()}),guard)
            item,row=cases[cid];_,y=batch_inputs([item],[row],args.device)
            if cid not in texts:
                texts[cid]=guard.invoke('caption',1,lambda:model.encode_text([row['text']]))
                if texts[cid].dtype!=torch.float32 or texts[cid].shape[1]!=1:raise ValueError('Caption native batch1')
                save_npz(out/'cache'/('text-'+cid+'.npz'),dict(text=numpy(texts[cid])),guard)
            key=(sid,cid)
            if key not in controls:
                value,present=encode_control(CountedEncoder(encoder,guard),y,mode='static_delta')
                if value.dtype!=torch.float32 or not present.all():raise ValueError('Original paired control presence/precision')
                controls[key]=value
                cp=out/'cache'/(sid+'-'+cid)
                save_npz(cp.with_suffix('.npz'),dict(control=numpy(value)),guard)
                save_json(cp.with_suffix('.json'),dict(checkpoint=package.snapshots[sid]['artifact'],case_id=cid,
                    input_arrays={k:describe(v) for k,v in input_arrays(item,row).items()},control=describe(value),text=describe(texts[cid])),guard)
            y.update(text_embed=texts[cid],imu_control=controls[key],imu_control_mode='static_delta',text_scale=1.,imu_scale=1.,joint_scale=1.,
                     backbone_bypass=torch.tensor([False],device=args.device));y.pop('imu_reference')
            return y
        def observe(t,arrays):
            active=guard.state['active'];key=active['id'];table=active['table']
            path=out/('steps' if table=='steps' else 'probes')/(key+'.npz')
            save_npz(path,arrays,guard)
            guard.update(table,key,raw_saved=True,raw_artifact=artifact(path),diagnostic_only=True)
            if table=='probes':
                for b in BRANCHES:guard.update('branches',key+'-'+b,executed=True,raw_saved=True,diagnostic_only=True)
                guard.persist();return
            trajectory=current['trajectory'];tid=trajectory['trajectory_id']
            ss=state_specs.get((tid,t))
            if ss:
                sp=out/'states'/(ss['state_id']+'.npz');save_npz(sp,arrays,guard)
                guard.update('states',ss['state_id'],status='running',raw_saved=True,artifact=artifact(sp))
            guard.persist()
            reports=combination_check([arrays[b] for b in BRANCHES],arrays['actual_combined_prediction'],enforce=False)
            guard.equation('CFG-'+key,dict(passed=all(v['passed'] for v in reports.values()),reports=reports))
            if trajectory['path_arm']=='T':
                for a,b in [('f00','f01'),('f10','f11')]:guard.equation('ABS-'+key+'-'+a+'='+b,difference(arrays[a],arrays[b],enforce=False))
            if t==900:
                for anchor in (a for a in package.anchors if a['trajectory_id']==tid):
                    with np.load(verify(anchor['npz']),allow_pickle=False) as z:
                        for b in anchor['arrays']:guard.equation('ANCH-'+tid+'-'+anchor['historical_arm']+'-'+b,difference(arrays[b],z[b],enforce=False))
            current['raw']=arrays
        observer=Observe(guided,counted,observe);observer.eval()
        with torch.no_grad():
            for sid in ('F200','U200'):
                set_weight(sid)
                paths=[r for r in package.trajectories if r['checkpoint']==sid]
                for trajectory in paths:
                    tid=trajectory['trajectory_id'];cid=trajectory['panel_case_id'];current.clear();current['trajectory']=trajectory
                    guard.update('trajectories',tid,status='running');guard.state['active']=dict(table='trajectories',id=tid);guard.persist()
                    y=condition(cid,sid)
                    if trajectory['path_arm']=='T':y.update(imu_control=torch.zeros_like(y['imu_control']),imu_condition_present=torch.zeros_like(y['imu_condition_present']))
                    frozen=package.noise[trajectory['noise_id']]['draws'];noise_by_step={r['step']:r for r in frozen};drawn=[]
                    def draw(step):
                        spec=noise_by_step[step]
                        value,meta=guard.invoke('noise',1,lambda:sampling_noise(tuple(spec['shape']),backbone_id=spec['backbone_id'],
                            case_id=spec['case_id'],sample_seed=spec['sample_seed'],step=step))
                        if meta!=spec or value.dtype!=np.float32:raise ValueError('Frozen addressed noise identity')
                        drawn.append(meta)
                        if step==-1:
                            save_npz(out/'initial'/(tid+'.npz'),dict(x_t=value),guard)
                            initials.setdefault(cid,{})[trajectory['path_arm']]=value.copy()
                            values=list(initials[cid].values());same=all(np.array_equal(value,v) for v in values)
                            if not same or len(values)==3:guard.equation('INIT-'+cid,dict(passed=same,bitwise=same,paths=len(values)))
                        return torch.from_numpy(value).to(args.device)
                    def before(t,x):
                        key=step_id(tid,t);guard.state['active']=dict(table='steps',id=key)
                        guard.update('steps',key,status='running');guard.persist();guard.check()
                    def after(t,x,posterior):
                        guard.ensure_finite(posterior,'prefix posterior');guard.check();key=step_id(tid,t)
                        save_json(out/'steps'/(key+'.json'),dict(status='completed',package_sha256=PACKAGE_SHA,
                            implementation_sha256=guard.context['implementation_sha256'],trajectory=trajectory,timestep=t,
                            checkpoint=package.snapshots[sid]['artifact'],raw_artifact=guard.maps['steps'][key]['raw_artifact'],
                            arrays={k:describe(v) for k,v in current['raw'].items()},next_x=describe(x),noise=drawn[-1]),guard)
                        guard.update('steps',key,status='completed',diagnostic_only=False)
                        ss=state_specs.get((tid,t))
                        if ss:guard.update('states',ss['state_id'],status='completed',diagnostic_only=False)
                        guard.persist()
                    terminal=replay_prefix(observer,diffusion,y,draw,before=before,after=after)
                    save_npz(out/'terminal'/(tid+'.npz'),dict(x_899=numpy(terminal),frame_mask=numpy(y['imu_frame_mask'])),guard)
                    save_json(out/'trajectories'/(tid+'.json'),dict(trajectory=trajectory,status='completed',noise_draws=drawn,
                        checkpoint=package.snapshots[sid]['artifact'],terminal=describe(terminal)),guard)
                    guard.update('trajectories',tid,status='completed');guard.persist()
                    print('prefix',tid,cid,guard.state['counts']['mdm']['completed_calls'],flush=True)
            states_by_id={s['state_id']:s for s in package.states}
            for sid in ('F200','U200'):
                set_weight(sid)
                for p in (p for p in package.probes if p['checkpoint']==sid):
                    pid=p['probe_id'];ss=states_by_id[p['state_id']];trajectory=trajectory_specs[ss['trajectory_id']];cid=trajectory['panel_case_id']
                    guard.state['active']=dict(table='probes',id=pid);guard.update('probes',pid,status='running')
                    for b in BRANCHES:guard.update('branches',pid+'-'+b,status='running')
                    guard.persist();y=condition(cid,sid)
                    with np.load(out/'states'/(ss['state_id']+'.npz'),allow_pickle=False) as z:saved={k:z[k] for k in z.files}
                    xt=torch.from_numpy(saved['x_t']).to(args.device);t=torch.full((1,),ss['timestep'],device=args.device,dtype=torch.long)
                    observer(xt,diffusion._scale_timesteps(t),y=y)
                    with np.load(out/'probes'/(pid+'.npz'),allow_pickle=False) as z:pred={k:z[k] for k in z.files}
                    arm=trajectory['path_arm']
                    if arm=='T' or sid==trajectory['checkpoint']:
                        for b in (('f00','f10') if arm=='T' else BRANCHES):guard.equation('OWN-'+pid+'-'+b,difference(pred[b],saved[b],enforce=False))
                    if sid=='U200':
                        fp=probe_lookup[(ss['state_id'],'F200')]
                        with np.load(out/'probes'/(fp['probe_id']+'.npz'),allow_pickle=False) as z:
                            for b in ('f00','f10'):guard.equation('CROSS-'+ss['state_id']+'-'+b,difference(pred[b],z[b],enforce=False))
                    save_json(out/'probes'/(pid+'.json'),dict(probe=p,state=ss,status='completed',checkpoint=package.snapshots[sid]['artifact'],
                        package_sha256=PACKAGE_SHA,implementation_sha256=guard.context['implementation_sha256'],
                        raw_artifact=guard.maps['probes'][pid]['raw_artifact'],state_artifact=guard.maps['states'][ss['state_id']]['artifact'],
                        arrays={k:describe(v) for k,v in pred.items()}),guard)
                    guard.update('probes',pid,status='completed',diagnostic_only=False)
                    for b in BRANCHES:guard.update('branches',pid+'-'+b,status='completed',diagnostic_only=False)
                    guard.persist()
        guard.state['active']=None;guard.persist()
    finally:handles.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('package','run-context','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.device='cuda:0';package=PrefixPackage(a.package);context=read(a.run_context)
    guard=PrefixGuard(context,package)
    try:
        for e in context['expected_artifacts']:verify(e);guard.check()
        if environment()!=package.common['environment']:raise ValueError('Frozen environment')
        guard.configure_cuda(torch,torch.device(a.device));run(package,a,guard);guard.finish()
    except BaseException as e:guard.fail(e);raise


if __name__=='__main__':main()
