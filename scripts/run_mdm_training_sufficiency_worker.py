#!/usr/bin/env python
"""One frozen saved-state queue; no training, diffusion steps, new noise or decode."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from itm.models.mdm_correction import read,verify,artifact,environment
from itm.models.mdm_sampling_alignment import describe,difference,numpy,save_npz,save_json,input_arrays
from itm.models.mdm_training_sufficiency import TrainingPackage,TrainingGuard,SnapshotCache,PACKAGE_SHA,BRANCHES
from run_mdm_correction_worker import load_model,batch_inputs


def forbidden(*args,**kwargs):
    raise RuntimeError('Frozen diagnostic forbids diffusion/noise/CFG/decode/training')


def diagnose(package,args,guard):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    first=package.snapshots[0];initial=package.state(first)
    model,diffusion,controlled,encoder,original,handles,backbone=load_model(package,initial,args.device)
    try:
        for name in ('q_sample','p_sample','p_mean_variance','p_sample_loop','ddim_sample_loop','training_losses'):
            if hasattr(diffusion,name):setattr(diffusion,name,forbidden)
        backbone.decode_motion=forbidden
        if diffusion.num_timesteps!=1000 or any(p.dtype!=torch.float32 for n,p in model.named_parameters()
                if p.is_floating_point() and not n.startswith('clip_model.')):
            raise ValueError('Original inference steps/precision changed')
        for p in model.parameters():p.requires_grad_(False)
        states={r['state_id']:r for r in package.states};cases={r['cache_id']:r for r in package.cases}
        jrows={r['job_id']:r for r in guard.state['jobs']}
        brows={j['job_id']:[r for r in guard.state['branches'] if r['job_id']==j['job_id']] for j in package.jobs}
        crows={r['check_id']:r for r in guard.state['checks']}
        specs={j['job_id']:[r for r in package.checks if r['job_id']==j['job_id']] for j in package.jobs}
        srows={r['state_id']:r for r in guard.state['states']}
        input_cache={};saved_cache={};reference_cache={};cache=SnapshotCache()
        with torch.no_grad():
            for snapshot in package.snapshots:
                sid=snapshot['snapshot_id'];guard.check()
                if sid!='I':
                    state=package.state(snapshot)
                    if state['control_config']!=initial['control_config']:raise ValueError('Cross-snapshot runtime configuration changed')
                    encoder.load_state_dict(state['imu_encoder']);controlled.adapters.load_state_dict(state['adapters'])
                    del state
                    model.eval();original.eval()
                cache.begin(snapshot['artifact']['sha256'])
                for job in (r for r in package.jobs if r['snapshot_id']==sid):
                    jid=job['job_id'];state_row=states[job['state_id']];case=cases[job['cache_id']]
                    jr=jrows[jid];br=brows[jid];started=time.monotonic()
                    guard.state['active_job']=jid;jr['status']='running'
                    for r in br:r['status']='running'
                    guard.persist();guard.check()
                    if case['cache_id'] not in input_cache:
                        items,rows,descriptors=package.inputs(case)
                        input_cache[case['cache_id']]=(items,rows,descriptors)
                        for branch,item,row in zip(BRANCHES,items,rows):
                            save_npz(out/'inputs'/(case['cache_id']+'-'+branch+'.npz'),input_arrays(item,row),guard)
                        save_json(out/'inputs'/(case['cache_id']+'.json'),dict(case=case,descriptors=descriptors),guard)
                    items,rows,descriptors=input_cache[case['cache_id']]
                    if state_row['state_id'] not in saved_cache:
                        saved=package.saved(state_row);saved_cache[state_row['state_id']]=saved
                        sr=srows[state_row['state_id']];sr['status']='running'
                        path=out/'states'/(state_row['state_id']+'.npz')
                        save_npz(path,saved,guard);sr.update(raw_saved=True,artifact=artifact(path));guard.persist()
                    saved=saved_cache[state_row['state_id']]
                    if any(len(i['motion'])!=state_row['frames'] or not np.array_equal(i['motion'].T[:,None],saved['target']) for i in items):
                        raise ValueError('Same-state target/length changed')
                    before={k:v['actual_calls'] for k,v in guard.state['counts'].items()}
                    entry,text_built,control_built=cache.get(case,items,rows,descriptors,package,model,encoder,guard,args.device,batch_inputs)
                    if (int(text_built)!=job['caption_calls'] or 6*int(control_built)!=job['encoder_calls']
                            or guard.state['counts']['encoder']['actual_calls']-before['encoder']!=job['encoder_calls']):
                        raise ValueError('Frozen caption/control cache population changed')
                    if text_built:
                        save_npz(out/'cache'/('text-'+case['cache_id']+'.npz'),dict(text=numpy(entry['text'])),guard)
                    if control_built:
                        cp=out/'cache'/(sid+'-'+case['cache_id'])
                        save_npz(cp.with_suffix('.npz'),dict(control=numpy(entry['control'])),guard)
                        save_json(cp.with_suffix('.json'),dict(snapshot_id=sid,checkpoint=snapshot['artifact'],cache_id=case['cache_id'],
                                  text_identity=entry['text_identity'],control_identity=entry['control_identity'],
                                  text_descriptor=entry['text_descriptor'],control_descriptor=entry['control_descriptor']),guard)
                    _,y=batch_inputs(items,rows,args.device)
                    if not np.array_equal(numpy(y['imu_frame_mask'][0:1]),saved['frame_mask']):raise ValueError('Same-state mask changed')
                    y.update(text_embed=entry['text'].repeat(1,4,1),imu_control=entry['control'],imu_control_mode='static_delta')
                    y.pop('imu_reference')
                    if y['imu_condition_present'].tolist()!=[False,True,True,True] or y['imu_control'][0].count_nonzero():
                        raise ValueError('Strict-null branch changed')
                    xt=torch.from_numpy(saved['x_t']).to(args.device).repeat(4,1,1,1)
                    t=torch.full((4,),state_row['timestep'],device=args.device,dtype=torch.long)
                    scaled=diffusion._scale_timesteps(t)
                    if list(xt.shape)!=job['input_batch_shape'] or numpy(scaled).tolist()!=job['scaled_t']:
                        raise ValueError('Same-state batch/timestep identity changed')
                    pred=numpy(guard.invoke('mdm',4,lambda:model(xt,scaled,y=y)))
                    if pred.dtype!=np.float32 or pred.shape!=tuple(job['input_batch_shape']):raise ValueError('Prediction dtype/shape changed')
                    raw_path=out/'predictions'/(jid+'.npz')
                    # Finite raw arrays are durable BEFORE any software anchor.
                    save_npz(raw_path,{b:pred[i] for i,b in enumerate(BRANCHES)},guard)
                    jr.update(raw_saved=True,raw_artifact=artifact(raw_path),diagnostic_only=True)
                    for r,b in zip(br,BRANCHES):r.update(raw_saved=True,array=b,diagnostic_only=True)
                    guard.persist()
                    reports=[]
                    for spec in specs[jid]:
                        ref=spec['reference'];branch=BRANCHES.index(spec['branch'])
                        if ref['kind']=='same_job':reference=pred[BRANCHES.index(ref['branch']):BRANCHES.index(ref['branch'])+1]
                        elif ref['kind']=='new_prediction':
                            with np.load(out/'predictions'/(ref['job_id']+'.npz'),allow_pickle=False) as z:reference=z[ref['branch']][None]
                        else:
                            key=(ref['artifact']['path'],ref['array'])
                            if key not in reference_cache:
                                with np.load(verify(ref['artifact']),allow_pickle=False) as z:reference_cache[key]=z[ref['array']].copy()
                            reference=reference_cache[key]
                            if describe(reference)!=ref['descriptor']:raise ValueError('Historical anchor identity changed')
                            if reference.ndim==3:reference=reference[None]
                        report=difference(pred[branch:branch+1],reference,atol=spec['atol'],rtol=spec['rtol'],enforce=False)
                        report.update(finite_count=report['count'],check_id=spec['check_id'])
                        crows[spec['check_id']]['status']='completed' if report['passed'] else 'failed';reports.append(report)
                        guard.persist()
                        if not report['passed']:break
                    passed=len(reports)==len(specs[jid]) and all(r['passed'] for r in reports)
                    if not passed:
                        for spec in specs[jid]:
                            if crows[spec['check_id']]['status']=='planned':crows[spec['check_id']]['status']='not_run_after_stop'
                    save_json(out/'jobs'/(jid+'.json'),dict(job=job,status='completed' if passed else 'failed',diagnostic_only=not passed,
                              state=state_row,checkpoint=snapshot['artifact'],raw_artifact=artifact(raw_path),
                              state_artifact=srows[state_row['state_id']]['artifact'],input_descriptors=descriptors,
                              text_identity=entry['text_identity'],control_identity=entry['control_identity'],
                              arrays={b:describe(pred[i]) for i,b in enumerate(BRANCHES)},checks=reports,
                              elapsed_seconds=time.monotonic()-started),guard)
                    # Completion is durable only after the result metadata was
                    # published; a metadata-write fault must leave a failed job.
                    for r in [jr,*br]:r.update(status='completed' if passed else 'failed',diagnostic_only=not passed)
                    if sid=='U200' and passed:srows[state_row['state_id']]['status']='completed'
                    guard.persist()
                    if not passed:raise ValueError('Frozen sufficiency software anchor failed: '+reports[-1]['check_id'])
                    if job['state_id'] in ('S048','S096','S144','S192'):
                        print(sid,job['state_id'],guard.state['counts']['mdm']['completed_calls'],flush=True)
    finally:handles.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','run-context','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.device='cuda:0';args.output=args.output.resolve()
    context=read(args.run_context.resolve());package=TrainingPackage(args.package);guard=TrainingGuard(context,package)
    try:
        for entry in context['expected_artifacts']:verify(entry);guard.check()
        if environment()!=package.common['environment']:raise ValueError('Bound environment changed')
        guard.configure_cuda(torch,torch.device(args.device));diagnose(package,args,guard);guard.finish()
    except BaseException as error:guard.fail(error);raise


if __name__=='__main__':main()
