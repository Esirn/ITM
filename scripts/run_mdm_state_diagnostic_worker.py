#!/usr/bin/env python
"""Authorized fixed-R V worker; reads saved x_t, never diffuses or decodes."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from itm.models.mdm_correction import read,verify,artifact,array_sha
from itm.models.mdm_sampling_alignment import describe,difference,reductions,save_npz,save_json,numpy
from itm.models.mdm_state_diagnostic import StatePackage,StateGuard,StateCache,PACKAGE_SHA,CHECKPOINT_SHA,BRANCHES
from run_mdm_sampling_alignment_worker import load_fixed
from run_mdm_correction_worker import batch_inputs


def diagnose(package,args,guard):
    _,(model,diffusion,controlled,encoder,original,handles,backbone)=load_fixed(package,args.device)
    cache=StateCache();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:
        # Preserve upstream CLIP's native half weights and its float32 embedding
        # return, exactly as in S1. Denoiser/control inference remains float32.
        if any(p.dtype!=torch.float32 for name,p in model.named_parameters()
               if p.is_floating_point() and not name.startswith('clip_model.')):
            raise ValueError('V requires float32 denoiser/control weights')
        with torch.no_grad():
            for group in package.groups:
                gid=group['group_id'];started=time.monotonic()
                grows=[r for r in guard.state['groups'] if r['group_id']==gid]
                brows=[r for r in guard.state['branches'] if r['group_id']==gid]
                crows=[r for r in guard.state['checks'] if r['group_id']==gid]
                slots=[r for r in package.slots if r['group_id']==gid]
                if [r['branch'] for r in slots]!=list(BRANCHES) or [r['slot_id'] for r in slots]!=group['slot_ids']:
                    raise ValueError('Frozen V branch order mismatch')
                for row in grows+brows:row['status']='running'
                guard.state['active_group']=gid;guard.persist();guard.check()
                case=next(r for r in package.cases if r['cache_id']==group['cache_id'])
                items,rows,descriptors=package.inputs(case);saved=package.saved(group)
                if any(len(v['motion'])!=group['frames'] or not np.array_equal(v['motion'].T[:,None],saved['target']) for v in items):
                    raise ValueError('V length/target identity changed')
                before={k:v['actual'] for k,v in guard.state['counts'].items()}
                entry,built=cache.get(case,items,rows,descriptors,package,model,encoder,guard,args.device,batch_inputs)
                if built!=(group['caption_encoder_calls']==1):raise ValueError('V cache population/order changed')
                if guard.state['counts']['encoder']['actual']-before['encoder']!=group['encoder_examples_max']:
                    raise ValueError('V cache encoder count changed')
                if built:
                    from itm.models.mdm_sampling_alignment import input_arrays
                    for row,item in zip(slots,items):
                        save_npz(out/'inputs'/(case['cache_id']+'-'+row['branch']+'.npz'),input_arrays(item,row),guard)
                    save_json(out/'inputs'/(case['cache_id']+'.json'),dict(case=case,identity=entry['identity'],
                        arrays=descriptors,text_embedding=entry['text_descriptor'],control=entry['control_descriptor']),guard)
                _,y=batch_inputs(items,rows,args.device)
                if not np.array_equal(numpy(y['imu_frame_mask'][0:1]),saved['frame_mask']):raise ValueError('V mask changed')
                y.update(text_embed=entry['text'].repeat(1,4,1),imu_control=entry['control'],imu_control_mode='static_delta')
                y.pop('imu_reference')
                if y['imu_condition_present'].tolist()!=[False,True,True,True] or y['imu_control'][0].count_nonzero():
                    raise ValueError('V strict null changed')
                xt=torch.from_numpy(saved['x_t']).to(args.device).repeat(4,1,1,1)
                t=torch.full((4,),group['timestep'],device=args.device,dtype=torch.long)
                scaled=diffusion._scale_timesteps(t)
                if list(xt.shape)!=group['input_batch_shape'] or numpy(scaled).tolist()!=group['scaled_t']*4:
                    raise ValueError('Saved V state/timestep mismatch')
                prediction=guard.invoke('mdm',4,lambda:model(xt,scaled,y=y))
                if prediction.dtype!=torch.float32 or prediction.shape!=xt.shape:raise ValueError('V prediction dtype/shape')
                pred=numpy(prediction)
                # Raw arrays and metadata precede EVERY anchor decision.
                shared=out/'states'/(gid+'.npz')
                save_npz(shared,dict(x_t=saved['x_t'],target=saved['target'],frame_mask=saved['frame_mask']),guard)
                for index,(slot,row) in enumerate(zip(slots,brows)):
                    path=out/'branches'/(slot['slot_id']+'.npz')
                    save_npz(path,dict(predicted_x0=pred[index]),guard)
                    row.update(raw_artifact=artifact(path),diagnostic_only=True);grows[0]['raw_artifact']=artifact(shared);guard.persist()
                    meta=dict(slot,status='raw_saved_pending_anchors',diagnostic_only=True,
                        final_group_metadata=str(out/'groups'/(gid+'.json')),package_sha256=PACKAGE_SHA,
                        implementation_sha256=guard.context['implementation_sha256'],checkpoint_sha256=CHECKPOINT_SHA,
                        group=group,state_artifact=artifact(shared),input_artifact=artifact(out/'inputs'/(case['cache_id']+'-'+slot['branch']+'.npz')),
                        input_descriptors=descriptors[index],cache_identity=entry['identity'],caption=case['caption'],
                        arrays=dict(predicted_x0=describe(pred[index])),scaled_t=numpy(scaled[index:index+1]).tolist(),
                        reductions=reductions(pred[index]),counts=guard.state['counts'])
                    save_json(path.with_suffix('.json'),meta,guard)
                reports=[]
                for spec,row in zip([r for r in package.checks if r['group_id']==gid],crows):
                    index=BRANCHES.index(spec['branch'])
                    reference=saved[spec['reference']] if spec['reference_kind']=='saved_paired_trace' else pred[0:1]
                    report=difference(pred[index:index+1],reference,atol=spec['atol'],rtol=spec['rtol'],enforce=False)
                    report['finite_count']=report['count']
                    row.update(status='completed' if report['passed'] else 'failed',**report)
                    record=dict(spec,status=row['status'],**report);reports.append(record)
                    save_json(out/'checks'/(spec['check_id']+'.json'),record,guard)
                    guard.persist()
                passed=all(r['passed'] for r in reports)
                for row in grows+brows:row.update(status='completed' if passed else 'failed',diagnostic_only=not passed)
                save_json(out/'groups'/(gid+'.json'),dict(group=group,status=grows[0]['status'],diagnostic_only=not passed,
                    checks=reports,branches=brows,counts=guard.state['counts'],elapsed_seconds=time.monotonic()-started,
                    deadline_monotonic=guard.deadline,peak_allocated_bytes=guard.state['peak_allocated_bytes']),guard)
                guard.persist()
                if not passed:raise ValueError('Frozen V software anchor failed')
    finally:
        handles.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--run-context',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.package=args.package.resolve();args.output=args.output.resolve();args.device='cuda:0'
    guard=StateGuard(read(args.run_context.resolve()))
    try:
        for entry in guard.context['expected_artifacts']:verify(entry);guard.check()
        package=StatePackage(args.package)
        guard.configure_cuda(torch,torch.device(args.device))
        diagnose(package,args,guard);guard.finish()
    except BaseException as error:
        guard.fail(error);raise

if __name__=='__main__':main()
