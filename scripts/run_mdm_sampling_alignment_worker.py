#!/usr/bin/env python
"""One authorized Q or S1 phase. CPU fixtures call helpers with tiny fake models."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from itm.models.mdm_correction import (array_sha, artifact, read, verify, probe_noise, encode_control,
                                       ConditionCache, environment)
from itm.models.mdm_imu_control import make_text_imu_guidance_model
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm.models.mdm_sampling_alignment import (AlignmentPackage, AlignmentGuard, PACKAGE_SHA, CHECKPOINT_SHA,
    GUIDANCE, BRANCHES, TraceGuidance, input_arrays, describe, digest, difference, reductions,
    reduced_analysis, save_npz, save_json, numpy)
from run_mdm_correction_worker import load_model, batch_inputs, embed_text


def load_fixed(package, device):
    state=package.state()
    loaded=load_model(package,state,device)
    model,diffusion=loaded[:2]
    if (diffusion.num_timesteps!=1000 or list(diffusion.timestep_map)!=list(range(1000))
            or getattr(diffusion,'rescale_timesteps',False)):
        raise ValueError('Frozen full diffusion schedule required')
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    if any(module.training for module in model.modules()):raise ValueError('Alignment requires all modules eval')
    return state,loaded


def Q(package,args,guard):
    _,(model,diffusion,controlled,encoder,original,handles,backbone)=load_fixed(package,args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    derived=[];within=[];did=[];shared={}
    try:
        with torch.no_grad():
            for group in package.groups:
                rows=package.probes[group['group_index']*5:group['group_index']*5+5]
                if [r['probe_id'] for r in rows]!=group['probe_ids'] or [r['branch'] for r in rows]!=BRANCHES:
                    raise ValueError('Frozen Q branch order mismatch')
                guard.state.update(active_probe_ids=group['probe_ids'],active_probe_forward_started=False);guard.persist()
                items=[package.inputs(r) for r in rows]
                if any(len(i['motion'])!=group['frames'] or not np.array_equal(i['motion'],items[0]['motion']) for i in items):
                    raise ValueError('Q requires five identical targets and lengths')
                x0,y=batch_inputs(items,rows,args.device)
                embed_text(model,y)
                controls=[]
                for i,(item,row) in enumerate(zip(items,rows)):
                    if i<2:
                        controls.append(torch.zeros((1,group['frames'],encoder.config.latent_dim),device=args.device))
                    else:
                        _,one=batch_inputs([item],[row],args.device)
                        control,present=encode_control(encoder,one,mode='static_delta',guard=guard)
                        if not present.all():raise ValueError('Q conditioned branch unexpectedly absent')
                        controls.append(control)
                y['imu_control']=torch.cat(controls);y['imu_control_mode']='static_delta';y.pop('imu_reference')
                spec=package.noise[group['group_id']]
                noise=probe_noise(spec)
                t=torch.full((5,),group['timestep'],device=args.device,dtype=torch.long)
                xt_one=diffusion.q_sample(x0[:1],t[:1],noise=torch.from_numpy(noise).to(args.device))
                xt=xt_one.repeat(5,1,1,1)
                if list(xt.shape)!=[5,263,1,group['frames']]:raise ValueError('Q batch shape mismatch')
                guard.before_q(group['probe_ids'])
                prediction=model(xt,diffusion._scale_timesteps(t),y=y)
                guard.ensure_finite(prediction,'Q prediction')
                if prediction.dtype!=torch.float32 or prediction.shape!=xt.shape:raise ValueError('Q output dtype/shape mismatch')
                pred=numpy(prediction);target=numpy(x0);state=numpy(xt)
                pending=[]
                for index,(row,item) in enumerate(zip(rows,items)):
                    ia=input_arrays(item,row);key=digest({k:describe(v) for k,v in ia.items()})
                    if key not in shared:
                        path=output/'inputs'/(key+'.npz');save_npz(path,ia,guard);shared[key]=artifact(path)
                    arrays=dict(predicted_x0=pred[index],target=target[index],x_t=state[index],
                                frame_mask=numpy(y['imu_frame_mask'][index]))
                    path=output/(row['probe_id']+'.npz');save_npz(path,arrays,guard)
                    record=dict(row,status='completed',package_sha256=PACKAGE_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                        input_artifact=shared[key],input_hashes={k:describe(v) for k,v in ia.items()},input_metadata=item['metadata'],
                        arrays={k:describe(v) for k,v in arrays.items()},noise=spec,actual_noise_sha256=array_sha(noise),
                        scaled_t=numpy(diffusion._scale_timesteps(t[index:index+1])).tolist(),
                        reductions={k:reductions(v) for k,v in arrays.items()},
                        x0_masked_mse=float(np.mean((pred[index].astype(float)-target[index])**2)),
                        diffusion=dict(num_timesteps=diffusion.num_timesteps,timestep_map=list(diffusion.timestep_map),
                                       betas_sha256=array_sha(diffusion.betas),clip_denoised=False))
                    pending.append((path,record))
                static=difference(pred[3],pred[1],enforce=False)
                guard.state['checks'].append(dict(group_id=group['group_id'],static_vs_text=static));guard.persist()
                if not static['passed']:raise ValueError('Q static/text software equality failed')
                analyses={}
                for index,condition in ((2,'paired'),(3,'physical_static'),(4,'tail_reversed')):
                    analysis=reduced_analysis(pred[0],pred[1],pred[index],target[index]);analyses[condition]=analysis
                    derived.append(dict(group_id=group['group_id'],condition=condition,**analysis))
                for other,index in (('physical_static',3),('tail_reversed',4)):
                    contrasts={}
                    for guidance,metric in (('unit','unit_mse'),('legacy_reduced','legacy_reduced_mse')):
                        contrasts[guidance]=analyses['paired'][metric]-analyses[other][metric]
                        within.append(dict(group_id=group['group_id'],guidance=guidance,comparator=other,
                            paired_minus_comparator_mse=contrasts[guidance],
                            prediction_difference=reductions(pred[2].astype(float)-pred[index])))
                    did.append(dict(group_id=group['group_id'],comparator=other,
                                    unit_minus_legacy_contrast=contrasts['unit']-contrasts['legacy_reduced']))
                for path,record in pending:
                    record['group_static_check']=static
                    save_json(path.with_suffix('.json'),record,guard)
                    guard.state['completed_slots'].append(record['probe_id']);guard.persist()
                guard.state['active_probe_ids']=[];guard.persist()
        save_json(output/'derived.json',dict(conditions=derived,within=within,difference_of_differences=did,
            interpretation='Float64 conceptual combinations, Q has no f01; not a float32 CFG replay'),guard)
    finally:
        handles.remove()


def sample(package,args,guard):
    row=package.samples[args.slot]
    if row['slot_index']!=args.slot or row['guidance']!=GUIDANCE:raise ValueError('Frozen S1 slot/guidance mismatch')
    state,(model,diffusion,controlled,encoder,original,handles,backbone)=load_fixed(package,args.device)
    observer=None
    try:
        item=package.inputs(row);_,y=batch_inputs([item],[row],args.device)
        guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched',bypass_encoder=original)
        guided.eval();cache=ConditionCache()
        historical=read(verify(row['historical_metadata']))
        trace_slots=[r for r in package.traces if r['case_id']==row['case_id']]
        if len(trace_slots)!=4 or [r['timestep'] for r in trace_slots]!=[900,500,250,50]:raise ValueError('Four frozen traces required')
        with torch.no_grad():
            embed_text(model,y)
            y['imu_control'],y['imu_condition_present']=cache.get(encoder,y,checkpoint_sha=CHECKPOINT_SHA,mode='static_delta',guard=guard)
            y['imu_control_mode']='static_delta';y.pop('imu_reference')
            y.update(text_scale=1.,imu_scale=1.,joint_scale=1.,
                     backbone_bypass=torch.tensor([row['condition']=='backbone_bypass'],device=args.device))
            metadata=dict(case_id=row['case_id'],package_sha256=PACKAGE_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                comparison_case_id=row['comparison_case_id'],guidance=GUIDANCE,condition=row['condition'],
                text_present=row['text_present'],imu_present=row['imu_present'],
                input_hashes={k:describe(v) for k,v in input_arrays(item,row).items()},
                random_plan_case_id=row['random_plan_case_id'],sampling_seed=row['sampling_seed'],
                random_plan_id=historical['random_plans'][0]['random_plan_id'],
                random_plan_sha256=row['historical_random_plan_sha256'])
            observer=TraceGuidance(guided,model,trace_slots,Path(args.output).parent.parent/'traces',metadata,guard,diffusion._scale_timesteps)
            observer.eval()
            sampling_started=time.monotonic()
            features,plans=sample_addressed_ddpm(observer,diffusion,y=y,lengths=y['lengths'],
                case_ids=[row['random_plan_case_id']],sample_seed=row['sampling_seed'],
                backbone_id=package.common['backbone']['sha256'],run_guard=guard)
            sampling_seconds=time.monotonic()-sampling_started
            if plans!=historical['random_plans'] or digest(plans[0])!=row['historical_random_plan_sha256']:
                raise ValueError('Historical full sampling noise identity mismatch')
            if guard.state['completed_traces']!=[r['trace_id'] for r in trace_slots]:raise ValueError('Missing S1 traces')
            joints=backbone.decode_motion(features);guard.ensure_finite(joints,'S1 decoded joints')
            # Independent CPU decode on the same generated features, not another model call.
            cpu_vectors=features.detach().cpu().permute(0,2,3,1)*backbone.std.detach().cpu()+backbone.mean.detach().cpu()
            cpu_joints=backbone.recover_from_ric(cpu_vectors.float(),22).view_as(joints.detach().cpu())
            decode_check=difference(joints,cpu_joints,atol=2e-6,rtol=1e-6)
        meta=dict(row,status='generated',package_sha256=PACKAGE_SHA,control_checkpoint_sha256=CHECKPOINT_SHA,
            initialization_sha256=state['initialization_sha256'],control_config=state['control_config'],
            training_input_provenance=state['training_input_provenance'],input_metadata=item['metadata'],
            random_plans=plans,guidance=GUIDANCE,device=args.device,condition_cache=True,sampler='ddpm',
            original_orientation0_sha256=array_sha(item['original_orientation0']),mdm_checkpoint_sha256=package.common['backbone']['sha256'],
            frame_count=row['frames'],original_lengths=[row['frames']],effective_lengths=[row['frames']],training_seed=20260925,
            peak_cuda_memory_bytes=guard.state['peak_allocated_bytes'],model_forward_calls=1000,
            MDM_branch_examples=guard.state['mdm_examples'],encoder_example_forwards=guard.state['encoder_examples'],
            contract_version='itm-rework-v0.3-draft',reference_mode='static_delta',trace_ids=guard.state['completed_traces'],
            decode_check=decode_check,sampling_seconds=sampling_seconds)
        arrays=dict(input_arrays(item,row),motion=numpy(joints),normalized_humanml263=numpy(features),fps=np.array(20.),
                    metadata=np.array(__import__('json').dumps(meta)))
        output=Path(args.output);save_npz(output,arrays,guard);save_json(output.with_suffix('.json'),meta,guard)
        guard.state['completed_slots'].append(row['case_id']);guard.persist()
    finally:
        if observer is not None:observer.close()
        handles.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True);parser.add_argument('--run-context',type=Path,required=True)
    parser.add_argument('--phase',choices=['Q','sample'],required=True);parser.add_argument('--slot',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.device='cuda:0';args.output=args.output.resolve();args.package=args.package.resolve()
    context=read(args.run_context.resolve())
    if context['phase']!=args.phase:raise ValueError('Wrong worker phase')
    guard=AlignmentGuard(context)
    try:
        for entry in context['expected_artifacts']:verify(entry);guard.check()
        package=AlignmentPackage(args.package)
        if context['package_sha256']!=PACKAGE_SHA:raise ValueError('Wrong context package')
        guard.configure_cuda(torch,torch.device(args.device))
        {'Q':Q,'sample':sample}[args.phase](package,args,guard)
        guard.finish()
    except BaseException as error:
        guard.fail(error);raise


if __name__=='__main__':main()
