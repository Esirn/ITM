#!/usr/bin/env python
"""One bounded F train/sample phase; CPU tests substitute only tiny fixtures."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from itm.models.mdm_correction import (read, verify, artifact, array_sha, training_noise, tensor_hashes,
    accumulate_update, ConditionCache)
from itm.models.mdm_imu_control import (MDMIMUControlConfig, configure_control_training, make_text_imu_guidance_model)
from itm.models.mdm_run_limits import atomic_artifact, sha256
from itm.models.mdm_sampling import sample_addressed_ddpm
from itm.models.mdm_sampling_alignment import (GUIDANCE, TraceGuidance, input_arrays, describe, digest,
    difference, save_npz, save_json, numpy)
from itm.models.mdm_feature_balance import (FeaturePackage, FeatureGuard, ForwardAccounting, PACKAGE_SHA,
    ARM, SCHEMA, feature_losses, audit_training, check_frozen)
from run_mdm_correction_worker import load_model, batch_inputs, embed_text, put_control


def counted_text(model,y,guard):
    # embed_text returns None; hook the same one batch1 encode_text call explicitly.
    guard.enter('caption',1)
    embed_text(model,y)
    guard.returned('caption',1,y['text_embed'])


def train(package,args,guard):
    initial=package.initial_state();identity=package.identity(guard.context['implementation_sha256'])
    config=asdict(MDMIMUControlConfig(**dict(package.common['control_config'],reference_mode='static_delta')))
    state=dict(initial,control_config=config)
    model,diffusion,controlled,encoder,original,handles,backbone=load_model(package,state,args.device)
    accounting=None
    try:
        if tensor_hashes(dict(imu_encoder=encoder.state_dict(),adapters=controlled.adapters.state_dict()))!=initial['tensor_hashes']:
            raise ValueError('Loaded F initialization differs')
        configure_control_training(model);audit=audit_training(model,initial)
        parameters=[p for p in model.parameters() if p.requires_grad]
        defaults=dict(package.common['optimizer']);defaults.pop('name');defaults['betas']=tuple(defaults['betas'])
        optimizer=torch.optim.AdamW(parameters,**defaults)
        if optimizer.state or optimizer.defaults!=defaults:raise ValueError('F requires fresh identical AdamW')
        out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
        accounting=ForwardAccounting(model,encoder,guard)
        items={(r['motion_id'],r['sensor_config']):package.inputs(r) for r in package.schedule[:6]}
        for update in range(1,201):
            rows=package.schedule[(update-1)*3:update*3]
            if [r['update'] for r in rows]!=[update]*3:raise ValueError('F update ordering changed')
            guard.state.update(active_update=update,active_exposure=None);guard.persist()
            receipts=[]
            def compute(row):
                guard.state['active_exposure']=row['exposure_id'];guard.state['exposures']+=1;guard.persist()
                item=items[(row['motion_id'],row['sensor_config'])]
                x0,y=batch_inputs([item],[row],args.device)
                counted_text(model,y,guard)
                put_control(encoder,y,'static_delta',row['encoder_dropout_seed'],None)
                noise=torch.from_numpy(training_noise(row)).to(args.device)
                t=torch.tensor([row['timestep']],device=args.device)
                x_t=diffusion.q_sample(x0,t,noise=noise)
                prediction=model(x_t,diffusion._scale_timesteps(t),y=y)
                if prediction.dtype!=torch.float32:raise ValueError('F production requires float32')
                losses=feature_losses(prediction,x0,y['imu_frame_mask'])
                guard.ensure_finite(losses,'F losses');loss=losses['balanced'].mean()
                record=dict(row,actual_shape=list(x0.shape),actual_noise_sha256=array_sha(noise),
                    loss=float(loss.detach()),loss_divided_by_three=float((loss/3).detach()),
                    loss_components={k:float(v.detach().mean()) for k,v in losses.items()},
                    has_gradient_graph=loss.requires_grad,backward_skipped=not loss.requires_grad)
                # Preserve same-forward finite diagnostics even if backward/step later fails.
                save_json(out/'exposures'/(row['exposure_id']+'.json'),record,guard)
                receipts.append(record)
                return loss
            losses=accumulate_update([lambda r=r:compute(r) for r in rows],optimizer,parameters,guard)
            save_json(out/f'update_{update:03d}.json',dict(update=update,examples=receipts,losses=losses,
                optimizer_step_completed=True),guard)
            guard.state['completed_slots'].append(f'U{update:03d}');guard.persist()
            if update in (50,100,150,200):
                check_frozen(model,audit)
                payload=dict(identity,checkpoint_schema=SCHEMA,arm=ARM,control_config=config,
                    training_run_directory=guard.context['run_directory'],
                    imu_encoder=encoder.state_dict(),adapters=controlled.adapters.state_dict(),
                    optimizer=optimizer.state_dict(),optimizer_defaults=optimizer.defaults,training_audit=audit,
                    execution=dict(updates_completed=update,updates_attempted=guard.state['updates_attempted'],
                        final_update_200=update==200,exposures=guard.state['exposures'],
                        mdm_examples=guard.state['mdm_examples'],encoder_examples=guard.state['encoder_examples'],
                        caption_calls=guard.state['caption_calls']))
                payload['tensor_hashes']=tensor_hashes(payload)
                atomic_artifact(out/('model.pt' if update==200 else f'update_{update:03d}.pt'),lambda f:torch.save(payload,f),guard)
    finally:
        if accounting:accounting.close()
        handles.remove()


def load_final(package,args,guard):
    checkpoint=guard.context['checkpoint'];state=package.state(checkpoint,guard.context['implementation_sha256'])
    if state['training_run_directory']!=guard.context['run_directory']:raise ValueError('F checkpoint run mismatch')
    loaded=load_model(package,state,args.device);model,diffusion=loaded[:2]
    if (diffusion.num_timesteps!=1000 or list(diffusion.timestep_map)!=list(range(1000))
            or getattr(diffusion,'rescale_timesteps',False)):
        raise ValueError('F full diffusion schedule changed')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    if any(m.training for m in model.modules()):raise ValueError('F sampling requires eval')
    if any(p.dtype!=torch.float32 for n,p in model.named_parameters() if not n.startswith('clip_model.')):
        raise ValueError('F sampling denoiser/control precision changed')
    return state,loaded


def sample(package,args,guard):
    CHECKPOINT_SHA=guard.context['checkpoint']['sha256']
    row=package.samples[args.slot]
    if row['slot_index']!=args.slot or row['guidance']!=GUIDANCE:raise ValueError('Frozen S1 slot/guidance mismatch')
    state,(model,diffusion,controlled,encoder,original,handles,backbone)=load_final(package,args,guard)
    observer=None
    accounting=ForwardAccounting(model,encoder,guard)
    try:
        item=package.inputs(row);_,y=batch_inputs([item],[row],args.device)
        guided=make_text_imu_guidance_model(model,mode='factorized',branch_execution='batched',bypass_encoder=original)
        guided.eval();cache=ConditionCache()
        historical=read(verify(row['historical_metadata']))
        trace_slots=[r for r in package.traces if r['case_id']==row['case_id']]
        if len(trace_slots)!=4 or [r['timestep'] for r in trace_slots]!=[900,500,250,50]:raise ValueError('Four frozen traces required')
        with torch.no_grad():
            counted_text(model,y,guard)
            y['imu_control'],y['imu_condition_present']=cache.get(encoder,y,checkpoint_sha=CHECKPOINT_SHA,mode='static_delta',guard=None)
            y['imu_control_mode']='static_delta';y.pop('imu_reference')
            y.update(text_scale=1.,imu_scale=1.,joint_scale=1.,
                     backbone_bypass=torch.tensor([row['condition']=='backbone_bypass'],device=args.device))
            metadata=dict(implementation_sha256=guard.context['implementation_sha256'],loss_version=state['loss_version'],loss_weights_sha256=state['loss_weights_sha256'],case_id=row['case_id'],package_sha256=PACKAGE_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                comparison_case_id=row['historical_case_id'],guidance=GUIDANCE,condition=row['condition'],
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
            # Preserve finite generated features before provenance/decode decisions.
            # This file is diagnostic-only; the final sample/receipt determines acceptance.
            raw_path=Path(args.output).parent.parent/'raw_samples'/(row['case_id']+'.npz')
            save_npz(raw_path,dict(normalized_humanml263=numpy(features),
                metadata=np.array(__import__('json').dumps(dict(metadata,status='raw_before_validation',
                    diagnostic_only=True,final_sample=str(Path(args.output)),loss_version=state['loss_version'])))),guard)
            if plans!=historical['random_plans'] or digest(plans[0])!=row['historical_random_plan_sha256']:
                raise ValueError('Historical full sampling noise identity mismatch')
            if guard.state['completed_traces']!=[r['trace_id'] for r in trace_slots]:raise ValueError('Missing S1 traces')
            joints=backbone.decode_motion(features);guard.ensure_finite(joints,'S1 decoded joints')
            # Independent CPU decode on the same generated features, not another model call.
            cpu_vectors=features.detach().cpu().permute(0,2,3,1)*backbone.std.detach().cpu()+backbone.mean.detach().cpu()
            cpu_joints=backbone.recover_from_ric(cpu_vectors.float(),22).view_as(joints.detach().cpu())
            decode_check=difference(joints,cpu_joints,atol=2e-6,rtol=1e-6)
        meta=dict(row,implementation_sha256=guard.context['implementation_sha256'],loss_weights=state['loss_weights'],loss_weights_sha256=state['loss_weights_sha256'],training_schedule_sha256=state['training_schedule_sha256'],initial_tensor_hashes=state['initial_tensor_hashes'],training_audit=state['training_audit'],status='generated',package_sha256=PACKAGE_SHA,control_checkpoint_sha256=CHECKPOINT_SHA,
            checkpoint_sha256=CHECKPOINT_SHA,checkpoint_schema=SCHEMA,initial_package_sha256=state['initial_package_sha256'],
            training_execution=state['execution'],optimizer_defaults=state['optimizer_defaults'],environment=state['environment'],
            training_run_directory=state['training_run_directory'],
            initialization_sha256=state['initialization_sha256'],control_config=state['control_config'],
            training_input_provenance=state['training_input_provenance'],input_metadata=item['metadata'],
            random_plans=plans,guidance=GUIDANCE,device=args.device,condition_cache=True,sampler='ddpm',
            original_orientation0_sha256=array_sha(item['original_orientation0']),mdm_checkpoint_sha256=package.common['backbone']['sha256'],
            frame_count=row['frames'],original_lengths=[row['frames']],effective_lengths=[row['frames']],training_seed=20260925,
            peak_cuda_memory_bytes=guard.state['peak_allocated_bytes'],model_forward_calls=1000,
            MDM_branch_examples=guard.state['mdm_examples'],encoder_example_forwards=guard.state['encoder_examples'],
            contract_version='itm-rework-v0.3-draft',reference_mode='static_delta',trace_ids=guard.state['completed_traces'],
            decode_check=decode_check,sampling_seconds=sampling_seconds,raw_generated_features=artifact(raw_path))
        arrays=dict(input_arrays(item,row),motion=numpy(joints),normalized_humanml263=numpy(features),fps=np.array(20.),
                    metadata=np.array(__import__('json').dumps(meta)))
        output=Path(args.output);save_npz(output,arrays,guard);save_json(output.with_suffix('.json'),meta,guard)
        guard.state['completed_slots'].append(row['case_id']);guard.persist()
    finally:
        if observer is not None:observer.close()
        accounting.close()
        handles.remove()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--run-context',type=Path,required=True)
    p.add_argument('--phase',choices=['train','sample'],required=True);p.add_argument('--slot',type=int)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.device='cuda:0';args.output=args.output.resolve()
    context=read(args.run_context.resolve())
    if context['phase']!=args.phase:raise ValueError('F worker phase mismatch')
    guard=FeatureGuard(context)
    try:
        for entry in context['expected_artifacts']:verify(entry);guard.check()
        package=FeaturePackage(args.package)
        guard.configure_cuda(torch,torch.device(args.device))
        {'train':train,'sample':sample}[args.phase](package,args,guard)
        guard.finish()
    except BaseException as error:
        guard.fail(error);raise

if __name__=='__main__':main()
