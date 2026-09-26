#!/usr/bin/env python
"""One bounded correction phase. Only CPU fixtures may call helpers directly."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from itm.backbones.base import MotionConditions
from itm.backbones.mdm import MDMBackbone
from itm.models.mdm_correction import (ROOT, PACKAGE_SHA, ARMS, CorrectionPackage, read, verify, array_sha,
    training_noise, probe_noise, encode_control, ConditionCache, accumulate_update, tensor_hashes, environment)
from itm.models.mdm_correction_limits import CorrectionGuard
from itm.models.mdm_imu_control import (MDMIMUControlConfig, install_imu_control, configure_control_training,
                                        make_text_imu_guidance_model)
from itm.models.mdm_run_limits import atomic_artifact, atomic_json, sha256
from itm.models.mdm_sampling import sample_addressed_ddpm


def load_model(package, state, device):
    from train_mdm_imu_control import _legacy_compatibility, _set_mdm_compatibility_defaults
    _legacy_compatibility()
    common = package.common
    mdm_path = verify(common['backbone'])
    args_path = verify(common['training_input_provenance']['payload']['artifacts']['mdm_args'])
    options = read(args_path); _set_mdm_compatibility_defaults(options)
    if options.get('mask_frames', False) or not options['predict_xstart']:
        raise ValueError('Frozen backbone must predict x0 without frame attention masking')
    mdm_root = ROOT/'outputs/mdm/runtime_root'
    os.chdir(mdm_root); sys.path.insert(0, str(mdm_root))
    from utils.model_util import create_model_and_diffusion, load_model_wo_clip
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    model, diffusion = create_model_and_diffusion(SimpleNamespace(**options), SimpleNamespace(dataset=SimpleNamespace(num_actions=1)))
    load_model_wo_clip(model, torch.load(mdm_path, map_location='cpu', weights_only=True))
    if diffusion.num_timesteps != 1000:
        raise ValueError('Frozen 1000-step diffusion required')
    original = model.seqTransEncoder
    config = MDMIMUControlConfig(**state['control_config'])
    controlled, encoder, handles = install_imu_control(model, config)
    encoder.load_state_dict(state['imu_encoder']); controlled.adapters.load_state_dict(state['adapters'])
    # Upstream MDM._apply/train return None; neither call supports chaining.
    model.to(device)
    model.eval()
    original.eval()
    backbone = MDMBackbone(model, diffusion, torch.from_numpy(package.mean).to(device),
                           torch.from_numpy(package.std).to(device), recover_from_ric)
    return model, diffusion, controlled, encoder, original, handles, backbone


def batch_inputs(items, rows, device, frames=None):
    lengths = [len(i['motion']) for i in items]
    frames = frames or max(lengths)
    def pad(array, n):
        result = np.zeros((n,)+array.shape[1:], dtype=array.dtype); result[:len(array)] = array
        return result
    x0 = torch.from_numpy(np.stack([pad(i['motion'], frames).T[:, None] for i in items])).to(device)
    valid = torch.arange(frames, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
    y = dict(imu=torch.from_numpy(np.stack([pad(i['imu'], frames) for i in items])).to(device),
             imu_reference=torch.from_numpy(np.stack([pad(i['reference'], frames) for i in items])).to(device),
             sensor_mask=torch.from_numpy(np.stack([i['sensor_mask'] for i in items])).to(device),
             imu_frame_mask=valid, imu_timestamps=torch.from_numpy(np.stack([pad(i['timestamps'], frames) for i in items])).to(device),
             imu_condition_present=torch.tensor([r.get('imu_present', True) for r in rows], device=device),
             text_condition_present=torch.tensor([r.get('text_present', True) for r in rows], device=device),
             mask=valid[:, None, None], lengths=torch.tensor(lengths, device=device),
             text=[i['text'] for i in items], uncond=False)
    return x0, y


def embed_text(model, y):
    with torch.no_grad():
        y['text_embed'] = model.encode_text(y['text'])
        y['text_embed'] = torch.where(y['text_condition_present'][None, :, None], y['text_embed'], 0)


def put_control(encoder, y, mode, seed, guard):
    y['imu_control'], y['imu_condition_present'] = encode_control(encoder, y, mode=mode, seed=seed, guard=guard)
    y['imu_control_mode'] = mode


def train(package, args, guard):
    initial = torch.load(args.initial, map_location='cpu', weights_only=True)
    if initial['tensor_hashes'] != tensor_hashes(initial) or initial['environment'] != environment():
        raise ValueError('Initial tensor/environment identity mismatch')
    if initial['package_sha256'] != PACKAGE_SHA:
        raise ValueError('Initial package mismatch')
    mode = 'static_delta' if args.arm == ARMS[1] else 'none'
    state = dict(initial, control_config=asdict(replace(MDMIMUControlConfig(**initial['control_config']), reference_mode=mode)))
    model, diffusion, controlled, encoder, original, handles, backbone = load_model(package, state, args.device)
    configure_control_training(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    defaults = dict(package.common['optimizer']); defaults.pop('name'); defaults['betas'] = tuple(defaults['betas'])
    optimizer = torch.optim.AdamW(parameters, **defaults)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    # Decode-only input assembly, verified once per admitted clip/config. No model calls.
    items = {(r['motion_id'], r['sensor_config']): package.inputs(r) for r in package.schedule[:6]}
    for update in range(1, 201):
        rows = package.schedule[(update-1)*3:update*3]
        if [r['update'] for r in rows] != [update]*3:
            raise ValueError('Frozen update ordering mismatch')
        receipts = []
        def compute(row):
            item = items[(row['motion_id'], row['sensor_config'])]
            x0, y = batch_inputs([item], [row], args.device)
            embed_text(model, y)
            put_control(encoder, y, mode, row['encoder_dropout_seed'], guard)
            noise = torch.from_numpy(training_noise(row)).to(args.device)
            t = torch.tensor([row['timestep']], device=args.device)
            x_t = diffusion.q_sample(x0, t, noise=noise)
            guard.reserve('exposures'); guard.reserve('mdm_examples')
            prediction = model(x_t, diffusion._scale_timesteps(t), y=y)
            guard.ensure_finite(prediction, 'training prediction')
            loss = (prediction-x0).square().mean()
            receipts.append(dict(**row, actual_shape=list(x0.shape), actual_noise_sha256=array_sha(noise),
                                 loss=float(loss.detach()), has_gradient_graph=loss.requires_grad))
            return loss
        losses = accumulate_update([lambda r=r: compute(r) for r in rows], optimizer, parameters, guard)
        record = dict(update=update, examples=receipts, losses=losses, optimizer_step_completed=True)
        atomic_artifact(out/f'update_{update:03d}.json', lambda f: f.write(json.dumps(record, allow_nan=False).encode()), guard)
        if update in (50, 100, 150, 200):
            payload = dict(checkpoint_schema='itm-correction-control-v1', correction_arm=args.arm,
                package_sha256=PACKAGE_SHA, initialization_sha256=sha256(args.initial), initial_tensor_hashes=initial['tensor_hashes'],
                control_config=asdict(encoder.config), imu_encoder=encoder.state_dict(), adapters=controlled.adapters.state_dict(),
                optimizer=optimizer.state_dict(), optimizer_defaults=optimizer.defaults, environment=environment(),
                imu_normalization=package.common['imu_normalization'], training_input_provenance=package.common['training_input_provenance'],
                execution=dict(updates_completed=update, updates_attempted=guard.state['updates_attempted'],
                               final_update_200=update == 200, exposures=guard.state['exposures'],
                               mdm_examples=guard.state['mdm_examples'], encoder_examples=guard.state['encoder_examples']))
            path = out/('model.pt' if update == 200 else f'update_{update:03d}.pt')
            atomic_artifact(path, lambda f: torch.save(payload, f), guard)
    handles.remove()


class ProbeHooks:
    def __init__(self, encoder, controlled):
        self.values = {}; self.handles = []
        modules = {'encoder_fused': encoder.fused_observer, 'encoder_timed': encoder.timed_observer, 'encoder_final': encoder}
        modules.update({f'encoder_layer_{i}': layer for i, layer in enumerate(encoder.temporal_encoder.layers)})
        modules.update({f'hidden_{i}': layer for i, layer in enumerate(controlled.hidden_observers)})
        modules.update({f'residual_{i}': layer for i, layer in enumerate(controlled.residual_observers)})
        for name, module in modules.items():
            def capture(module, inputs, output, key=name):
                self.values[key] = (output[0] if isinstance(output, tuple) else output).detach().clone()
            self.handles.append(module.register_forward_hook(capture))

    def close(self):
        for handle in self.handles: handle.remove()


def reductions(value):
    v = np.asarray(value, dtype=np.float64)
    if not np.isfinite(v).all(): raise FloatingPointError('nonfinite probe activation')
    return dict(count=int(v.size), finite_count=int(np.isfinite(v).sum()), sums_sq=float(np.square(v).sum()),
                rms=float(np.sqrt(np.square(v).mean())), max_abs=float(np.abs(v).max()))


def probe_groups(rows):
    groups = []
    padded = {}
    for row in rows:
        if row['context'] == 'exact_length': groups.append([row])
        else:
            key = (row['sensor_config'], row['timestep'], row['noise_seed'], row['condition'])
            padded.setdefault(key, []).append(row)
    for group in padded.values():
        group.sort(key=lambda r:r['motion_id'])
        if [r['motion_id'] for r in group] != ['000027', '000506', '004728']:
            raise ValueError('Frozen padded group mismatch')
        groups.append(group)
    return groups


def probe(package, args, guard):
    state = torch.load(verify(package.design['D']['checkpoint']), map_location='cpu', weights_only=True)
    model, diffusion, controlled, encoder, original, handles, backbone = load_model(package, state, args.device)
    hooks = ProbeHooks(encoder, controlled)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    try:
        with torch.no_grad():
            for group in probe_groups(package.probes):
                guard.state.update(active_probe_ids=[r['probe_id'] for r in group],active_probe_forward_started=False)
                guard.persist()
                items = [package.inputs(row) for row in group]
                x0, y = batch_inputs(items, group, args.device, group[0]['context_frames'])
                noise = torch.from_numpy(np.concatenate([probe_noise(package.noise[row['noise_key']], row['context'] != 'exact_length') for row in group])).to(args.device)
                t = torch.full((len(group),), group[0]['timestep'], device=args.device, dtype=torch.long)
                x_t = diffusion.q_sample(x0, t, noise=noise)
                embed_text(model, y); hooks.values.clear()
                # Use raw encoder hook within this one MDM call, including null E.
                guard.reserve('encoder_examples', len(group)); guard.reserve('mdm_examples', len(group))
                guard.state['active_probe_forward_started']=True;guard.persist()
                prediction = model(x_t, diffusion._scale_timesteps(t), y=y)
                guard.ensure_finite(prediction, 'probe x0')
                for i, row in enumerate(group):
                    n = row['frames']
                    arrays = dict(predicted_x0=prediction[i, :, :, :n].cpu().numpy(), target=x0[i, :, :, :n].cpu().numpy(),
                                  x_t=x_t[i, :, :, :n].cpu().numpy(), frame_mask=y['imu_frame_mask'][i, :n].cpu().numpy())
                    for name, value in hooks.values.items():
                        v = value[1:n+1, i] if name.startswith(('hidden_', 'residual_')) else value[i, :n]
                        arrays[name] = v.cpu().numpy().astype(np.float32)
                    if len(hooks.values) != 21:
                        raise ValueError('Missing probe hooks')
                    stats = {k:reductions(v) for k,v in arrays.items() if k != 'frame_mask'}
                    ratios = {str(j):(stats[f'residual_{j}']['rms']/stats[f'hidden_{j}']['rms']
                                if stats[f'hidden_{j}']['rms'] else None) for j in range(8)}
                    record = dict(**dict(row,status='completed'), x_t_sha256=array_sha(arrays['x_t']), target_sha256=array_sha(arrays['target']),
                                  noise=package.noise[row['noise_key']], scaled_t=float(diffusion._scale_timesteps(t)[i]),
                                  x0_masked_mse=float(np.mean((arrays['predicted_x0'].astype(np.float64)-arrays['target'])**2)),
                                  reductions=stats, residual_hidden_ratios=ratios,
                                  diffusion=dict(num_timesteps=diffusion.num_timesteps, betas=diffusion.betas.tolist(),
                                                 timestep_map=list(diffusion.timestep_map)))
                    path = out/(row['probe_id']+'.npz')
                    atomic_artifact(path, lambda f:np.savez_compressed(f, **arrays), guard)
                    atomic_artifact(path.with_suffix('.json'), lambda f:f.write(json.dumps(record, allow_nan=False).encode()), guard)
                    guard.state['completed_slots'].append(row['probe_id']); guard.persist(); guard.check()
        probe_differences(package.probes, out, guard)
    finally:
        hooks.close(); handles.remove()



def probe_differences(rows, out, guard):
    """CPU reductions of the saved same-forward arrays; never invokes a model."""
    index = {(r['split'],r['motion_id'],r['sensor_config'],r['timestep'],r['noise_seed'],r['context'],r['condition']):r for r in rows}
    results=[]
    for key,row in index.items():
        comparators=[]
        if row['condition'] != 'text_only': comparators.append(key[:-1]+('text_only',))
        if row['condition']=='paired':
            for condition in ('physical_static','tail_reversed'):
                candidate=key[:-1]+(condition,)
                if candidate in index: comparators.append(candidate)
        if row['context']!='exact_length':comparators.append(key[:-2]+('exact_length',key[-1]))
        for other in comparators:
            guard.check()
            baseline=index[other]
            with np.load(out/(row['probe_id']+'.npz'),allow_pickle=False) as a, np.load(out/(baseline['probe_id']+'.npz'),allow_pickle=False) as b:
                delta={name:reductions(a[name].astype(np.float64)-b[name]) for name in a.files if name!='frame_mask'}
                mse_a=float(np.mean((a['predicted_x0'].astype(np.float64)-a['target'])**2))
                mse_b=float(np.mean((b['predicted_x0'].astype(np.float64)-b['target'])**2))
                results.append(dict(probe_id=row['probe_id'],reference_probe_id=baseline['probe_id'],differences=delta,
                                    x0_mse_difference=mse_a-mse_b,no_added_padding=row['frames']==158))
    atomic_artifact(out/'paired_differences.json',lambda f:f.write(json.dumps(results,allow_nan=False).encode()),guard)


def sample(package, args, guard):
    row = package.samples[args.slot]
    if row['arm'] != args.arm:
        raise ValueError('Sample arm/slot mismatch')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if (state['correction_arm'] != args.arm or state['package_sha256'] != PACKAGE_SHA or
            state['initialization_sha256'] != guard.context['initialization_sha256'] or
            state['environment'] != environment() or not state['execution']['final_update_200'] or state['execution']['updates_completed'] != 200):
        raise ValueError('Sample must use the bound arm final200')
    model, diffusion, controlled, encoder, original, handles, backbone = load_model(package, state, args.device)
    item = package.inputs(row)
    x0, y = batch_inputs([item], [row], args.device)
    guided = make_text_imu_guidance_model(model, mode='factorized', branch_execution='batched', bypass_encoder=original)
    guided.eval(); backbone.guidance_model = guided
    mode = encoder.config.reference_mode
    cache = ConditionCache()
    sampling_start=time.monotonic()
    with torch.no_grad():
        embed_text(model, y)
        y['imu_control'], y['imu_condition_present'] = cache.get(encoder, y, checkpoint_sha=sha256(args.checkpoint), mode=mode, guard=guard)
        y['imu_control_mode'] = mode
        # Raw/reference arrays are saved separately; only cached embeddings enter CFG.
        y.pop('imu_reference')
        y.update(text_scale=2.5, imu_scale=1., joint_scale=1., backbone_bypass=torch.tensor([row['condition']=='backbone_bypass'], device=args.device))
        features, plans = sample_addressed_ddpm(guided, diffusion, y=y, lengths=y['lengths'],
                case_ids=[row['random_plan_case_id']], sample_seed=row['sampling_seed'],
                backbone_id=package.common['backbone']['sha256'], run_guard=guard)
        joints = backbone.decode_motion(features)
        guard.ensure_finite(joints, 'decoded joints')
    meta = dict(**dict(row,status='generated'), package_sha256=PACKAGE_SHA, control_checkpoint_sha256=sha256(args.checkpoint),
                initialization_sha256=state['initialization_sha256'], control_config=state['control_config'],
                input_metadata=item['metadata'], training_input_provenance=state['training_input_provenance'],
                random_plans=plans, device=args.device, condition_cache=True, sampler='ddpm',
                guidance=dict(text_scale=2.5,imu_scale=1.,joint_scale=1.,mode='factorized',branch_execution='batched'),
                original_orientation0_sha256=array_sha(item['original_orientation0']),
                mdm_checkpoint_sha256=package.common['backbone']['sha256'], frame_count=row['frames'],
                original_lengths=[row['frames']],effective_lengths=[row['frames']],training_seed=20260925,
                sampling_seconds=time.monotonic()-sampling_start,peak_cuda_memory_bytes=guard.state['peak_allocated_bytes'],
                model_forward_calls=1000,MDM_branch_examples=guard.state['mdm_examples'],
                encoder_example_forwards=guard.state['encoder_examples'],
                contract_version='itm-rework-v0.3-draft', reference_mode=mode)
    arrays = dict(motion=joints.cpu().numpy(), normalized_humanml263=features.cpu().numpy(), gt=item['gt'][None],
                  acceleration=item['raw_acceleration'][None], orientation=item['raw_orientation'][None],
                  normalized_imu=item['imu'][None], normalized_reference_imu=item['reference'][None],
                  original_orientation0=item['original_orientation0'], sensor_mask=item['sensor_mask'][None],
                  timestamps_seconds=item['timestamps'], frame_mask=y['imu_frame_mask'].cpu().numpy(), fps=np.array(20.),
                  text_condition_present=y['text_condition_present'].cpu().numpy(), imu_condition_present=y['imu_condition_present'].cpu().numpy(),
                  metadata=np.array(json.dumps(meta)))
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise FloatingPointError('nonfinite output '+key)
    output = Path(args.output)
    atomic_artifact(output, lambda f:np.savez_compressed(f, **arrays), guard)
    atomic_artifact(output.with_suffix('.json'), lambda f:f.write(json.dumps(meta, allow_nan=False).encode()), guard)
    guard.state['completed_slots'].append(row['case_id']); guard.persist()
    handles.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--run-context', type=Path, required=True)
    parser.add_argument('--phase', choices=['probe','train','sample'], required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--slot', type=int)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); args.device = 'cuda:0'
    args.output = str(Path(args.output).resolve())
    for key in ('package','initial','checkpoint','run_context'):
        if getattr(args,key) is not None: setattr(args,key,getattr(args,key).resolve())
    context = read(args.run_context)
    if context['phase'] != args.phase or context.get('arm') != args.arm:
        raise ValueError('Worker/context mismatch')
    guard = CorrectionGuard(context)
    try:
        for entry in context['expected_artifacts']: verify(entry)
        guard.check()
        package = CorrectionPackage(args.package)
        guard.configure_cuda(torch, torch.device(args.device))
        {'probe':probe,'train':train,'sample':sample}[args.phase](package,args,guard)
        guard.finish()
    except BaseException as error:
        guard.fail(error); raise


if __name__ == '__main__': main()
