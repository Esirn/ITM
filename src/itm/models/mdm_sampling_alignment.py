"""Frozen fixed-R sampling experiment: inputs, arithmetic and zero-forward tracing.

This module does not grant authorization or initialize a GPU during preparation.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from itm.models.mdm_correction import (ROOT, PACKAGE_SHA as OLD_PACKAGE_SHA, CorrectionPackage,
    array_sha, environment, read, verify)
from itm.models.mdm_run_limits import RunGuard, RunLimitExceeded, atomic_artifact, sha256
from itm.models.mdm_imu_control import MDMIMUControlConfig

PACKAGE_SHA = '537895c45695a48cb4b5114941e4bf8cab3962c40be893fcc79b4c9eeb311d60'
CHECKPOINT_SHA = '91744ac5a663d5b5279922fdf26e0792bf327bd79ee3eae035e065df8c13a642'
INITIAL_SHA = '3ad3c1bca7ff7398386662e4831eb29136d3c0f2e06e5abbea8e7c4cb13d0401'
PACKAGE = ROOT/'docs/itm_rework/protocol/b0_sampling_alignment_20260927_v1/package.json'
GUIDANCE = dict(text_scale=1., imu_scale=1., joint_scale=1., mode='factorized', branch_execution='batched')
BRANCHES = ['f00', 'f10', 'f11_paired', 'f11_static', 'f11_reversed']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def numpy(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value)


def describe(value):
    value = numpy(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=array_sha(value))


def numeric(value):
    value = numpy(value).astype(np.float64)
    if not value.size or not np.isfinite(value).all():
        raise FloatingPointError('Empty/nonfinite alignment array')
    return value


def reductions(value):
    value = numeric(value)
    result = dict(count=value.size, finite_count=int(np.isfinite(value).sum()),
                  sum_squares=float(np.sum(value**2)), rms=float(np.sqrt(np.mean(value**2))),
                  max_abs=float(np.max(np.abs(value))))
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite alignment reduction')
    return result


def difference(a, b, *, atol=1e-6, rtol=1e-6, enforce=True):
    raw_a,raw_b=numpy(a),numpy(b)
    bitwise=raw_a.dtype==raw_b.dtype and raw_a.shape==raw_b.shape and array_sha(raw_a)==array_sha(raw_b)
    a, b = numeric(a), numeric(b)
    if a.shape != b.shape:
        raise ValueError('Comparison shape mismatch')
    delta = a-b
    result = dict(max_abs=float(np.max(np.abs(delta))), rms=reductions(delta)['rms'],
                  bitwise=bitwise, count=a.size,
                  passed=bool(np.all(np.abs(delta) <= atol+rtol*np.abs(b))))
    if enforce and not result['passed']:
        raise ValueError('Frozen equality tolerance exceeded: '+str(result))
    return result


def reduced_analysis(f00, f10, f11, target):
    f00, f10, f11, target = map(numeric, (f00,f10,f11,target))
    if len({a.shape for a in (f00,f10,f11,target)}) != 1:
        raise ValueError('Q shape mismatch')
    a=f11-target; b=1.5*(f10-f00)
    unit=float(np.mean(a*a)); legacy=float(np.mean((a+b)**2))
    cross=float(2*np.mean(a*b)); energy=float(np.mean(b*b))
    error=abs((legacy-unit)-(cross+energy))
    bound=64*np.finfo(np.float64).eps*max(1.,unit,legacy,abs(cross),energy)
    result=dict(unit_mse=unit,legacy_reduced_mse=legacy,legacy_minus_unit=legacy-unit,
        twice_error_text_inner_product=cross,text_energy=energy,decomposition_error=error,
        decomposition_bound=bound,text_increment=reductions(b))
    if not np.isfinite([unit,legacy,cross,energy,error,bound]).all():
        raise FloatingPointError('Nonfinite Q arithmetic')
    if error>bound: raise ValueError('Frozen float64 identity bound exceeded')
    return result


def combination_check(branches, actual, wt=1., *, enforce=True):
    a,b,c,d=map(numeric,branches);actual=numeric(actual)
    if len({v.shape for v in (a,b,c,d,actual)})!=1:raise ValueError('Trace shape mismatch')
    full=a+wt*(b-a)+(c-a)+(d-b-c+a)
    reduced=d+(wt-1)*(b-a)
    scale=np.abs(a)+abs(wt)*(np.abs(b)+np.abs(a))+(np.abs(c)+np.abs(a))+(np.abs(d)+np.abs(b)+np.abs(c)+np.abs(a))
    u=2.**-24;bound=(16*u/(1-16*u))*np.maximum(1.,scale)
    if not np.isfinite(bound).all():raise FloatingPointError('Nonfinite rounding bound')
    reports={}
    for name, reference in (('factorized_float64',full),('reduced_float64',reduced)):
        err=np.abs(actual-reference)
        reports[name]=dict(**difference(actual,reference,enforce=False),max_error_over_bound=float(np.max(err/bound)))
        reports[name]['passed']=bool(np.all(err<=bound))
        if enforce and not reports[name]['passed']:raise ValueError('Frozen float32 combination bound exceeded: '+name)
    return reports


def input_arrays(item, row):
    return dict(acceleration=item['raw_acceleration'][None],orientation=item['raw_orientation'][None],
        normalized_imu=item['imu'][None],normalized_reference_imu=item['reference'][None],
        original_orientation0=item['original_orientation0'],sensor_mask=item['sensor_mask'][None],
        timestamps_seconds=item['timestamps'],frame_mask=np.ones((1,len(item['motion'])),bool),gt=item['gt'][None],
        text_condition_present=np.array([row['text_present']],bool),imu_condition_present=np.array([row['imu_present']],bool))


class AlignmentPackage:
    def __init__(self, path=PACKAGE):
        self.path=Path(path).resolve()
        if sha256(self.path)!=PACKAGE_SHA:raise ValueError('Exact frozen alignment package required')
        self.package=read(self.path)
        self.files={Path(a['path']).stem:verify(a) for a in self.package['artifacts']}
        self.sources=read(self.files['sources'])
        self.source_paths=[verify(a) for a in self.sources]
        self.design=read(self.files['design']);self.common=self.design['common']
        self.old_manifest=read(verify(self.design['old_implementation']))
        self.old=CorrectionPackage(verify(self.design['old_package']))
        self.mean,self.std=self.old.mean,self.old.std
        self.probes=read(self.files['probe_slots']);self.groups=read(self.files['probe_groups'])
        self.samples=read(self.files['sampling_slots']);self.traces=read(self.files['trace_slots'])
        self.noise=read(self.files['probe_noise']);self.budget=read(self.files['budget'])
        self.checkpoint=verify(self.design['checkpoint'])
        if sha256(self.checkpoint)!=CHECKPOINT_SHA:raise ValueError('Wrong fixed R checkpoint')
        if (len(self.probes),len(self.groups),len(self.samples),len(self.traces))!=(480,96,60,240):
            raise ValueError('Frozen denominator mismatch')
        self._inputs={}

    def inputs(self, row):
        condition=row.get('input_transform',row['condition'])
        key=(row.get('data_split',row.get('split')),row['motion_id'],row['sensor_config'],condition)
        if key not in self._inputs:self._inputs[key]=self.old.inputs(dict(row,condition=condition))
        item=self._inputs[key]
        if row['text']!=item['text']:raise ValueError('Caption changed')
        if 'historical_input_hashes' in row:
            arrays=input_arrays(item,row)
            with np.load(verify(row['historical_output']),allow_pickle=False) as old:
                for key, spec in row['historical_input_hashes'].items():
                    if describe(old[key])!=spec:raise ValueError('Historical input identity changed')
                    a,b=arrays[key],old[key]
                    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Input shape/dtype changed: '+key)
                    if key in ('orientation','normalized_imu','normalized_reference_imu'):
                        difference(a,b,atol=2e-7,rtol=0)
                    elif not np.array_equal(a,b):raise ValueError('Exact input mismatch: '+key)
            meta=read(verify(row['historical_metadata']))
            if digest(meta['random_plans'][0])!=row['historical_random_plan_sha256']:
                raise ValueError('Historical random plan changed')
        return item

    def state(self):
        verify(self.design['checkpoint'])
        state=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        validate_state(state,self)
        return state


def validate_state(state, package):
    execution=state['execution']
    if (state['correction_arm']!='R_static_reference_delta' or state['package_sha256']!=OLD_PACKAGE_SHA
        or state['initialization_sha256']!=INITIAL_SHA or execution['final_update_200'] is not True
        or execution['updates_completed']!=200 or execution['updates_attempted']!=200
        or execution['exposures']!=600 or execution['mdm_examples']!=600 or execution['encoder_examples']!=1200):
        raise ValueError('Only fixed R final200 with original provenance is permitted')
    expected_config=asdict(MDMIMUControlConfig(**dict(package.old.common['control_config'],reference_mode='static_delta')))
    if state['control_config']!=expected_config:
        raise ValueError('R control configuration changed')
    if state['initial_tensor_hashes']!=package.old_manifest['initialization']['tensor_hashes']:
        raise ValueError('Original initial tensor identities changed')
    for key in ('imu_normalization','training_input_provenance','environment'):
        if state[key]!=package.common[key]:raise ValueError('Checkpoint '+key+' changed')
    if environment()!=package.common['environment']:raise ValueError('Frozen environment changed')


class AlignmentGuard(RunGuard):
    def __init__(self,context,**kwargs):
        phase=context.get('phase')
        if context.get('version')!='itm-alignment-context-v1' or phase not in ('Q','sample'):
            raise ValueError('Alignment context required; no training phase')
        super().__init__(dict(context,phase='sample'),**kwargs)
        self.context=context;self.phase=phase
        self.caps={'mdm_examples':480 if phase=='Q' else 4000,'encoder_examples':576 if phase=='Q' else 2}
        if context['counters']!=self.caps:raise ValueError('Frozen counters required')
        if context['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Wrong R identity')
        self.state.update(version='itm-alignment-receipt-v1',phase=phase,context=context,
            mdm_examples=0,encoder_examples=0,mdm_calls=0,completed_slots=[],completed_traces=[],
            checkpoint_sha256=CHECKPOINT_SHA,checks=[])
        self.persist()

    def before_update(self):raise RunLimitExceeded('Alignment forbids optimizer updates')

    def reserve(self,kind,count=1):
        self.check()
        if kind not in self.caps or count<1 or self.state[kind]+count>self.caps[kind]:
            raise RunLimitExceeded('Alignment '+kind+' limit')
        self.state[kind]+=count;self.persist()

    def before_q(self,ids):
        if self.phase!='Q' or len(ids)!=5:raise ValueError('Q must be one batch5')
        self.state.update(active_probe_ids=list(ids),active_probe_forward_started=False)
        self.reserve('mdm_examples',5)
        self.state.update(mdm_calls=self.state['mdm_calls']+1,active_probe_forward_started=True);self.persist()

    def before_denoise(self,case_id,step,branch_examples):
        if self.phase!='sample' or branch_examples!=4 or step!=999-self.state['denoising_calls']:
            raise RunLimitExceeded('Frozen four-branch/1000-step order required')
        self.reserve('mdm_examples',4)
        super().before_denoise(case_id,step,branch_examples)
        self.state['mdm_calls']+=1;self.persist()

    def finish(self):
        self.check()
        if self.state['updates_attempted'] or self.state['updates_completed']:raise RunLimitExceeded('Unexpected update')
        expected=480 if self.phase=='Q' else 4000
        calls=96 if self.phase=='Q' else 1000
        if self.state['mdm_examples']!=expected or self.state['mdm_calls']!=calls:
            raise RunLimitExceeded('Incomplete alignment forward count')
        if self.state['completed_slots']!=self.context['expected_slots']:
            raise RunLimitExceeded('Incomplete slot identities')
        if self.state['completed_traces']!=self.context.get('expected_traces',[]):
            raise RunLimitExceeded('Incomplete traces')
        if self.phase=='sample' and (self.state['denoising_calls']!=1000 or self.state['branch_examples']!=4000):
            raise RunLimitExceeded('Incomplete sampling')
        self.state['status']='completed';self.persist()


def save_npz(path,arrays,guard):
    for value in arrays.values():
        value=np.asarray(value)
        if np.issubdtype(value.dtype,np.number):numeric(value)
    atomic_artifact(path,lambda f:np.savez_compressed(f,**arrays),guard)


def save_json(path,value,guard):
    atomic_artifact(path,lambda f:f.write(json.dumps(value,allow_nan=False).encode()),guard)


class TraceGuidance(torch.nn.Module):
    """Observe the actual MDM batch4 and actual guided return; never replace either."""
    def __init__(self,guided,raw_model,slots,output,metadata,guard,scaled_t):
        super().__init__();self.guided=guided
        self.njoints=guided.njoints;self.nfeats=guided.nfeats
        self.slots={r['timestep']:r for r in slots};self.output=Path(output)
        self.metadata=metadata;self.guard=guard;self.scaled_t=scaled_t
        self.pending=None;self.captured=None;self.seen=[]
        self.handle=raw_model.register_forward_hook(self.capture)

    def capture(self,module,args,result):
        if self.pending is not None:
            if self.captured is not None or result.shape[0]!=4:raise ValueError('Trace requires exactly one batch4')
            self.captured=tuple(v.detach().clone() for v in result.chunk(4))

    def close(self):self.handle.remove()

    def forward(self,x,t,y=None):
        step=int(t[0]);slot=self.slots.get(step)
        if slot is None:return self.guided(x,t,y=y)
        if step in self.seen or list(x.shape)!=slot['shape']:raise ValueError('Trace slot/order/shape mismatch')
        if step!=list(self.slots)[len(self.seen)]:raise ValueError('Trace visitation order changed')
        self.guard.state['active_trace_ids']=[slot['trace_id']];self.guard.persist()
        self.pending=slot;self.captured=None;snapshot=x.detach().clone()
        try:
            result=self.guided(x,t,y=y)
            if self.captured is None:raise ValueError('Trace missed existing model forward')
            if result.dtype!=torch.float32 or any(v.dtype!=torch.float32 for v in self.captured):
                raise ValueError('Frozen trace requires float32 predictions')
            arrays=dict(zip(('f00','f10','f01','f11'),map(numpy,self.captured)))
            arrays.update(x_t=numpy(snapshot),actual_combined_prediction=numpy(result),frame_mask=numpy(y['imu_frame_mask']))
            # Preserve finite arrays even when the following software check fails.
            path=self.output/(slot['trace_id']+'.npz');save_npz(path,arrays,self.guard)
            checks=dict(combination=combination_check(self.captured,result,enforce=False))
            pairs=[]
            if not self.metadata['text_present']:pairs.extend((('f00','f10'),('f01','f11')))
            if not self.metadata['imu_present'] or self.metadata['condition'] in ('backbone_bypass','physical_static'):
                pairs.extend((('f00','f01'),('f10','f11')))
            for a,b in pairs:checks[a+'='+b]=difference(arrays[a],arrays[b],enforce=False)
            returned=array_sha(result)
            if returned!=array_sha(arrays['actual_combined_prediction']):raise ValueError('Trace return hash changed')
            branches=[arrays[k].astype(np.float64) for k in ('f00','f10','f01','f11')]
            delta=1.5*(branches[1]-branches[0])
            record=dict(slot,status='completed')
            record.update(**self.metadata,actual_return_sha256=returned,checks=checks,
                branch_flags=[dict(branch=b,text_present=text and self.metadata['text_present'],
                    imu_present=imu and self.metadata['imu_present'])
                    for b,text,imu in (('f00',False,False),('f10',True,False),('f01',False,True),('f11',True,True))],
                backbone_bypass=self.metadata['condition']=='backbone_bypass',
                scaled_t=numpy(self.scaled_t(t)).tolist(),arrays={k:describe(v) for k,v in arrays.items()},
                reductions={k:reductions(v) for k,v in arrays.items()},text_increment=reductions(delta),
                offline_legacy_minus_unit=reductions(delta),
                interpretation='Same S1 state; float64 conceptual recombination, not historical trajectory')
            passed=all(r['passed'] for r in checks['combination'].values()) and all(
                r['passed'] for k,r in checks.items() if k!='combination')
            if not passed:record['status']='failed'
            save_json(path.with_suffix('.json'),record,self.guard)
            if not passed:raise ValueError('Frozen trace arithmetic/absent-branch equality failed')
            self.guard.state['completed_traces'].append(slot['trace_id']);self.guard.state['active_trace_ids']=[]
            self.guard.persist();self.seen.append(step)
            return result
        finally:
            self.pending=None;self.captured=None
