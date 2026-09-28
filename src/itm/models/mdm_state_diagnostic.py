"""Frozen V saved-state inputs, counters and failure ledger; no run authorization."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from itm.models.mdm_correction import ROOT,artifact,read,verify,transform_inputs
from itm.models.mdm_sampling_alignment import (AlignmentPackage,CHECKPOINT_SHA,input_arrays,describe,digest,
    difference,numeric,save_npz,save_json)
from itm.models.mdm_run_limits import RunGuard,RunLimitExceeded,sha256

PACKAGE=ROOT/'docs/itm_rework/protocol/b0_state_diagnostic_20260928_v1/package.json'
PACKAGE_SHA='6f050dbfbe57c508e86d83a30b3cdfb505f7700b6781cb52b43d85b6ffc0b012'
BRANCHES=('T','P','S','R')
CAPS=dict(mdm=192,encoder=72,caption=12)
CALL_CAPS=dict(mdm=48,encoder=72,caption=12)


def ledger():
    groups=[dict(group_id=f'V{i:03d}',status='planned',executed=False) for i in range(1,49)]
    branches=[dict(slot_id=g['group_id']+'-'+b,group_id=g['group_id'],status='planned',executed=False) for g in groups for b in BRANCHES]
    checks=[dict(check_id=g['group_id']+'-'+c,group_id=g['group_id'],status='planned') for g in groups
            for c in ('P_vs_trace_f11','T_vs_trace_f10','S_vs_T')]
    return dict(groups=groups,branches=branches,checks=checks)


def stop_rows(state):
    for key in ('groups','branches','checks'):
        for row in state[key]:
            if row['status']=='planned':row['status']='not_run_after_stop'
            elif row['status']=='running':
                row['status']='failed'
                if key!='checks':row['diagnostic_only']=bool(row.get('raw_artifact'))


class StatePackage:
    def __init__(self,path=PACKAGE):
        self.path=Path(path).resolve()
        if sha256(self.path)!=PACKAGE_SHA:raise ValueError('Exact frozen V package required')
        self.files={Path(a['path']).stem:verify(a) for a in read(self.path)['artifacts']}
        self.source_paths=[verify(a) for a in read(self.files['sources'])]
        self.design=read(self.files['design']);self.alignment=AlignmentPackage(verify(self.design['old_package']))
        self.common=self.alignment.common
        if self.design['common']!=self.common:raise ValueError('V common identity changed')
        self.checkpoint=verify(self.design['checkpoint'])
        if sha256(self.checkpoint)!=CHECKPOINT_SHA:raise ValueError('V fixed R identity mismatch')
        self.mean,self.std=self.alignment.mean,self.alignment.std
        self.groups=read(self.files['state_groups']);self.slots=read(self.files['branch_slots'])
        self.checks=read(self.files['software_checks']);self.cases=read(self.files['input_cases'])
        canonical=ledger()
        for key,rows,idkey in (('groups',self.groups,'group_id'),('branches',self.slots,'slot_id'),('checks',self.checks,'check_id')):
            if [v[idkey] for v in rows]!=[v[idkey] for v in canonical[key]]:raise ValueError('Frozen V population changed')
        if len(self.cases)!=12:raise ValueError('Twelve input caches required')

    def state(self):return self.alignment.state()

    def inputs(self,case):
        with np.load(verify(case['original_sample']),allow_pickle=False) as z:
            original={k:z[k] for k in z.files if k!='metadata'}
        items=[];rows=[];descriptors=[]
        for branch in case['branches']:
            row=dict(case['sample_spec'],condition=branch['condition'],text_present=True,imu_present=branch['imu_present'])
            item=self.alignment.old.inputs(row)
            if item['text']!=case['caption']:raise ValueError('V caption identity changed')
            actual=input_arrays(item,row)
            imu,reference,a,o=transform_inputs(original['acceleration'][0],original['orientation'][0],self.common['imu_normalization'],row['condition'])
            expected={k:original[k].copy() for k in branch['arrays']}
            expected.update(acceleration=a[None],orientation=o[None],normalized_imu=imu[None],normalized_reference_imu=reference[None],
                            text_condition_present=np.array([True]),imu_condition_present=np.array([branch['imu_present']]))
            for name,spec in branch['arrays'].items():
                if describe(expected[name])!=spec:raise ValueError('Frozen reference input changed: '+name)
                if name in ('orientation','normalized_imu','normalized_reference_imu'):
                    if actual[name].dtype!=expected[name].dtype:raise ValueError('V input dtype mismatch')
                    difference(actual[name],expected[name],atol=2e-7,rtol=0)
                elif describe(actual[name])!=spec:raise ValueError('Exact V input changed: '+name)
            items.append(item);rows.append(row);descriptors.append({k:describe(v) for k,v in actual.items()})
        return items,rows,descriptors

    def saved(self,group):
        with np.load(verify(group['saved_state']),allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in ('x_t','f10','f11','frame_mask')}
        with np.load(verify(group['target_artifact']),allow_pickle=False) as z:arrays['target']=z[group['target_array']].copy()
        for name,spec in (('x_t',group['state_descriptor']),('target',group['target_descriptor']),('frame_mask',group['frame_mask']),
                          ('f10',group['historical_T']),('f11',group['historical_P'])):
            if describe(arrays[name])!={k:v for k,v in spec.items() if k!='array'}:raise ValueError('Saved V '+name+' identity mismatch')
            numeric(arrays[name])
        meta=read(verify(group['saved_state_metadata']))
        if meta['checkpoint_sha256']!=CHECKPOINT_SHA or meta['scaled_t']!=group['scaled_t'] or meta['trace_id']!=group['trace_id']:
            raise ValueError('Saved trace lineage mismatch')
        return arrays


class StateGuard(RunGuard):
    def __init__(self,context,**kwargs):
        if context.get('version')!='itm-state-context-v1' or context.get('phase')!='V':raise ValueError('V context required')
        if context.get('package_sha256')!=PACKAGE_SHA or context.get('checkpoint_sha256')!=CHECKPOINT_SHA:raise ValueError('V context identity')
        super().__init__(dict(context,phase='sample'),**kwargs)
        self.context=context;self.phase='V'
        self.state.update(version='itm-state-receipt-v1',phase='V',context=context,**ledger(),
            counts={k:dict(reserved=0,actual=0,completed=0,reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
        self.persist()

    def before_update(self):raise RunLimitExceeded('V forbids training')
    def before_denoise(self,*args):raise RunLimitExceeded('V forbids denoising')

    def invoke(self,kind,n,function):
        self.check()
        if kind not in CAPS or n!=(4 if kind=='mdm' else 1):raise RunLimitExceeded('Wrong V call kind/batch')
        count=self.state['counts'][kind]
        if count['reserved']+n>CAPS[kind] or count['reserved_calls']+1>CALL_CAPS[kind]:raise RunLimitExceeded('V call cap exceeded')
        count['reserved']+=n;count['reserved_calls']+=1;self.persist();self.check()
        count['actual']+=n;count['actual_calls']+=1
        if kind=='mdm':
            gid=self.state['active_group']
            for key in ('groups','branches'):
                for row in self.state[key]:
                    if row['group_id']==gid:row['executed']=True
        self.persist()
        value=function();self.ensure_finite(value,kind);self.check()
        count['completed']+=n;count['completed_calls']+=1;self.persist()
        return value

    def finish(self):
        self.check()
        for kind,cap in CAPS.items():
            c=self.state['counts'][kind]
            if any(c[k]!=cap for k in ('reserved','actual','completed')) or any(c[k]!=CALL_CAPS[kind] for k in ('reserved_calls','actual_calls','completed_calls')):
                raise RunLimitExceeded('Incomplete V count: '+kind)
        if any(r['status']!='completed' for k in ('groups','branches','checks') for r in self.state[k]):raise RunLimitExceeded('Incomplete V ledger')
        if self.state['updates_attempted'] or self.state['updates_completed'] or self.state['denoising_calls']:raise RunLimitExceeded('Forbidden V work')
        self.state['status']='completed';self.persist()

    def fail(self,error):
        stop_rows(self.state);super().fail(error)


class CountedEncoder(torch.nn.Module):
    def __init__(self,encoder,guard):
        super().__init__();self.encoder=encoder;self.guard=guard;self.config=encoder.config;self.eval()
    def forward(self,*args,**kwargs):
        return self.guard.invoke('encoder',len(args[0]),lambda:self.encoder(*args,**kwargs))


class StateCache:
    def __init__(self):self.entries={}
    def get(self,case,items,rows,descriptors,package,model,encoder,guard,device,batch_inputs):
        if any(m.training for m in model.modules()) or encoder.training:raise ValueError('V caches require eval')
        if [{k:describe(v) for k,v in input_arrays(item,row).items()} for item,row in zip(items,rows)]!=descriptors:
            raise ValueError('V cache actual inputs differ from descriptors')
        identity=digest(dict(checkpoint=CHECKPOINT_SHA,case=case,inputs=descriptors,
                             normalization=package.common['imu_normalization'],dtype='float32',eval=True))
        if case['cache_id'] in self.entries:
            old=self.entries[case['cache_id']]
            if old['identity']!=identity:raise ValueError('V cache identity changed')
            if describe(old['text'])!=old['text_descriptor'] or describe(old['control'])!=old['control_descriptor']:
                raise ValueError('V cached tensor changed')
            return old,False
        from itm.models.mdm_correction import encode_control
        text=guard.invoke('caption',1,lambda:model.encode_text([case['caption']]))
        if text.dtype!=torch.float32 or text.shape[1]!=1:raise ValueError('V caption must be batch1 float32')
        controls=[torch.zeros((1,case['frames'],encoder.config.latent_dim),device=device,dtype=torch.float32)]
        counted=CountedEncoder(encoder,guard)
        for item,row in zip(items[1:],rows[1:]):
            _,y=batch_inputs([item],[row],device)
            control,present=encode_control(counted,y,mode='static_delta')
            if not present.all() or control.dtype!=torch.float32:raise ValueError('V encoded condition mismatch')
            guard.ensure_finite(control,'V delta control');controls.append(control)
        entry=dict(identity=identity,text=text,control=torch.cat(controls))
        entry.update(text_descriptor=describe(text),control_descriptor=describe(entry['control']))
        self.entries[case['cache_id']]=entry
        return entry,True
