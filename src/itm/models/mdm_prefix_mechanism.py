"""Immutable 100-step mechanism protocol, native posterior replay, durable limits."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from itm.models.mdm_correction import ROOT,read,verify,artifact,environment
from itm.models.mdm_native_training_sufficiency import TrainingPackage
from itm.models.mdm_run_limits import RunGuard,RunLimitExceeded,sha256
from itm.models.mdm_training_sufficiency import compact_json

PACKAGE=ROOT/'docs/itm_rework/protocol/b0_prefix_mechanism_20261008_v1/package.json'
PACKAGE_SHA='8b32b863da7a530cc71350384663678adfd072f8db53bbb01add176b3040853c'
BRANCHES=('f00','f10','f01','f11')
CAPS=dict(mdm=15840,encoder=48,caption=12,noise=3636)
CALL_CAPS=dict(mdm=3960,encoder=48,caption=12,noise=3636)
TABLES=('trajectories','steps','states','probes','branches','checks')


class PrefixPackage:
    def __init__(self,path=PACKAGE):
        self.path=Path(path).resolve()
        if sha256(self.path)!=PACKAGE_SHA:raise ValueError('Exact prefix mechanism package required')
        self.files={Path(a['path']).stem:verify(a) for a in read(self.path)['artifacts']}
        self.source_paths=[verify(a) for a in read(self.files['sources'])]
        self.design=read(self.files['design']);self.budgets=read(self.files['budgets'])
        self.trajectories=read(self.files['trajectories']);self.states=read(self.files['states'])
        self.probes=read(self.files['probes']);self.anchors=read(self.files['historical_anchors'])
        self.noise={r['noise_id']:r for r in read(self.files['noise'])}
        self.native=TrainingPackage();self.common=self.native.common;self.mean,self.std=self.native.mean,self.native.std
        self.snapshots={r['snapshot_id']:r for r in self.native.snapshots if r['snapshot_id'] in ('F200','U200')}
        self.slots={r['slot_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
        self.cases={r['panel_case_id']:self.slots[r['input_slot_id']] for r in self.trajectories if r['path_arm']=='F'}
        if (len(self.trajectories),len(self.states),len(self.probes),len(self.anchors))!=(36,180,360,48):
            raise ValueError('Prefix population changed')
    def state(self,sid):return self.native.state(self.snapshots[sid])
    def inputs(self,trajectory):
        row=self.slots[trajectory['input_slot_id']]
        return self.native.alignment.inputs(row),row


def step_id(tid,t):return tid+'-'+str(t)
def ledger(package):
    def row(key,**values):return dict(id=key,status='planned',**values)
    trajectories=[row(r['trajectory_id']) for r in package.trajectories]
    steps=[row(step_id(r['trajectory_id'],t),trajectory_id=r['trajectory_id'],timestep=t)
           for r in package.trajectories for t in r['diffusion_steps']]
    states=[row(r['state_id'],trajectory_id=r['trajectory_id'],timestep=r['timestep']) for r in package.states]
    probes=[row(r['probe_id'],state_id=r['state_id'],checkpoint=r['checkpoint']) for r in package.probes]
    branches=[row(r['probe_id']+'-'+b,probe_id=r['probe_id'],branch=b) for r in package.probes for b in BRANCHES]
    checks=[]
    for cid in sorted({r['panel_case_id'] for r in package.trajectories}):checks.append(row('INIT-'+cid,kind='initial_identity'))
    for a in package.anchors:
        for b in a['arrays']:checks.append(row('ANCH-'+a['trajectory_id']+'-'+a['historical_arm']+'-'+b,kind='historical'))
    paths={r['trajectory_id']:r['path_arm'] for r in package.trajectories}
    states_by_id={r['state_id']:r for r in package.states}
    for p in package.probes:
        s=states_by_id[p['state_id']];arm=paths[s['trajectory_id']]
        if arm=='T' or p['checkpoint']==('F200' if arm=='F' else 'U200'):
            for b in (('f00','f10') if arm=='T' else BRANCHES):checks.append(row('OWN-'+p['probe_id']+'-'+b,kind='own_prediction'))
    for s in package.states:
        for b in ('f00','f10'):checks.append(row('CROSS-'+s['state_id']+'-'+b,kind='inactive_adapter'))
    for s in steps:checks.append(row('CFG-'+s['id'],kind='CFG'))
    for r in package.trajectories:
        if r['path_arm']=='T':
            for t in r['diffusion_steps']:
                for b in ('f00=f01','f10=f11'):checks.append(row('ABS-'+step_id(r['trajectory_id'],t)+'-'+b,kind='absent'))
    return dict(trajectories=trajectories,steps=steps,states=states,probes=probes,branches=branches,checks=checks)


def stop_rows(state):
    for key in TABLES:
        for r in state[key]:
            if r['status']=='planned':r['status']='not_run_after_stop'
            elif r['status']=='running':r['status']='failed'
            if r.get('raw_saved'):r['diagnostic_only']=True


class PrefixGuard(RunGuard):
    def __init__(self,context,package,**kwargs):
        if context.get('version')!='itm-prefix-context-v1' or context.get('package_sha256')!=PACKAGE_SHA or context.get('phase')!='prefix':
            raise ValueError('Prefix bounded context identity')
        super().__init__(dict(context,phase='sample',max_denoising_calls=3600,max_branch_examples=15840),**kwargs)
        self.context=context;self.phase='prefix';self.package=package
        self.state.update(version='itm-prefix-receipt-v1',phase='prefix',context={k:v for k,v in context.items() if k!='expected_artifacts'},
            **ledger(package),counts={k:dict(reserved=0,actual=0,completed=0,reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS},
            journal_applied_sequence=0)
        self.maps={key:{r['id']:r for r in self.state[key]} for key in TABLES}
        self.dirty={key:set() for key in TABLES};self.sequence=0;self.artifact_count=0
        self.journal=self.path.with_suffix('.journal.jsonl')
        if self.journal.exists():raise FileExistsError(self.journal)
        compact_json(self.path,self.state);self.ready=True
    def update(self,table,key,**fields):
        self.maps[table][key].update(fields);self.dirty[table].add(key)
        return self.maps[table][key]
    def persist(self):
        self.state['last_monotonic']=self.clock()
        if not getattr(self,'ready',False):compact_json(self.path,self.state);return
        self.sequence+=1
        event=dict(sequence=self.sequence,counts=self.state['counts'],
            fields={k:v for k,v in self.state.items() if k not in (*TABLES,'artifacts','counts','context')},
            rows={key:[self.maps[key][i] for i in sorted(self.dirty[key])] for key in TABLES},
            artifacts=self.state['artifacts'][self.artifact_count:])
        data=(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n').encode()
        with self.journal.open('ab',buffering=0) as f:
            rest=memoryview(data)
            while rest:
                n=f.write(rest)
                if not n:raise OSError('Journal append made no progress')
                rest=rest[n:]
            os.fsync(f.fileno())
        self.artifact_count=len(self.state['artifacts'])
        for value in self.dirty.values():value.clear()
        if self.state['status'] in ('completed','failed'):
            self.state['journal_applied_sequence']=self.sequence;compact_json(self.path,self.state)
    def invoke(self,kind,n,fn):
        self.check()
        if kind not in CAPS or n!=(4 if kind=='mdm' else 1):raise RunLimitExceeded('Prefix call kind/batch')
        c=self.state['counts'][kind]
        if c['reserved']+n>CAPS[kind] or c['reserved_calls']+1>CALL_CAPS[kind]:raise RunLimitExceeded('Prefix call budget')
        c['reserved']+=n;c['reserved_calls']+=1;self.persist();self.check()
        c['actual']+=n;c['actual_calls']+=1
        if kind=='mdm':
            active=self.state['active'];self.update(active['table'],active['id'],executed=True)
            if active['table']=='steps':self.state['denoising_calls']+=1
            self.state['branch_examples']+=n
        self.persist()
        result=fn();self.ensure_finite(result,kind)
        if kind=='noise' and not np.isfinite(result[0]).all():raise FloatingPointError('Nonfinite addressed noise')
        self.check();c['completed']+=n;c['completed_calls']+=1;self.persist();return result
    def equation(self,key,report):
        self.update('checks',key,status='completed' if report['passed'] else 'failed',result=report);self.persist()
        if not report['passed']:raise ValueError('Frozen prefix equation failed: '+key)
    def finish(self):
        self.check()
        for k,n in CAPS.items():
            c=self.state['counts'][k]
            if any(c[f]!=n for f in ('reserved','actual','completed')) or any(c[f]!=CALL_CAPS[k] for f in ('reserved_calls','actual_calls','completed_calls')):
                raise RunLimitExceeded('Incomplete prefix counters: '+k)
        if self.state['denoising_calls']!=3600 or self.state['branch_examples']!=15840 or self.state['updates_attempted'] or self.state['updates_completed']:
            raise RunLimitExceeded('Prefix work identity')
        if any(r['status']!='completed' for key in TABLES for r in self.state[key]):raise RunLimitExceeded('Incomplete prefix ledger')
        self.state['status']='completed';self.persist()
    def fail(self,error):
        stop_rows(self.state)
        for key in TABLES:self.dirty[key].update(self.maps[key])
        super().fail(error)
    def before_update(self):raise RunLimitExceeded('Prefix forbids training')
    def optimizer_step(self,*a,**k):raise RunLimitExceeded('Prefix forbids optimizer')


def read_receipt(path):
    path=Path(path);state=read(path);journal=path.with_suffix('.journal.jsonl')
    if not journal.exists():return state
    maps={k:{r['id']:r for r in state[k]} for k in TABLES};previous=0;applied=state.get('journal_applied_sequence',0)
    for line in journal.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):state['truncated_journal_tail']=True;break
        e=json.loads(line)
        if e['sequence']!=previous+1:raise ValueError('Prefix journal sequence gap')
        previous=e['sequence']
        if previous<=applied:continue
        state.update(e['fields'],counts=e['counts'])
        for key in TABLES:
            for r in e['rows'][key]:maps[key][r['id']].update(r)
        state['artifacts'].extend(e['artifacts']);state['journal_applied_sequence']=previous
    if applied>previous:raise ValueError('Missing prefix journal events')
    return state


def replay_prefix(model,diffusion,y,draw,*,before=lambda t,x:None,after=lambda t,x,p:None):
    """Exact native sampler operations; only truncate the existing schedule."""
    if diffusion.num_timesteps!=1000 or getattr(diffusion,'rescale_timesteps',False):raise ValueError('Full native1000 schedule required')
    x=draw(-1)
    for step in range(999,899,-1):
        before(step,x)
        t=torch.full((1,),step,device=x.device,dtype=torch.long)
        posterior=diffusion.p_mean_variance(model,x,t,clip_denoised=False,model_kwargs={'y':y})
        x=posterior['mean']
        x=x+torch.exp(.5*posterior['log_variance'])*draw(step)
        if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite prefix posterior transition')
        after(step,x,posterior)
    return x


def required_artifacts(package,out):
    out=Path(out);paths=set()
    for c in package.cases:paths.update([out/'inputs'/(c+'.npz'),out/'inputs'/(c+'.json'),out/'cache'/('text-'+c+'.npz')])
    for c in package.cases:
        for w in ('F200','U200'):
            for ext in ('.npz','.json'):paths.add(out/'cache'/(w+'-'+c+ext))
    for r in package.trajectories:
        tid=r['trajectory_id'];paths.update([out/'initial'/(tid+'.npz'),out/'terminal'/(tid+'.npz'),out/'trajectories'/(tid+'.json')])
        for t in r['diffusion_steps']:
            for ext in ('.npz','.json'):paths.add(out/'steps'/(step_id(tid,t)+ext))
    paths.update(out/'states'/(s['state_id']+'.npz') for s in package.states)
    for p in package.probes:
        for ext in ('.npz','.json'):paths.add(out/'probes'/(p['probe_id']+ext))
    return {str(p.resolve()) for p in paths}
