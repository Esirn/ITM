"""Frozen text-prefix history-by-suffix completion with durable limits."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from itm.models.mdm_correction import ROOT,read,verify,artifact,environment
from itm.models.mdm_native_training_sufficiency import TrainingPackage
from itm.models.mdm_run_limits import RunGuard,RunLimitExceeded,sha256
from itm.models.mdm_training_sufficiency import compact_json

PACKAGE=ROOT/'docs/itm_rework/protocol/b0_suffix_history_factorial_20261008_v2/package.json'
PACKAGE_SHA='abd7894aaf4fdc90f7fa35d595b51152fb8a9533c64accbed0c0e5928034d40c'
BRANCHES=('f00','f10','f01','f11')
CAPS=dict(mdm=14400,encoder=4,caption=2,noise=3596,decode=4)
CALL_CAPS=dict(mdm=3600,encoder=4,caption=2,noise=3596,decode=4)
TABLES=('trajectories','steps','traces','finals','checks')


class SuffixPackage:
    def __init__(self,path=PACKAGE):
        self.path=Path(path).resolve()
        if sha256(self.path)!=PACKAGE_SHA:raise ValueError('Exact suffix package required')
        self.files={Path(a['path']).stem:verify(a) for a in read(self.path)['artifacts']}
        self.source_paths=[verify(a) for a in read(self.files['sources'])]
        self.design=read(self.files['design']);self.budgets=read(self.files['budgets'])
        self.trajectories=read(self.files['trajectories']);self.anchors=read(self.files['historical_anchors'])
        self.starts={r['panel_case_id']:r for r in read(self.files['starts'])}
        self.noise={r['noise_id']:r for r in read(self.files['noise'])}
        self.native=TrainingPackage();self.common=self.native.common;self.mean,self.std=self.native.mean,self.native.std
        self.snapshots={r['snapshot_id']:r for r in self.native.snapshots if r['snapshot_id']=='F200'}
        self.slots={r['slot_id']:r for r in read(ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/sampling_slots.json')}
        self.cases={r['panel_case_id']:self.slots[r['input_slot_id']] for r in self.trajectories}
        if len(self.trajectories)!=4 or len(self.starts)!=2 or len(self.anchors)!=8:raise ValueError('Suffix denominator changed')
        if any(r['diffusion_steps']!=list(range(899,-1,-1)) for r in self.trajectories):raise ValueError('Suffix schedule changed')
        for cid in self.cases:
            pair=[r for r in self.trajectories if r['panel_case_id']==cid]
            if len(pair)!=2 or pair[0]['start']!=pair[1]['start'] or {(r['condition'],r['imu_present']) for r in pair}!={('keep_text_only',False),('switch_imu_on',True)}:raise ValueError('Frozen history/suffix roles')
        for anchor in self.anchors:
            if read(verify(anchor['metadata']))['condition']!='text_only':raise ValueError('Original text-only baseline anchor required')
    def state(self,sid):return self.native.state(self.snapshots[sid])
    def inputs(self,trajectory):
        row=self.slots[trajectory['input_slot_id']]
        return self.native.alignment.inputs(row),row


def step_id(tid,t):return tid+'-'+str(t)
def ledger(package):
    def row(key,**values):return dict(id=key,status='planned',**values)
    paths=package.trajectories
    steps=[row(step_id(r['trajectory_id'],t),trajectory_id=r['trajectory_id'],timestep=t) for r in paths for t in r['diffusion_steps']]
    traces=[row(step_id(r['trajectory_id'],t),trajectory_id=r['trajectory_id'],timestep=t) for r in paths for t in r['trace_timesteps']]
    checks=[row('START-'+c,kind='shared_start') for c in sorted(package.cases)]
    for r in paths:
        tid=r['trajectory_id']
        for t in r['diffusion_steps']:
            key=step_id(tid,t);checks.append(row('CFG-'+key,kind='CFG'))
            if not r['imu_present']:
                for a,b in [('f00','f01'),('f10','f11')]:checks.append(row('ABS-'+key+'-'+a+'='+b,kind='absent'))
    for anchor in package.anchors:
        for name in anchor['arrays']:
            suffix='FINAL-'+anchor['trajectory_id'] if anchor.get('kind')=='final' else 'ANCH-'+step_id(anchor['trajectory_id'],anchor['timestep'])
            checks.append(row(suffix+'-'+name,kind='historical_final' if anchor.get('kind')=='final' else 'historical_trace'))
    return dict(trajectories=[row(r['trajectory_id']) for r in paths],steps=steps,traces=traces,
        finals=[row(r['trajectory_id']) for r in paths],checks=checks)


def stop_rows(state):
    for key in TABLES:
        for r in state[key]:
            if r['status']=='planned':r['status']='not_run_after_stop'
            elif r['status']=='running':r['status']='failed'
            if r.get('raw_saved'):r['diagnostic_only']=True


class SuffixGuard(RunGuard):
    def __init__(self,context,package,**kwargs):
        if context.get('version')!='itm-history-suffix-context-v1' or context.get('package_sha256')!=PACKAGE_SHA or context.get('phase')!='history_suffix':
            raise ValueError('Suffix bounded context identity')
        super().__init__(dict(context,phase='sample',max_denoising_calls=3600,max_branch_examples=14400),**kwargs)
        self.context=context;self.phase='history_suffix';self.package=package
        self.state.update(version='itm-history-suffix-receipt-v1',phase='history_suffix',context={k:v for k,v in context.items() if k!='expected_artifacts'},
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
        if kind not in CAPS or n!=(4 if kind=='mdm' else 1):raise RunLimitExceeded('Suffix call kind/batch')
        c=self.state['counts'][kind]
        if c['reserved']+n>CAPS[kind] or c['reserved_calls']+1>CALL_CAPS[kind]:raise RunLimitExceeded('Suffix call budget')
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
        if not report['passed']:raise ValueError('Frozen suffix equation failed: '+key)
    def finish(self):
        self.check()
        for k,n in CAPS.items():
            c=self.state['counts'][k]
            if any(c[f]!=n for f in ('reserved','actual','completed')) or any(c[f]!=CALL_CAPS[k] for f in ('reserved_calls','actual_calls','completed_calls')):
                raise RunLimitExceeded('Incomplete suffix counters: '+k)
        if self.state['denoising_calls']!=3600 or self.state['branch_examples']!=14400 or self.state['updates_attempted'] or self.state['updates_completed']:
            raise RunLimitExceeded('Suffix work identity')
        if any(r['status']!='completed' for key in TABLES for r in self.state[key]):raise RunLimitExceeded('Incomplete suffix ledger')
        self.state['status']='completed';self.persist()
    def fail(self,error):
        stop_rows(self.state)
        for key in TABLES:self.dirty[key].update(self.maps[key])
        super().fail(error)
    def before_update(self):raise RunLimitExceeded('Suffix forbids training')
    def optimizer_step(self,*a,**k):raise RunLimitExceeded('Suffix forbids optimizer')


def read_receipt(path):
    path=Path(path);state=read(path);journal=path.with_suffix('.journal.jsonl')
    if not journal.exists():return state
    maps={k:{r['id']:r for r in state[k]} for k in TABLES};previous=0;applied=state.get('journal_applied_sequence',0)
    for line in journal.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):state['truncated_journal_tail']=True;break
        e=json.loads(line)
        if e['sequence']!=previous+1:raise ValueError('Suffix journal sequence gap')
        previous=e['sequence']
        if previous<=applied:continue
        state.update(e['fields'],counts=e['counts'])
        for key in TABLES:
            for r in e['rows'][key]:maps[key][r['id']].update(r)
        state['artifacts'].extend(e['artifacts']);state['journal_applied_sequence']=previous
    if applied>previous:raise ValueError('Missing suffix journal events')
    return state


def replay_suffix(model,diffusion,y,start,draw,*,before=lambda t,x:None,after=lambda t,x,p:None):
    if (diffusion.num_timesteps!=1000 or getattr(diffusion,'rescale_timesteps',False)
        or list(getattr(diffusion,'timestep_map',range(1000)))!=list(range(1000))):raise ValueError('Full native1000 schedule required')
    if start.dtype!=torch.float32 or start.ndim!=4 or start.shape[:3]!=(1,263,1):raise ValueError('Native full-frame x899 required')
    x=start.clone()
    for step in range(899,-1,-1):
        before(step,x)
        t=torch.full((1,),step,device=x.device,dtype=torch.long)
        posterior=diffusion.p_mean_variance(model,x,t,clip_denoised=False,model_kwargs={'y':y})
        x=posterior['mean']
        if step:x=x+torch.exp(.5*posterior['log_variance'])*draw(step)
        if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite suffix posterior')
        after(step,x,posterior)
    return x


def required_artifacts(package,out):
    out=Path(out);paths=set()
    for cid in package.cases:
        paths.update(out/'inputs'/(cid+ext) for ext in ('.npz','.json'))
        paths.add(out/'cache'/('text-'+cid+'.npz'))
        paths.update(out/'cache'/('F200-'+cid+ext) for ext in ('.npz','.json'))
    for r in package.trajectories:
        tid=r['trajectory_id'];paths.update([out/'starts'/(tid+'.npz'),out/'trajectories'/(tid+'.json'),out/'raw_finals'/(tid+'.npz')])
        paths.update(out/'finals'/(tid+ext) for ext in ('.npz','.json'))
        for t in r['diffusion_steps']:paths.update(out/'steps'/(step_id(tid,t)+ext) for ext in ('.npz','.json'))
        paths.update(out/'traces'/(step_id(tid,t)+'.npz') for t in r['trace_timesteps'])
    return {str(p.resolve()) for p in paths}
