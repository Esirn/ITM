"""Six-snapshot saved-state diagnostic, bounded by the immutable v2 protocol."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from itm.models.mdm_correction import ROOT, read, verify, tensor_hashes
from itm.models.mdm_sampling_alignment import describe, difference, digest, input_arrays, AlignmentPackage
from itm.models.mdm_state_diagnostic import StatePackage, CountedEncoder
from itm.models.mdm_run_limits import RunGuard, RunLimitExceeded, sha256

PACKAGE = ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v2/package.json'
PACKAGE_SHA = '471376c4799b80975c09faee4a80e2adfa220a6119a73a325761ed2a89029593'
BRANCHES = ('T','P','S','R')
SNAPSHOTS = ('I','F050','F100','F150','F200','U200')
CAPS = dict(mdm=4608, encoder=432, caption=12)
CALL_CAPS = dict(mdm=1152, encoder=432, caption=12)


def compact_json(path, value):
    """Durable compact ledger; same atomic publication semantics as RunGuard."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.partial-', dir=path.parent)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, separators=(',', ':'), allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


def ledger(package):
    return dict(states=[dict(state_id=r['state_id'], status='planned') for r in package.states],
                jobs=[dict(job_id=r['job_id'], state_id=r['state_id'], snapshot_id=r['snapshot_id'],
                           status='planned', executed=False) for r in package.jobs],
                branches=[dict(branch_id=b, job_id=r['job_id'], status='planned', executed=False)
                          for r in package.jobs for b in r['branch_ids']],
                checks=[dict(check_id=r['check_id'], job_id=r['job_id'], status='planned') for r in package.checks])


def stop_rows(state):
    for key in ('states','jobs','branches','checks'):
        for row in state[key]:
            if row['status']=='planned': row['status']='not_run_after_stop'
            elif row['status']=='running':
                row['status']='failed'
                if key in ('jobs','branches'): row['diagnostic_only']=bool(row.get('raw_saved'))


class TrainingPackage:
    def __init__(self, path=PACKAGE):
        self.path=Path(path).resolve()
        if sha256(self.path)!=PACKAGE_SHA: raise ValueError('Exact frozen sufficiency v2 package required')
        self.files={Path(a['path']).stem:verify(a) for a in read(self.path)['artifacts']}
        self.source_paths=[verify(a) for a in read(self.files['sources'])]
        self.design=read(self.files['design']); self.common=dict(self.design['common'], environment=self.design['environment'])
        self.alignment=AlignmentPackage(verify(self.design['original_Q_package']))
        self.mean,self.std=self.alignment.mean,self.alignment.std
        self.states=read(self.files['states']);self.jobs=read(self.files['jobs'])
        self.checks=read(self.files['software_checks']);self.cases=read(self.files['input_cases'])
        self.snapshots=read(self.files['checkpoints'])
        if ((len(self.states),len(self.jobs),len(self.checks),len(self.cases))!=(192,1152,2880,12)
                or [r['snapshot_id'] for r in self.snapshots]!=list(SNAPSHOTS)):
            raise ValueError('Frozen diagnostic population changed')

    inputs=StatePackage.inputs

    def state(self, snapshot):
        state=torch.load(verify(snapshot['artifact']), map_location='cpu', weights_only=True)
        if tensor_hashes(state)!=snapshot['tensor_hashes'] or state['control_config']!=snapshot['raw_control_config']:
            raise ValueError('Snapshot tensor/config identity mismatch')
        if snapshot['snapshot_id']!='I' and state['execution']!=snapshot['execution']:
            raise ValueError('Snapshot execution identity mismatch')
        for group in ('imu_encoder','adapters'):
            for value in state[group].values():
                if value.dtype!=torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Snapshot dtype/finite mismatch')
        return dict(state, control_config=dict(snapshot['runtime_control_config']))

    def saved(self, row):
        with np.load(verify(row['saved_state']),allow_pickle=False) as z:
            xt,mask=z['x_t'].copy(),z['frame_mask'].copy()
        with np.load(verify(row['target_artifact']),allow_pickle=False) as z:target=z[row['target_array']].copy()
        if row['state_family']=='GT_noised_Q': xt,mask=xt[None],mask[None]
        values=dict(x_t=xt,target=target,frame_mask=mask)
        if {k:describe(v) for k,v in values.items()}!=row['arrays']:
            raise ValueError('Saved state/target/mask identity mismatch')
        if not all(np.isfinite(v).all() for v in values.values()): raise FloatingPointError('Nonfinite saved state')
        return values


class TrainingGuard(RunGuard):
    def __init__(self,context,package,**kwargs):
        if (context.get('version')!='itm-sufficiency-context-v1' or context.get('phase')!='diagnostic'
                or context.get('package_sha256')!=PACKAGE_SHA):
            raise ValueError('Sufficiency bounded context identity mismatch')
        super().__init__(dict(context,phase='sample'),**kwargs)
        self.context=context;self.phase='diagnostic'
        self.state.update(version='itm-sufficiency-receipt-v1',phase=self.phase,
                          context={k:v for k,v in context.items() if k!='expected_artifacts'},
                          **ledger(package),counts={k:dict(reserved=0,actual=0,completed=0,
                            reserved_calls=0,actual_calls=0,completed_calls=0) for k in CAPS})
        self.state['journal_applied_sequence']=0
        self.journal=self.path.with_suffix('.journal.jsonl')
        if self.journal.exists():raise FileExistsError(self.journal)
        self._sequence=0;self._artifact_count=0
        self._jobs={r['job_id']:r for r in self.state['jobs']}
        self._branches={jid:[r for r in self.state['branches'] if r['job_id']==jid] for jid in self._jobs}
        self._checks={jid:[r for r in self.state['checks'] if r['job_id']==jid] for jid in self._jobs}
        self._states={r['state_id']:r for r in self.state['states']}
        compact_json(self.path,self.state);self._journal_ready=True

    def persist(self):
        self.state['last_monotonic']=self.clock()
        if not getattr(self,'_journal_ready',False):
            compact_json(self.path,self.state);return
        jid=self.state.get('active_job');self._sequence+=1
        event=dict(sequence=self._sequence,last_monotonic=self.state['last_monotonic'],
                   status=self.state['status'],counts=self.state['counts'],
                   fields={k:self.state[k] for k in ('active_job','peak_allocated_bytes','allocator_cap_bytes','gpu_uuid','failure','peak_query_error') if k in self.state},
                   jobs=[self._jobs[jid]] if jid else [],branches=self._branches[jid] if jid else [],
                   checks=self._checks[jid] if jid else [],
                   states=[self._states[self._jobs[jid]['state_id']]] if jid else [],
                   artifacts=self.state['artifacts'][self._artifact_count:])
        # A single append followed by fsync durably reserves each call; full
        # snapshots are written at final success/failure, never on every call.
        encoded=(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n').encode()
        with self.journal.open('ab',buffering=0) as stream:
            remaining=memoryview(encoded)
            while remaining:
                written=stream.write(remaining)
                if not written:raise OSError('Journal append made no progress')
                remaining=remaining[written:]
            os.fsync(stream.fileno())
        self._artifact_count=len(self.state['artifacts'])
        if self.state['status'] in ('completed','failed'):
            self.state['journal_applied_sequence']=self._sequence;compact_json(self.path,self.state)

    def before_update(self):raise RunLimitExceeded('Diagnostic forbids training')
    def optimizer_step(self,*args,**kwargs):raise RunLimitExceeded('Diagnostic forbids optimizer')
    def before_denoise(self,*args,**kwargs):raise RunLimitExceeded('Diagnostic forbids denoising')

    def invoke(self,kind,n,function):
        self.check()
        if kind not in CAPS or n!=(4 if kind=='mdm' else 1): raise RunLimitExceeded('Diagnostic call kind/batch mismatch')
        count=self.state['counts'][kind]
        if count['reserved']+n>CAPS[kind] or count['reserved_calls']+1>CALL_CAPS[kind]: raise RunLimitExceeded('Diagnostic call cap')
        count['reserved']+=n;count['reserved_calls']+=1;self.persist();self.check()
        count['actual']+=n;count['actual_calls']+=1
        if kind=='mdm':
            jid=self.state['active_job']
            for key in ('jobs','branches'):
                for row in self.state[key]:
                    if row['job_id']==jid:row['executed']=True
        self.persist()
        result=function();self.ensure_finite(result,kind);self.check()
        count['completed']+=n;count['completed_calls']+=1;self.persist()
        return result

    def finish(self):
        self.check()
        for kind,cap in CAPS.items():
            c=self.state['counts'][kind]
            if (any(c[k]!=cap for k in ('reserved','actual','completed'))
                    or any(c[k]!=CALL_CAPS[kind] for k in ('reserved_calls','actual_calls','completed_calls'))):
                raise RunLimitExceeded('Incomplete diagnostic count: '+kind)
        if any(r['status']!='completed' for k in ('states','jobs','branches','checks') for r in self.state[k]):
            raise RunLimitExceeded('Incomplete diagnostic ledger')
        if self.state['updates_attempted'] or self.state['updates_completed'] or self.state['denoising_calls']:
            raise RunLimitExceeded('Forbidden diagnostic work')
        self.state['status']='completed';self.persist()

    def fail(self,error):stop_rows(self.state);super().fail(error)


def read_receipt(path):
    """Replay durable events after the latest snapshot, including killed workers."""
    path=Path(path);state=read(path);journal=path.with_suffix('.journal.jsonl')
    if not journal.exists():return state
    maps={key:{r[idkey]:r for r in state[key]} for key,idkey in
          (('states','state_id'),('jobs','job_id'),('branches','branch_id'),('checks','check_id'))}
    applied=state.get('journal_applied_sequence',0);previous=0
    with journal.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                state['truncated_journal_tail']=True;break
            event=json.loads(line)
            if event['sequence']!=previous+1:raise ValueError('Journal event sequence gap')
            previous=event['sequence']
            if previous<=applied:continue
            state.update(last_monotonic=event['last_monotonic'],status=event['status'],counts=event['counts'],**event['fields'])
            for key,idkey in (('states','state_id'),('jobs','job_id'),('branches','branch_id'),('checks','check_id')):
                for row in event[key]:maps[key][row[idkey]].update(row)
            state['artifacts'].extend(event['artifacts'])
            state['journal_applied_sequence']=previous
    if applied>previous:raise ValueError('Snapshot references missing journal events')
    if state['status']=='failed':stop_rows(state)
    return state


class SnapshotCache:
    def __init__(self):self.text={};self.control={};self.checkpoint=None

    def begin(self,checkpoint):
        if checkpoint==self.checkpoint: raise ValueError('Snapshot must advance, not resume')
        self.checkpoint=checkpoint;self.control.clear()

    def get(self,case,items,rows,descriptors,package,model,encoder,guard,device,batch_inputs):
        if self.checkpoint is None or any(m.training for m in model.modules()) or encoder.training:
            raise ValueError('Snapshot cache requires bound checkpoint/eval')
        actual=[{k:describe(v) for k,v in input_arrays(i,r).items()} for i,r in zip(items,rows)]
        if actual!=descriptors: raise ValueError('Snapshot cache inputs changed')
        cid=case['cache_id']
        text_id=digest(dict(backbone=package.common['backbone'],caption=case['caption'],
                            cache_id=cid,dtype='float32',eval=True))
        caption_built=cid not in self.text
        if caption_built:
            text=guard.invoke('caption',1,lambda:model.encode_text([case['caption']]))
            if text.dtype!=torch.float32 or text.ndim!=3 or text.shape[1]!=1:
                raise ValueError('Caption must preserve batch1 float32 native return')
            self.text[cid]=dict(identity=text_id,value=text,descriptor=describe(text))
        entry=self.text[cid]
        if entry['identity']!=text_id or describe(entry['value'])!=entry['descriptor']:
            raise ValueError('Caption cache identity/tensor changed')
        identity=digest(dict(checkpoint=self.checkpoint,case=case,inputs=descriptors,
                            normalization=package.common['imu_normalization'],dtype='float32',eval=True))
        control_built=cid not in self.control
        if control_built:
            from itm.models.mdm_correction import encode_control
            counted=CountedEncoder(encoder,guard)
            controls=[torch.zeros((1,case['frames'],encoder.config.latent_dim),device=device,dtype=torch.float32)]
            for item,row in zip(items[1:],rows[1:]):
                _,y=batch_inputs([item],[row],device)
                value,present=encode_control(counted,y,mode='static_delta')
                if not present.all() or value.dtype!=torch.float32:raise ValueError('Control presence/dtype changed')
                controls.append(value)
            value=torch.cat(controls);guard.ensure_finite(value,'control')
            self.control[cid]=dict(identity=identity,value=value,descriptor=describe(value))
        control=self.control[cid]
        if control['identity']!=identity or describe(control['value'])!=control['descriptor']:
            raise ValueError('Control cache identity/tensor changed')
        return dict(text=entry['value'],control=control['value'],text_identity=text_id,
                    control_identity=identity,text_descriptor=entry['descriptor'],control_descriptor=control['descriptor']),caption_built,control_built
