"""Frozen F loss, provenance and bounded counters. No compute authorization."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from itm.models.mdm_correction import (ROOT, PACKAGE_SHA as INITIAL_PACKAGE_SHA, read, verify,
    artifact, array_sha, tensor_hashes, environment)
from itm.models.mdm_sampling_alignment import AlignmentPackage, input_arrays, describe
from itm.models.mdm_imu_control import MDMIMUControlConfig
from itm.models.mdm_run_limits import RunGuard, RunLimitExceeded, sha256

PACKAGE = ROOT/'docs/itm_rework/protocol/b0_feature_balance_20261003_v1/package.json'
PACKAGE_SHA = 'a4189b89f1924af022c67dea71c02c2792e65f4a185775501b07d037fba51af9'
ARM = 'F_root_ric_balanced'
LOSS_VERSION = 'normalized-x0-root-ric-half-v1'
SCHEMA = 'itm-feature-balance-control-v1'


def feature_losses(prediction, target, valid):
    """Per-example valid-frame means; production F is immutable, all263 retained."""
    if (prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1:3] != (263, 1)
            or valid.shape != (prediction.shape[0], prediction.shape[-1]) or valid.dtype != torch.bool
            or prediction.dtype not in (torch.float32, torch.float64) or target.dtype != prediction.dtype):
        raise ValueError('F requires [B,263,1,T] float32/64 predictions and boolean [B,T] mask')
    if not valid.any(-1).all():
        raise ValueError('No empty F examples')
    # Select before arithmetic: invalid padding cannot contribute NaNs or gradients.
    error = torch.where(valid[:, None, None], prediction-target, 0)
    if not torch.isfinite(error).all():
        raise FloatingPointError('Nonfinite F error')
    squared = error.square(); frames = valid.sum(-1)
    d = squared[:, :67].sum((1, 2, 3))/(67*frames)
    a = squared[:, 67:].sum((1, 2, 3))/(196*frames)
    u = squared.sum((1, 2, 3))/(263*frames)
    return dict(uniform=u, root_ric=d, remaining=a, balanced=.5*d+.5*a)


class FeaturePackage:
    def __init__(self, path=PACKAGE):
        self.path = Path(path).resolve()
        if sha256(self.path) != PACKAGE_SHA: raise ValueError('Exact frozen F package required')
        self.files = {Path(a['path']).stem: verify(a) for a in read(self.path)['artifacts']}
        self.source_paths = [verify(a) for a in read(self.files['sources'])]
        self.design = read(self.files['design']); self.common = dict(self.design['common'], environment=self.design['environment'])
        self.loss = read(self.files['loss'])
        weights = np.r_[np.full(67, 263/134), np.full(196, 263/392)]
        if (self.loss['loss_version'] != LOSS_VERSION or self.loss['weights'] != weights.tolist()
                or self.loss['weights_sha256'] != array_sha(weights) or self.loss != self.design['loss']):
            raise ValueError('Frozen F loss changed')
        self.alignment = AlignmentPackage(verify(self.design['old_sampling_package']))
        for k in ('backbone', 'control_config', 'imu_normalization', 'training_input_provenance',
                  'motion_normalization', 'optimizer', 'environment'):
            baseline=self.alignment.common.get(k,self.alignment.old.common.get(k))
            if self.common[k] != baseline: raise ValueError('F common identity: '+k)
        self.mean, self.std = self.alignment.mean, self.alignment.std
        self.initial = verify(self.design['initial'])
        self.schedule = read(self.files['training_schedule'])['rows']
        if self.schedule != self.alignment.old.schedule: raise ValueError('F training plan changed')
        self.samples = read(self.files['sampling_slots']); self.traces = read(self.files['trace_slots'])
        self.checks = read(self.files['software_checks'])
        if (len(self.schedule), len(self.samples), len(self.traces), len(self.checks)) != (600,60,240,48):
            raise ValueError('F denominator changed')

    def inputs(self, row):
        return self.alignment.inputs(row) if 'case_id' in row else self.alignment.old.inputs(row)

    def initial_state(self):
        verify(self.design['initial'])
        state = torch.load(self.initial, map_location='cpu', weights_only=True)
        expected = self.alignment.old_manifest['initialization']['tensor_hashes']
        if (tensor_hashes(state) != expected or state['tensor_hashes'] != expected
                or state['environment'] != self.common['environment'] or state['package_sha256'] != INITIAL_PACKAGE_SHA
                or environment() != self.common['environment']):
            raise ValueError('Original initialization/environment mismatch')
        return state

    def identity(self, implementation_sha):
        state = self.initial_state()
        return deepcopy(dict(package_sha256=PACKAGE_SHA, implementation_sha256=implementation_sha,
            loss_version=LOSS_VERSION, loss_weights=self.loss['weights'], loss_weights_sha256=self.loss['weights_sha256'],
            initialization_sha256=sha256(self.initial), initial_tensor_hashes=state['tensor_hashes'],
            initial_package_sha256=INITIAL_PACKAGE_SHA, training_schedule_sha256=sha256(self.files['training_schedule']),
            imu_normalization=self.common['imu_normalization'], training_input_provenance=self.common['training_input_provenance'],
            environment=self.common['environment']))

    def state(self, checkpoint, implementation_sha):
        state = torch.load(verify(checkpoint), map_location='cpu', weights_only=True)
        validate_state(state, self, implementation_sha)
        return state


def parameter_contract(initial):
    # Module registration preserves backbone's seqTransEncoder position; E is appended.
    return ([dict(name='seqTransEncoder.adapters.'+k, shape=list(v.shape), dtype=str(v.dtype))
             for k,v in initial['adapters'].items() if k.endswith('.weight')]
            + [dict(name='imu_control_encoder.'+k, shape=list(v.shape), dtype=str(v.dtype))
               for k,v in initial['imu_encoder'].items()])


def audit_training(model, initial):
    actual = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype)) for n,p in model.named_parameters() if p.requires_grad]
    expected = parameter_contract(initial)
    if actual != expected: raise ValueError('F trainable parameter set/order changed')
    for n,p in model.named_parameters():
        if p.dtype != torch.float32 and not n.startswith('clip_model.'):
            raise ValueError('F native denoiser/control precision changed')
    for n,m in model.named_modules():
        train = n == 'imu_control_encoder' or n.startswith('imu_control_encoder.') or n == 'seqTransEncoder.adapters' or n.startswith('seqTransEncoder.adapters.')
        if m.training != train: raise ValueError('F frozen/eval mode changed: '+n)
    frozen = {n:array_sha(p) for n,p in model.named_parameters() if not p.requires_grad}
    return dict(trainable=actual, frozen_hashes=frozen, frozen_unchanged=True)


def check_frozen(model, audit):
    if {n:array_sha(p) for n,p in model.named_parameters() if not p.requires_grad} != audit['frozen_hashes']:
        raise ValueError('F frozen parameters changed')


def validate_state(state, package, implementation_sha):
    if state.get('checkpoint_schema') != SCHEMA or state.get('arm') != ARM:
        raise ValueError('Only F checkpoint schema/arm permitted')
    for k,v in package.identity(implementation_sha).items():
        if state.get(k) != v: raise ValueError('F checkpoint provenance mismatch: '+k)
    expected = asdict(MDMIMUControlConfig(**dict(package.common['control_config'], reference_mode='static_delta')))
    if state['control_config'] != expected: raise ValueError('F control config changed')
    execution = state['execution']
    if execution != dict(updates_completed=200, updates_attempted=200, final_update_200=True,
                         exposures=600, mdm_examples=600, encoder_examples=1200, caption_calls=600):
        raise ValueError('F requires complete final200')
    initial = package.initial_state()
    if state['training_audit']['trainable'] != parameter_contract(initial) or not state['training_audit']['frozen_unchanged']:
        raise ValueError('F training parameter audit mismatch')
    defaults = dict(package.common['optimizer']); defaults.pop('name'); defaults['betas'] = tuple(defaults['betas'])
    if state['optimizer_defaults'] != defaults: raise ValueError('F optimizer changed')
    if tensor_hashes(state) != state['tensor_hashes']: raise ValueError('F trained tensor identity changed')
    if state['tensor_hashes'] == initial['tensor_hashes']: raise ValueError('F checkpoint contains only initialization')
    historical=torch.load(verify(package.design['historical_checkpoint']),map_location='cpu',weights_only=True)
    if tensor_hashes(state)==tensor_hashes(historical):raise ValueError('Historical R relabeled as F')
    for group in ('imu_encoder','adapters'):
        if list(state[group])!=list(initial[group]):raise ValueError('F tensor keys changed')
        for name,value in state[group].items():
            reference=initial[group][name]
            if value.shape!=reference.shape or value.dtype!=reference.dtype or not torch.isfinite(value).all():
                raise ValueError('F trained tensor shape/dtype/finite mismatch')
            if group=='adapters' and name.endswith('.bias') and not torch.equal(value,reference):
                raise ValueError('F frozen adapter bias changed')
    if len(state['optimizer']['param_groups']) != 1 or len(state['optimizer']['param_groups'][0]['params']) != len(parameter_contract(initial)):
        raise ValueError('F optimizer parameter count changed')
    if not state['optimizer']['state']:raise ValueError('F final optimizer state missing')


def ledger(package):
    return dict(updates=[dict(update=i,status='planned') for i in range(1,201)],
        exposures=[dict(exposure_id=r['exposure_id'],update=r['update'],status='planned') for r in package.schedule],
        samples=[dict(case_id=r['case_id'],status='planned',output='samples/'+r['case_id']+'.npz') for r in package.samples],
        traces=[dict(trace_id=r['trace_id'],case_id=r['case_id'],status='planned') for r in package.traces],
        checks=[dict(check_id=r['check_id'],status='planned') for r in package.checks])


def stop_rows(status):
    for key in ('updates','exposures','samples','traces','checks'):
        for row in status.get(key,[]):
            if row['status']=='planned': row['status']='not_run_after_stop'
            elif row['status']=='running': row['status']='failed'


class FeatureGuard(RunGuard):
    def __init__(self, context, **kwargs):
        if context.get('version')!='itm-feature-context-v1' or context.get('package_sha256')!=PACKAGE_SHA:
            raise ValueError('F context required')
        super().__init__(context, **kwargs)
        self.caps = dict(mdm=600 if self.phase=='train' else 4000,
                         encoder=1200 if self.phase=='train' else 2, caption=600 if self.phase=='train' else 1)
        self.state.update(version='itm-feature-receipt-v1', completed_slots=[],completed_traces=[],
            exposures=0,mdm_examples=0,encoder_examples=0,mdm_calls=0,caption_calls=0,
            counts={k:dict(reserved=0,actual=0,completed=0,calls=0,completed_calls=0) for k in self.caps})
        self.persist()

    def enter(self, kind, n):
        self.check(); c=self.state['counts'][kind]
        if n!=(4 if kind=='mdm' and self.phase=='sample' else 1) or c['reserved']+n>self.caps[kind]:
            raise RunLimitExceeded('F '+kind+' call cap/batch')
        c['reserved']+=n;c['actual']+=n;c['calls']+=1
        self.state.update(mdm_examples=self.state['counts']['mdm']['actual'],mdm_calls=self.state['counts']['mdm']['calls'],
            encoder_examples=self.state['counts']['encoder']['actual'],caption_calls=self.state['counts']['caption']['calls'])
        self.persist()

    def returned(self,kind,n,value):
        self.ensure_finite(value,kind);self.check()
        c=self.state['counts'][kind];c['completed']+=n;c['completed_calls']+=1;self.persist()
        return value

    def invoke(self,kind,n,fn):
        self.enter(kind,n)
        return self.returned(kind,n,fn())

    def before_update(self):
        if self.phase!='train': raise RunLimitExceeded('F sample forbids training')
        super().before_update()

    def before_denoise(self,case_id,step,branch_examples):
        if self.phase!='sample' or branch_examples!=4 or step!=999-self.state['denoising_calls']:
            raise RunLimitExceeded('F sample 1000-step order')
        super().before_denoise(case_id,step,branch_examples)

    def finish(self):
        self.check()
        for kind,cap in self.caps.items():
            c=self.state['counts'][kind]
            if any(c[k]!=cap for k in ('reserved','actual','completed')) or c['calls']!=c['completed_calls']:
                raise RunLimitExceeded('Incomplete F '+kind)
        if self.phase=='train':
            if self.state['exposures']!=600 or self.state['updates_completed']!=200 or self.state['updates_attempted']!=200:
                raise RunLimitExceeded('Incomplete F training')
        elif self.state['updates_attempted'] or self.state['denoising_calls']!=1000 or self.state['branch_examples']!=4000:
            raise RunLimitExceeded('Incomplete F sampling')
        if self.state['completed_slots']!=self.context['expected_slots'] or self.state['completed_traces']!=self.context['expected_traces']:
            raise RunLimitExceeded('Incomplete F slot identities')
        self.state['status']='completed';self.persist()


class ForwardAccounting:
    """Non-mutating hooks on existing forwards, not extra model evaluations."""
    def __init__(self, model, encoder, guard):
        self.handles=[]
        for kind,module in (('mdm',model),('encoder',encoder)):
            self.handles.append(module.register_forward_pre_hook(lambda m,args,k=kind:guard.enter(k,len(args[0]))))
            def after(m,args,result,k=kind): guard.returned(k,len(args[0]),result)
            self.handles.append(module.register_forward_hook(after))
    def close(self):
        for h in self.handles:h.remove()
