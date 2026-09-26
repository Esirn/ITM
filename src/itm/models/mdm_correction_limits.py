"""Correction-specific cooperative accounting, retaining the old B0 guard unchanged."""
from itm.models.mdm_run_limits import RunGuard, RunLimitExceeded


class CorrectionGuard(RunGuard):
    def __init__(self, context, **kwargs):
        if context.get('version') != 'itm-correction-context-v1':
            raise ValueError('Correction context required')
        phase = context['phase']
        if phase not in ('probe', 'train', 'sample'):
            raise ValueError('Unknown correction phase')
        super().__init__(dict(context, phase='sample' if phase == 'probe' else phase), **kwargs)
        self.phase = phase
        self.context = context
        self.state.update(version='itm-correction-receipt-v1', phase=phase, context=context,
                          mdm_examples=0, encoder_examples=0, exposures=0, completed_slots=[],
                          initialization_sha256=context.get('initialization_sha256'))
        self.caps = context['counters']
        limits = {'probe': dict(mdm_examples=480, encoder_examples=480, exposures=0),
                  'train': dict(mdm_examples=600, encoder_examples=1200 if context.get('arm') == 'R_static_reference_delta' else 600, exposures=600),
                  'sample': dict(mdm_examples=4000, encoder_examples=2 if context.get('arm') == 'R_static_reference_delta' else 1, exposures=0)}[phase]
        if self.caps != limits:
            raise ValueError('Correction counters differ from frozen phase/arm')
        self.persist()

    def reserve(self, kind, count=1):
        self.check()
        if kind not in self.caps or count < 1 or self.state[kind]+count > self.caps[kind]:
            raise RunLimitExceeded('correction '+kind+' cap reached')
        self.state[kind] += count
        self.persist()

    def before_denoise(self, case_id, step, branch_examples):
        self.reserve('mdm_examples', branch_examples)
        super().before_denoise(case_id, step, branch_examples)

    def finish(self):
        self.check()
        expected = 480 if self.phase == 'probe' else 600 if self.phase == 'train' else 4000
        if self.state['mdm_examples'] != expected:
            raise RunLimitExceeded('Incomplete correction MDM count')
        if self.phase == 'probe' and len(self.state['completed_slots']) != 480:
            raise RunLimitExceeded('Incomplete 480 probe slots')
        if self.phase == 'train' and (self.state['updates_completed'] != 200 or self.state['updates_attempted'] != 200 or self.state['exposures'] != 600):
            raise RunLimitExceeded('Incomplete final200 training')
        if self.phase == 'sample' and (self.state['denoising_calls'] != 1000 or self.state['branch_examples'] != 4000):
            raise RunLimitExceeded('Incomplete 1000-step sampling')
        self.state['status'] = 'completed'
        self.persist()
