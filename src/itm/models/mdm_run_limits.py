"""Method-local bounded-run receipts, CUDA allocator cap and numerical stops.

The parent supervisor supplies absolute monotonic deadlines and kills a hung
process group. These cooperative checks never extend those deadlines.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

GIB = 1024**3


class RunLimitExceeded(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.partial-', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_artifact(path, writer, guard=None):
    """Publish once after a complete write; leave partial files on hard failures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    if guard:
        guard.check()
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.partial-', dir=path.parent)
    with os.fdopen(fd, 'wb') as stream:
        writer(stream)
        stream.flush(); os.fsync(stream.fileno())
    if guard:
        guard.check()
    os.link(temporary, path)  # atomic, and never overwrites an existing artifact
    os.unlink(temporary)
    if guard:
        guard.state['artifacts'].append({'path': str(path.resolve()), 'sha256': sha256(path)})
        guard.persist()
        guard.check()


class RunGuard:
    def __init__(self, context, *, clock=time.monotonic):
        self.context = context
        self.clock = clock
        self.deadline = float(context['deadline_monotonic'])
        if not math.isfinite(self.deadline):
            raise ValueError('Finite absolute monotonic deadline required')
        self.phase = context['phase']
        if self.phase not in ('train', 'sample'):
            raise ValueError('Unknown bounded-run phase')
        self.max_updates = int(context.get('max_updates', 200))
        self.max_steps = int(context.get('max_denoising_calls', 1000))
        self.max_branches = int(context.get('max_branch_examples', 4000))
        self.memory_bytes = int(context.get('memory_bytes', 24*GIB))
        if not (0 < self.max_updates <= 200 and 0 < self.max_steps <= 36000
                and 0 < self.max_branches <= 144000 and 0 < self.memory_bytes <= 24*GIB):
            raise ValueError('Limits exceed the B0 budget')
        self.path = Path(context['receipt_path']).resolve()
        if self.path.exists():
            raise FileExistsError(self.path)
        self.torch = self.device = None
        self.state = dict(version='itm-b0-run-receipt-v1', phase=self.phase, status='running',
                          context=context, pid=os.getpid(), started_monotonic=clock(),
                          updates_attempted=0, updates_completed=0, denoising_calls=0,
                          branch_examples=0, peak_allocated_bytes=0, artifacts=[])
        self.persist()

    @classmethod
    def from_path(cls, path, phase):
        if not path:
            return None
        context = json.loads(Path(path).read_text())
        if context['phase'] != phase:
            raise ValueError('Wrong bounded-run context phase')
        return cls(context)

    def persist(self):
        self.state['last_monotonic'] = self.clock()
        atomic_json(self.path, self.state)

    def configure_cuda(self, torch, device):
        """Called before any model transfer/allocation. No CPU execution fallback."""
        expected = self.context['gpu_uuid']
        if (not expected.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != expected
                or device.type != 'cuda' or device.index != 0 or torch.cuda.device_count() != 1):
            raise ValueError('Bounded B0 requires exactly the isolated authorized GPU at cuda:0')
        total = int(torch.cuda.get_device_properties(device).total_memory)
        torch.cuda.set_per_process_memory_fraction(min(self.memory_bytes, total)/total, device)
        self.torch, self.device = torch, device
        self.state['allocator_cap_bytes'] = min(self.memory_bytes, total)
        self.state['gpu_uuid'] = expected
        self.persist()
        self.check()

    def check(self):
        if self.clock() >= self.deadline:
            raise RunLimitExceeded('absolute wall deadline reached')
        if self.torch is not None:
            self.torch.cuda.synchronize(self.device)
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
            self.state['peak_allocated_bytes'] = max(self.state['peak_allocated_bytes'], peak)
            if peak > self.memory_bytes:
                raise RunLimitExceeded('allocated CUDA memory cap exceeded')
            if self.clock() >= self.deadline:
                raise RunLimitExceeded('absolute wall deadline reached after CUDA synchronization')

    def ensure_finite(self, value, label):
        import torch
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                raise FloatingPointError('nonfinite '+label)
        elif isinstance(value, dict):
            for key, child in value.items():
                self.ensure_finite(child, label+'.'+str(key))
        elif isinstance(value, (tuple, list)):
            for child in value:
                self.ensure_finite(child, label)
        elif isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError('nonfinite '+label)

    def before_update(self):
        self.check()
        if self.state['updates_attempted'] >= self.max_updates:
            raise RunLimitExceeded('optimizer update cap reached')

    def optimizer_step(self, loss, optimizer, parameters):
        import torch
        self.before_update()
        self.ensure_finite(loss, 'loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.check()
        self.ensure_finite([p.grad for p in parameters if p.grad is not None], 'gradients')
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        self.before_update()
        self.state['updates_attempted'] += 1
        self.persist()  # reserve the update even if optimizer.step subsequently fails
        optimizer.step()
        self.ensure_finite(parameters, 'updated parameters')
        self.ensure_finite(list(optimizer.state.values()), 'optimizer state')
        self.state['updates_completed'] += 1
        self.persist()
        self.check()

    def before_denoise(self, case_id, step, branch_examples):
        self.check()
        if (self.state['denoising_calls'] >= self.max_steps or
                self.state['branch_examples'] + branch_examples > self.max_branches):
            raise RunLimitExceeded('denoising/branch forward cap reached')
        self.state.update(active_case_id=str(case_id), active_step=int(step))
        self.state['denoising_calls'] += 1
        self.state['branch_examples'] += branch_examples
        self.persist()

    def finish(self):
        self.check()
        if self.phase == 'train' and self.state['updates_completed'] != 200:
            raise RunLimitExceeded('only final update 200 qualifies for B0 sampling')
        self.state['status'] = 'completed'
        self.persist()

    def fail(self, error):
        name = type(error).__name__
        kind = ('oom' if 'OutOfMemory' in name or 'out of memory' in str(error).lower() else
                'nonfinite' if isinstance(error, FloatingPointError) or 'non-finite' in str(error).lower() else
                'limit' if isinstance(error, RunLimitExceeded) else 'exception')
        # Reading allocator counters allocates no CUDA tensors. No rescue step.
        if self.torch is not None:
            try:
                peak = int(self.torch.cuda.max_memory_allocated(self.device))
                self.state['peak_allocated_bytes'] = max(self.state['peak_allocated_bytes'], peak)
            except Exception as query_error:
                self.state['peak_query_error'] = str(query_error)
        self.state.update(status='failed', failure={'kind': kind, 'type': name, 'message': str(error)})
        self.persist()


def guarded_main(args, phase, implementation):
    guard = RunGuard.from_path(getattr(args, 'run_context', None), phase)
    try:
        if guard:
            for entry in guard.context.get('expected_artifacts', []):
                if sha256(entry['path']) != entry['sha256']:
                    raise ValueError('Run input identity mismatch: '+entry['path'])
            guard.check()
        result = implementation(args, guard)
        if guard:
            guard.finish()
        return result
    except BaseException as error:
        if guard:
            guard.fail(error)
        raise
