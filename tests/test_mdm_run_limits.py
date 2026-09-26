"""CPU-only fault injection; no CUDA context, training run or real diffusion model."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from itm.models.mdm_run_limits import (RunGuard, RunLimitExceeded, GIB, atomic_artifact,
                                       atomic_json, sha256, guarded_main)
from itm.models.mdm_sampling import sample_addressed_ddpm

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'docs/itm_rework/protocol/canonical_method_acceptance_20260925_v1/execution_plan/execution_plan.json'


def runner():
    spec = importlib.util.spec_from_file_location('bounded_runner', ROOT/'scripts/run_mdm_b0_bounded.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def guard(tmp_path, phase='train', **kwargs):
    clock = kwargs.pop('clock', time.monotonic)
    context = dict(phase=phase, deadline_monotonic=clock()+30, receipt_path=str(tmp_path/'receipt.json'),
                   gpu_uuid='GPU-01234567', memory_bytes=24*GIB)
    context.update(kwargs)
    return RunGuard(context, clock=clock)


def test_200_updates_and_final_gate(tmp_path):
    g = guard(tmp_path)
    p = torch.nn.Parameter(torch.tensor(1.))
    optimizer = torch.optim.SGD([p], lr=.001)
    for _ in range(200):
        g.optimizer_step(p.square(), optimizer, [p])
    assert g.state['updates_attempted'] == g.state['updates_completed'] == 200
    before = p.detach().clone()
    with pytest.raises(RunLimitExceeded, match='update cap'):
        g.optimizer_step(p.square(), optimizer, [p])
    torch.testing.assert_close(p, before)
    g.finish()
    assert json.loads(g.path.read_text())['status'] == 'completed'


@pytest.mark.parametrize('where', ['loss', 'gradient', 'gradient_norm', 'optimizer_result'])
def test_nonfinite_never_publishes_checkpoint_or_retries(tmp_path, where):
    g = guard(tmp_path)
    p = torch.nn.Parameter(torch.tensor([1., 1.]))
    optimizer = torch.optim.SGD([p], lr=.1)
    before = p.detach().clone()
    loss = p.sum()
    if where == 'loss':
        loss = loss*float('nan')
    elif where == 'gradient':
        p.register_hook(lambda grad: grad*float('inf'))
    elif where == 'gradient_norm':
        p.register_hook(lambda grad: grad*1e30)  # finite entries, overflowing total norm
    else:
        def corrupt():
            with torch.no_grad(): p.fill_(float('nan'))
        optimizer.step = corrupt
    with pytest.raises((FloatingPointError, RuntimeError)) as caught:
        g.optimizer_step(loss, optimizer, [p])
    g.fail(caught.value)
    assert g.state['updates_completed'] == 0
    assert g.state['failure']['kind'] == 'nonfinite'
    assert not g.state['artifacts']
    if where != 'optimizer_result':
        torch.testing.assert_close(p, before)
        assert g.state['updates_attempted'] == 0
    else:
        assert g.state['updates_attempted'] == 1


def test_timeout_after_update_is_not_final_success(tmp_path):
    now = [0.]
    g = guard(tmp_path, clock=lambda: now[0], deadline_monotonic=1.)
    p = torch.nn.Parameter(torch.tensor(1.)); opt = torch.optim.SGD([p], lr=.1)
    step = opt.step
    def slow_step():
        step(); now[0] = 2.
    opt.step = slow_step
    with pytest.raises(RunLimitExceeded, match='wall deadline') as caught:
        g.optimizer_step(p.square(), opt, [p])
    g.fail(caught.value)
    assert g.state['updates_completed'] == 1
    assert g.state['status'] == 'failed'


def test_allocator_cap_configured_before_allocations_and_one_device(tmp_path, monkeypatch):
    g = guard(tmp_path)
    calls = []
    fake = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_properties=lambda _: SimpleNamespace(total_memory=48*GIB),
        set_per_process_memory_fraction=lambda fraction, device: calls.append((fraction, str(device))),
        synchronize=lambda _: None, max_memory_allocated=lambda _: 10))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-01234567')
    g.configure_cuda(fake, torch.device('cuda:0'))
    assert calls == [(.5, 'cuda:0')]
    assert g.state['allocator_cap_bytes'] == 24*GIB
    fake.cuda.max_memory_allocated = lambda _: 24*GIB+1
    with pytest.raises(RunLimitExceeded, match='memory cap'):
        g.check()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-01234567,GPU-76543210')
    with pytest.raises(ValueError, match='isolated'):
        g.configure_cuda(fake, torch.device('cuda:0'))
    with pytest.raises(ValueError, match='isolated'):
        g.configure_cuda(fake, torch.device('cpu'))


class TinyModel(torch.nn.Module):
    njoints, nfeats = 2, 1


class FakeDiffusion:
    num_timesteps = 3
    def __init__(self, error=None):
        self.calls = 0; self.error = error
    def p_mean_variance(self, model, x, t, **kwargs):
        self.calls += 1
        if self.error == 'oom':
            raise torch.OutOfMemoryError('injected CPU fixture OOM')
        return {'mean': x, 'log_variance': torch.zeros_like(x),
                'pred_xstart': x*float('nan') if self.error == 'nan' else x}


@pytest.mark.parametrize('fault', ['steps', 'branches', 'oom', 'nan', 'timeout'])
def test_actual_ddpm_loop_stops_before_excess_work(tmp_path, fault):
    now = [0.]
    g = guard(tmp_path, 'sample', clock=lambda: now[0], deadline_monotonic=1.,
              max_denoising_calls=2 if fault == 'steps' else 1000,
              max_branch_examples=4 if fault == 'branches' else 4000)
    diffusion = FakeDiffusion(fault)
    if fault == 'timeout':
        now[0] = 1.
    with pytest.raises((RunLimitExceeded, torch.OutOfMemoryError, FloatingPointError)) as caught:
        sample_addressed_ddpm(TinyModel().eval(), diffusion, y={}, lengths=torch.tensor([3]),
                              case_ids=['fixture'], sample_seed=1, backbone_id='fixture', run_guard=g)
    g.fail(caught.value)
    assert diffusion.calls == {'steps': 2, 'branches': 1, 'oom': 1, 'nan': 1, 'timeout': 0}[fault]
    assert g.state['denoising_calls'] == diffusion.calls
    assert g.state['branch_examples'] == 4*diffusion.calls
    assert g.state['status'] == 'failed'


def test_partial_atomic_writes_and_completed_artifacts_survive(tmp_path):
    old = tmp_path/'old.bin'; atomic_artifact(old, lambda f: f.write(b'completed'))
    original = sha256(old)
    new = tmp_path/'new.bin'
    def fail(stream):
        stream.write(b'partial'); raise OSError('injected write error')
    with pytest.raises(OSError): atomic_artifact(new, fail)
    assert not new.exists()
    assert list(tmp_path.glob('new.bin.partial-*'))
    assert sha256(old) == original
    with pytest.raises(FileExistsError): atomic_artifact(old, fail)


CHILD = '''
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
from itm.models.mdm_run_limits import guarded_main,atomic_artifact
output=Path(sys.argv[1]); behavior=sys.argv[2]
def work(args,g):
    if behavior=='hang': time.sleep(10)
    if behavior=='oom':
        class OutOfMemoryError(RuntimeError): pass
        raise OutOfMemoryError('CPU fault injection, no GPU allocated')
    if behavior=='nan': raise FloatingPointError('nonfinite CPU fixture')
    g.state['denoising_calls']=1000;g.state['branch_examples']=4000;g.persist()
    atomic_artifact(output,lambda f:f.write(b'CPU fixture output'),g)
    atomic_artifact(output.with_suffix('.json'),lambda f:f.write(b'{}'),g)
    return 0
guarded_main(SimpleNamespace(run_context=sys.argv[-1]),'sample',work)
'''


@pytest.mark.parametrize('fault', ['oom', 'nan', 'hang'])
def test_queue_keeps_completed_slot_and_stops_remaining_without_retry(tmp_path, fault):
    module = runner()
    jobs = [dict(argv=[sys.executable, '-c', CHILD, str(tmp_path/f'{i}.npz'), behavior],
                 output=str(tmp_path/f'{i}.npz')) for i, behavior in enumerate(['ok', fault, 'ok'])]
    status = dict(status='sampling', slots=[dict(case_id=str(i), status='planned') for i in range(3)],
                  denoising_calls=0, branch_examples=0)
    deadline = time.monotonic()+(1.5 if fault == 'hang' else 8.)
    assert not module.run_jobs(jobs, root=tmp_path, deadline=deadline, gpu_uuid='GPU-fixture', env=os.environ.copy(),
                               status=status, status_path=tmp_path/'status.json')
    assert [s['status'] for s in status['slots']] == ['completed', 'failed', 'not_run_after_stop']
    assert (tmp_path/'0.npz').read_bytes() == b'CPU fixture output'
    assert not (tmp_path/'2.npz').exists() and not (tmp_path/'sample_02.context.json').exists()
    contexts = [json.loads((tmp_path/f'sample_{i:02d}.context.json').read_text()) for i in range(2)]
    assert {c['deadline_monotonic'] for c in contexts} == {deadline}
    assert status['denoising_calls'] == 1000 and status['branch_examples'] == 4000
    if fault == 'hang':
        assert status['slots'][1]['reason'] == 'wall_timeout_process_group_killed'
    else:
        receipt = json.loads((tmp_path/'sample_01.receipt.json').read_text())
        assert receipt['failure']['kind'] == ('oom' if fault == 'oom' else 'nonfinite')


def test_deadline_exhausted_never_starts_child(tmp_path):
    marker = tmp_path/'must-not-exist'
    result = runner().supervise([sys.executable, '-c', f'open({str(marker)!r},"w").close()'],
                                deadline=time.monotonic()-1, log_path=tmp_path/'log', env=os.environ.copy())
    assert result['launched'] is False and not marker.exists()


def test_timeout_kills_child_process_group(tmp_path):
    marker = tmp_path/'orphan-completed'
    nested = f'import time;time.sleep(1.);open({str(marker)!r},"w").close()'
    command = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",sys.argv[1]]);time.sleep(10)'
    result = runner().supervise([sys.executable, '-c', command, nested], deadline=time.monotonic()+.25,
                                log_path=tmp_path/'log', env=os.environ.copy())
    assert result['reason'] == 'wall_timeout_process_group_killed'
    time.sleep(1.1)
    assert not marker.exists()


def test_only_completed_final_200_checkpoint_passes(tmp_path):
    module = runner(); path = tmp_path/'model.pt'
    state = {'execution': {'final_update_200': True, 'updates_completed': 200}}
    torch.save(state, path)
    receipt = dict(status='completed', updates_completed=200, updates_attempted=200,
                   artifacts=[dict(path=str(path), sha256=sha256(path))])
    module.validate_final_checkpoint(path, receipt)
    with pytest.raises(ValueError, match='exactly 200'):
        module.validate_final_checkpoint(path, dict(receipt, updates_completed=199))
    state['execution']['final_update_200'] = False; torch.save(state, path)
    receipt['artifacts'][0]['sha256'] = sha256(path)
    with pytest.raises(ValueError, match='Partial checkpoint'):
        module.validate_final_checkpoint(path, receipt)


def test_prepare_needs_no_gpu_and_execution_needs_authorization(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, 'preflight_gpu', lambda _: pytest.fail('GPU must not be queried'))
    monkeypatch.setattr(sys, 'argv', ['runner', '--plan', str(PLAN), '--run-dir', str(tmp_path/'prepared')])
    assert module.main() == 0
    state = json.loads((tmp_path/'prepared/runtime/status.json').read_text())
    assert state['status'] == 'prepared_not_executed' and len(state['slots']) == 36
    assert state['caps']['shared_sampling_seconds'] == 10800
    monkeypatch.setattr(sys, 'argv', ['runner', '--plan', str(PLAN), '--run-dir', str(tmp_path/'never'), '--execute'])
    with pytest.raises(ValueError, match='authorization'):
        module.main()
    assert not (tmp_path/'never').exists()


def test_changed_checkpoint_identity_stops_before_model_setup(tmp_path):
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'accepted')
    entry = dict(path=str(checkpoint), sha256=sha256(checkpoint))
    checkpoint.write_bytes(b'changed')
    context = tmp_path/'context.json'
    atomic_json(context, dict(phase='sample', deadline_monotonic=time.monotonic()+10,
                             receipt_path=str(tmp_path/'receipt.json'), expected_artifacts=[entry]))
    with pytest.raises(ValueError, match='identity mismatch'):
        guarded_main(SimpleNamespace(run_context=str(context)), 'sample',
                     lambda *_: pytest.fail('Must stop before model setup'))
    receipt = json.loads((tmp_path/'receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['denoising_calls'] == 0


def test_partial_training_finish_is_rejected(tmp_path):
    g = guard(tmp_path)
    g.state['updates_attempted'] = g.state['updates_completed'] = 199
    with pytest.raises(RunLimitExceeded, match='final update 200'):
        g.finish()


def test_authorization_is_bound_to_one_run_directory_and_gpu(tmp_path):
    module = runner()
    approval = tmp_path/'approval.json'
    record = dict(authorized=True, approved_by='user', user_instruction_reference='CPU fixture, not a real approval',
                  budget_sha256='fixture', execution_plan_sha256=sha256(PLAN),
                  run_directory=str(tmp_path/'run1'), gpu_uuid='GPU-fixture')
    atomic_json(approval, record)
    module.authorization(approval, PLAN, 'fixture', run_directory=tmp_path/'run1', gpu_uuid='GPU-fixture')
    for run, gpu in [('run2', 'GPU-fixture'), ('run1', 'GPU-other')]:
        with pytest.raises(ValueError, match='authorization'):
            module.authorization(approval, PLAN, 'fixture', run_directory=tmp_path/run, gpu_uuid=gpu)
