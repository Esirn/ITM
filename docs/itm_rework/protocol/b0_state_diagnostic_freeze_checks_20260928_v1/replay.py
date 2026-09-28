"""CPU-only independent-directory reconstruction of both frozen evidence bundles."""
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import builtins
import io
import json
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[4]
CHECKS = Path(__file__).resolve().parent
original = builtins.__import__
blocked = []

def restricted(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'tensorflow', 'jax'):
        blocked.append(name)
        raise AssertionError('Neural import forbidden')
    return original(name, *args, **kwargs)

def reject(*args, **kwargs):
    blocked.append('process')
    raise AssertionError('Process/GPU query forbidden')

with ExitStack() as stack:
    stack.enter_context(patch.object(builtins, '__import__', restricted))
    for name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
        stack.enter_context(patch.object(subprocess, name, reject))
    freeze = runpy.run_path(str(ROOT / 'scripts/freeze_b0_state_diagnostic.py'))
    proposal = ROOT / 'docs/itm_rework/method/b0_state_diagnostic_20260928_v1'
    checks = []
    output = io.StringIO()
    with TemporaryDirectory(prefix='itm02_state_freeze_') as directory:
        temporary = Path(directory)
        p = runpy.run_path(str(proposal / 'build_proposal.py'))
        rebuilt_proposal = temporary / 'proposal'
        rebuilt_proposal.mkdir()
        p['main'].__globals__['OUT'] = rebuilt_proposal
        with redirect_stdout(output):
            p['main']()
        names = sorted(x.name for x in rebuilt_proposal.iterdir())
        assert len(names) == 7
        for name in names:
            assert (rebuilt_proposal / name).read_bytes() == (proposal / name).read_bytes(), name
            checks.append(dict(bundle='01_proposal', name=name, byte_identical=True, sha256=freeze['sha'](proposal / name)))
        rebuilt = temporary / 'freeze'
        with redirect_stdout(output):
            freeze['guarded_build'](rebuilt)
        frozen = ROOT / freeze['REL']
        assert sorted(x.name for x in rebuilt.iterdir()) == sorted(x.name for x in frozen.iterdir())
        for path in sorted(rebuilt.iterdir()):
            assert path.read_bytes() == (frozen / path.name).read_bytes(), path.name
            checks.append(dict(bundle='02_freeze', name=path.name, byte_identical=True, sha256=freeze['sha'](path)))
        assert len(checks) == 20
        for entry in freeze['read'](frozen / 'sources.json'):
            assert freeze['sha'](ROOT / entry['path']) == entry['sha256']
    assert not blocked
    result = dict(proposal_files_byte_identical=7, freeze_files_byte_identical=13,
        sources_verified=len(freeze['read'](frozen / 'sources.json')), forbidden_operations=blocked,
        real_model_forwards=0, GPU_queries=0, new_samples=0, checks=checks)
    freeze['write'](CHECKS / 'replay_verification.json', result)
    (CHECKS / 'replay_details.txt').write_text(output.getvalue())
    print(json.dumps({k: value for k, value in result.items() if k != 'checks'}))
