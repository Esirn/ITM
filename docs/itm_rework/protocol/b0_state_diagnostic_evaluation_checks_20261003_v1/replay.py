"""Recompute all fifteen result files in fresh directory; compare bytes."""
from contextlib import ExitStack,redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import builtins
import io
import json
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_b0_state_diagnostic as evaluate
import summarize_b0_state_diagnostic as tables
from freeze_b0_state_diagnostic import sha,write
CHECKS=Path(__file__).resolve().parent
original=builtins.__import__
def restrict(name,*a,**kw):
    if name.split('.')[0] in ('torch','tensorflow','jax'):raise AssertionError('Neural imports forbidden')
    return original(name,*a,**kw)
def reject(*a,**kw):raise AssertionError('Process/GPU query forbidden')
with ExitStack() as stack:
    stack.enter_context(patch.object(builtins,'__import__',restrict))
    for name in ('Popen','run','call','check_call','check_output'):stack.enter_context(patch.object(subprocess,name,reject))
    old=io.StringIO()
    with redirect_stdout(old):
        runpy.run_path(str(evaluate.METHOD/'verify_delivery.py'),run_name='__main__')
    assert json.loads(old.getvalue())==evaluate.read(evaluate.METHOD/'delivery_integrity.json')
    messages=io.StringIO()
    with TemporaryDirectory(prefix='itm02_V_effects_replay_') as d:
        out=Path(d)/'evaluation'
        with redirect_stdout(messages):
            evaluate.guarded_evaluate(out)
            tables.build(out)
        assert sorted(p.name for p in out.iterdir())==sorted(p.name for p in evaluate.OUT.iterdir())
        results=[]
        for p in sorted(out.iterdir()):
            assert p.read_bytes()==(evaluate.OUT/p.name).read_bytes(),p.name
            results.append(dict(name=p.name,byte_identical=True,sha256=sha(p)))
        assert len(results)==15
    write(CHECKS/'replay_verification.json',dict(files=results,byte_identical_files=15,
        owner01_integrity_replayed_equal=True,real_model_forwards=0,GPU_queries=0,new_samples=0))
    (CHECKS/'replay_details.txt').write_text(messages.getvalue())
    print('15 result files byte-identical;01 integrity replay equal;no model/GPU operations')
