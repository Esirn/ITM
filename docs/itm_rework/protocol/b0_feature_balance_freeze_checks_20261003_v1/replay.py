"""Independent-directory reconstruction, NumPy/files only; no model or GPU."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack,redirect_stdout
from unittest.mock import patch
import builtins
import io
import runpy
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4];CHECKS=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import freeze_b0_feature_balance as f
original=builtins.__import__
def restrict(name,*a,**kw):
    if name.split('.')[0] in ('torch','tensorflow','jax'):raise AssertionError('Model import forbidden')
    return original(name,*a,**kw)
def reject(*a,**kw):raise AssertionError('Process/GPU query forbidden')
with ExitStack() as stack:
    stack.enter_context(patch.object(builtins,'__import__',restrict))
    for name in ('Popen','run','call','check_call','check_output'):stack.enter_context(patch.object(subprocess,name,reject))
    log=io.StringIO();results=[]
    with TemporaryDirectory(prefix='itm02_F_freeze_') as directory:
        p=Path(directory)
        proposal=runpy.run_path(str(f.PROPOSAL/'build_proposal.py'))
        with redirect_stdout(log):proposal['main'](p/'proposal')
        assert len(list((p/'proposal').iterdir()))==7
        for a in sorted((p/'proposal').iterdir()):
            assert a.read_bytes()==(f.PROPOSAL/'package'/a.name).read_bytes(),a.name
            results.append(dict(bundle='01proposal',name=a.name,sha256=f.sha(a),byte_identical=True))
        with redirect_stdout(log):f.guarded_build(p/'freeze')
        assert len(list((p/'freeze').iterdir()))==16
        for a in sorted((p/'freeze').iterdir()):
            assert a.read_bytes()==(ROOT/f.REL/a.name).read_bytes(),a.name
            results.append(dict(bundle='02freeze',name=a.name,sha256=f.sha(a),byte_identical=True))
    f.write(CHECKS/'replay_verification.json',dict(proposal_files=7,freeze_files=16,all_byte_identical=True,
        real_model_forwards=0,GPU_queries=0,new_samples=0,files=results))
    (CHECKS/'replay_details.txt').write_text(log.getvalue())
    print('7proposal files and16freeze files byte-identical; no model/GPU calls')
