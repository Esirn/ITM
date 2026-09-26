"""Owner02 differential acceptance; CPU loader fixtures, never real MDM/GPU."""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('revision,mode', [('before','none'),('fixed','none'),('fixed','static_delta')])
def test_old_failure_and_both_fixed_reference_modes(tmp_path,monkeypatch,revision,mode):
    # Extend 01's actual-loader fixture to both arms and an in-memory old loader.
    spec=importlib.util.spec_from_file_location('owner02_loader_fixture',ROOT/'tests/test_mdm_correction.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    original_script=fixture.script;original_config=fixture.small_config
    monkeypatch.setattr(fixture,'small_config',lambda **kw:replace(original_config(**kw),reference_mode=mode))
    checked=[]
    def load_script(name):
        worker=original_script(name)
        if revision=='before':
            source=subprocess.check_output(['git','show','8e4df1e:scripts/run_mdm_correction_worker.py'],cwd=ROOT,text=True)
            fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_model')
            namespace=dict(worker.__dict__)
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'git:8e4df1e:load_model','exec'),namespace)
            worker.load_model=namespace['load_model']
        target=worker.load_model
        def checked_loader(*a,**kw):
            result=target(*a,**kw)
            model,diffusion,controlled,encoder,raw,handles,backbone=result
            assert encoder.config.reference_mode==controlled.reference_mode==mode
            assert all(p.device.type=='cpu' for p in model.parameters())
            assert all(not m.training for m in model.modules())
            if mode=='static_delta':assert all(not layer.bias.requires_grad for layer in controlled.adapters)
            checked.append(mode)
            return result
        worker.load_model=checked_loader
        return worker
    monkeypatch.setattr(fixture,'script',load_script)
    # Fixture extracts original upstream _apply/train by AST and prohibits every
    # nn.Module forward; the heavyweight factory/imports alone are substituted.
    if revision=='before':
        with pytest.raises(AttributeError,match="'NoneType' object has no attribute 'eval'"):
            fixture.test_load_model_with_upstream_nonreturning_device_and_train_methods(tmp_path,monkeypatch)
        assert checked==[]
    else:
        fixture.test_load_model_with_upstream_nonreturning_device_and_train_methods(tmp_path,monkeypatch)
        assert checked==[mode]
