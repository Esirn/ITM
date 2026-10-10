"""Synthetic NVML replies only; no GPU or model access in these tests."""
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared_preflight',ROOT/'scripts/itm_shared_gpu_preflight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup(monkeypatch,gpu='GPU-abc, 24564, 22000, 23, 5\n',apps='GPU-abc, 1234, python, 1600\n'):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        assert argv[1]=='--id=GPU-abc' and kwargs['check'] and kwargs['timeout']==10
        return SimpleNamespace(stdout=apps if 'query-compute-apps' in argv[2] else gpu,stderr='')
    monkeypatch.setattr(module.subprocess,'run',run)
    return calls


def test_low_load_with_existing_process_is_allowed_and_archived(tmp_path,monkeypatch):
    calls=setup(monkeypatch);sleeps=[]
    result=module.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',sleep=sleeps.append)
    assert result['status']=='passed' and result['compute_process_count']==1
    assert result['compute_processes'][0]['pid']==1234
    assert len(result['samples'])==3 and result['queries_attempted']==len(calls)==4
    assert sleeps==[2,2] and (tmp_path/'preflight/compute_processes.stdout').read_text().startswith('GPU-abc')


def test_no_compute_process_is_also_allowed(tmp_path,monkeypatch):
    setup(monkeypatch,apps='')
    result=module.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',sleep=lambda _:None)
    assert result['compute_process_count']==0


@pytest.mark.parametrize('gpu,match',[
    ('GPU-abc, 24564, 3000, 3, 1\n','free memory'),
    ('GPU-abc, 24564, 22000, 80, 1\n','utilization'),
    ('GPU-abc, 24564, 22000, 3, 80\n','utilization'),
    ('GPU-abc, 24564, 22000, N/A, 1\n','unavailable'),
    ('GPU-def, 24564, 22000, 3, 1\n','identity'),
    ('GPU-abc, 24564, 25000, 3, 1\n','Invalid'),
    ('GPU-abc, 24564, 22000, 3, 1\nGPU-abc,24564,22000,3,1\n','shape')])
def test_rejected_resources_or_unknown_identity_stop_and_keep_raw(tmp_path,monkeypatch,gpu,match):
    calls=setup(monkeypatch,gpu=gpu)
    with pytest.raises(RuntimeError,match=match):
        module.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',sleep=lambda _:None)
    assert len(calls)==1 and (tmp_path/'preflight/gpu_01.stdout').read_text()==gpu


def test_equal_resource_boundaries_are_allowed(tmp_path,monkeypatch):
    setup(monkeypatch,gpu='GPU-abc, 24564, 4096, 50, 50\n')
    assert module.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',sleep=lambda _:None)['status']=='passed'


def test_command_failure_stdout_stderr_are_preserved(tmp_path,monkeypatch):
    def reject(*args,**kwargs):raise subprocess.CalledProcessError(1,args[0],output='partial',stderr='fixture failure')
    monkeypatch.setattr(module.subprocess,'run',reject)
    with pytest.raises(subprocess.CalledProcessError):module.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight')
    assert (tmp_path/'preflight/gpu_01.stdout').read_text()=='partial'
    assert (tmp_path/'preflight/gpu_01.stderr').read_text()=='fixture failure'


def test_deadline_cannot_be_reset_during_samples(tmp_path,monkeypatch):
    calls=setup(monkeypatch);clock=[0.]
    def sleep(seconds):clock[0]+=seconds
    def check():
        if clock[0]>=2:raise TimeoutError('CPU absolute deadline fixture')
    with pytest.raises(TimeoutError):module.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',check=check,sleep=sleep)
    assert len(calls)==1


def test_existing_preflight_directory_cannot_be_overwritten(tmp_path,monkeypatch):
    calls=setup(monkeypatch)
    with pytest.raises(FileExistsError):module.preflight_gpu('GPU-abc',evidence_dir=tmp_path,sleep=lambda _:None)
    assert not calls
