"""Synthetic NVML only; high compute activity never becomes a free-SM gate."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('gpu,memory',[(99,28),(100,100),(0,0)])
def test_busy_gpu_and_existing_processes_are_allowed(tmp_path,monkeypatch,gpu,memory):
    m=module('itm_contended_gpu_preflight');calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='GPU-abc, 1234, python, 16000\n' if 'query-compute-apps' in argv[2]
            else f'GPU-abc, 24564, 7560, {gpu}, {memory}\n',stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    r=m.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',sleep=lambda _:None)
    assert r['status']=='passed' and len(calls)==4 and r['compute_process_count']==1
    assert len(r['samples'])==3 and r['samples'][0]['GPU_utilization_percent']==gpu
    assert (tmp_path/'preflight/compute_processes.stdout').exists()


@pytest.mark.parametrize('line,reason',[
    ('GPU-abc,24564,4095,99,28\n','free memory'),
    ('GPU-def,24564,7560,99,28\n','identity'),
    ('GPU-abc,24564,7560,101,28\n','Invalid'),
    ('GPU-abc,24564,7560,N/A,28\n','unavailable')])
def test_memory_and_identity_checks_remain(tmp_path,monkeypatch,line,reason):
    m=module('itm_contended_gpu_preflight')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=line,stderr=''))
    with pytest.raises(RuntimeError,match=reason):m.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',sleep=lambda _:None)


def test_parent_reuses_exact_scientific_worker_and_budget():
    old=module('run_mdm_prefix_mechanism_bounded');new=module('run_mdm_contended_prefix_mechanism_bounded')
    assert old.PACKAGE_SHA==new.PACKAGE_SHA and old.CAPS==new.CAPS and old.CALL_CAPS==new.CALL_CAPS
    assert old.validate_receipt.__code__.co_code==new.validate_receipt.__code__.co_code
    assert new.shared_policy()['path'].endswith('GPU_CONTENDED_EXECUTION_POLICY_20261008.json')
    assert new.LIMITS==dict(minimum_free_MiB=4096,samples=3,sample_interval_seconds=2)


def test_deadline_still_stops_between_samples(tmp_path,monkeypatch):
    m=module('itm_contended_gpu_preflight');clock=[0.];calls=[]
    def run(*a,**k):calls.append(1);return SimpleNamespace(stdout='GPU-abc,24564,7560,99,28\n',stderr='')
    monkeypatch.setattr(m.subprocess,'run',run)
    def check():
        if clock[0]>=2:raise TimeoutError('fixture absolute deadline')
    with pytest.raises(TimeoutError):m.preflight_gpu('GPU-abc',evidence_dir=tmp_path/'preflight',check=check,sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    assert len(calls)==1
