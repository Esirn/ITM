"""CPU fake-clock tests; keep the scientific worker/limits unchanged."""
from pathlib import Path

import pytest

from test_contended_prefix_preflight import module
from itm.models.mdm_correction import artifact,read
from itm.models.mdm_run_limits import atomic_json


def test_budget_and_receipt_rules_change_only_wall_time():
    old=module('run_mdm_contended_prefix_mechanism_bounded');new=module('run_mdm_long_prefix_mechanism_bounded')
    assert new.ABSOLUTE_SECONDS==1200
    assert new.PACKAGE_SHA==old.PACKAGE_SHA and new.CAPS==old.CAPS and new.CALL_CAPS==old.CALL_CAPS
    assert new.validate_receipt.__code__.co_code==old.validate_receipt.__code__.co_code
    assert new.shared_policy()==old.shared_policy()
    record=read(new.budget_amendment()['path']);assert record['absolute_seconds']==1200 and record['scientific_changes']==[]


def setup(tmp_path,monkeypatch):
    m=module('run_mdm_long_prefix_mechanism_bounded');p=m.PrefixPackage();clock=[0.]
    manifest=tmp_path/'manifest.json';atomic_json(manifest,dict(package=artifact(p.path),expected_artifacts=[],environment=p.common['environment']))
    monkeypatch.setattr(m,'authorization',lambda *a:dict(implementation_sha256='CPUfixture'))
    monkeypatch.setattr(m,'environment',lambda:p.common['environment'])
    monkeypatch.setattr(m,'PrefixPackage',lambda *a:p)
    return m,p,clock,manifest


def test_parent_and_child_context_share1200_deadline_without_reset(tmp_path,monkeypatch):
    m,p,clock,manifest=setup(tmp_path,monkeypatch);captured={}
    def preflight(*a,**k):clock[0]=800.;k['check']();return dict(status='passed')
    monkeypatch.setattr(m,'preflight_gpu',preflight)
    def supervise(argv,deadline,**kwargs):
        assert deadline==1200.
        cp=Path(argv[argv.index('--run-context')+1]);ctx=read(cp);captured.update(ctx)
        assert ctx['deadline_monotonic']==1200.
        atomic_json(Path(ctx['receipt_path']),{})
        atomic_json(Path(ctx['receipt_path']).with_suffix('.journal.jsonl'),{})
        return dict(status='completed')
    monkeypatch.setattr(m,'supervise',supervise)
    monkeypatch.setattr(m,'read_receipt',lambda *a:dict(**m.ledger(p),counts={},peak_allocated_bytes=0))
    monkeypatch.setattr(m,'validate_receipt',lambda *a,**k:None)
    s=m.execute(manifest,tmp_path/'run',tmp_path/'record','GPU-CPU-longbudget-fixture',clock=lambda:clock[0])
    assert s['status']=='completed' and s['started_monotonic']==0. and s['deadline_monotonic']==1200.
    assert captured['deadline_monotonic']==s['deadline_monotonic']


def test_parent_stops_at1200_before_worker_and_keeps_full_denominator(tmp_path,monkeypatch):
    m,p,clock,manifest=setup(tmp_path,monkeypatch)
    def preflight(*a,**k):clock[0]=1200.;k['check']()
    monkeypatch.setattr(m,'preflight_gpu',preflight)
    monkeypatch.setattr(m,'supervise',lambda *a,**k:pytest.fail('Worker must not start'))
    with pytest.raises(TimeoutError,match='1200'):m.execute(manifest,tmp_path/'run',tmp_path/'record','GPU-CPU-longbudget-fixture',clock=lambda:clock[0])
    s=read(tmp_path/'run/runtime/status.json')
    assert len(s['states'])==180 and len(s['probes'])==360 and len(s['checks'])==7380
    assert all(r['status']=='not_run_after_stop' for r in s['probes'])


def test_real_prepare_contains_effective_budget_and_corrected_evaluator(tmp_path,monkeypatch):
    from test_mdm_sampling_alignment import forbid_gpu_and_models
    forbid_gpu_and_models(monkeypatch);m=module('run_mdm_long_prefix_mechanism_bounded')
    path=m.prepare(tmp_path/'implementation');x=read(path)
    assert x['effective_absolute_seconds']==1200 and x['resource_budget']==m.budget_amendment()
    assert x['model_constructions']==x['GPU_queries']==0 and x['planned_worker_artifacts']==8292
    assert any(e['path'].endswith('/evaluate_b0_prefix_mechanism_v2.py') for e in x['expected_artifacts'])
