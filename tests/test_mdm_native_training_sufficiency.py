"""Native batch-path corrected fixed population; CPU synthetic model only."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import pytest

from itm.models.mdm_native_training_sufficiency import TrainingPackage,TrainingGuard,PACKAGE_SHA,CAPS,CALL_CAPS
from itm.models.mdm_correction import read
from test_mdm_sampling_alignment import script,forbid_gpu_and_models
from test_mdm_training_sufficiency import cpu_package,FixtureModel,FixtureEncoder


@pytest.fixture(scope='module')
def package():return TrainingPackage()


def test_native_plan_exact_populations_and_caps(package):
    assert sum(r['MDM_examples'] for r in package.jobs)==5184
    assert sum(r['caption_calls'] for r in package.jobs)==24
    assert sum(r['caption_examples'] for r in package.jobs)==72
    assert sum(r['encoder_calls'] for r in package.jobs)==432
    assert sum(r['auxiliary_f00'] for r in package.jobs)==576
    assert len(package.checks)==2880
    families={r['state_id']:r['state_family'] for r in package.states}
    for j in package.jobs:
        batch=5 if families[j['state_id']]=='GT_noised_Q' else 4
        assert j['input_batch_shape'][0]==batch
        assert j['main_branch_indices']==([1,2,3,4] if batch==5 else [0,1,2,3])


def test_native_worker_full_synthetic_queue(tmp_path,monkeypatch,package):
    worker=script('run_mdm_native_training_sufficiency_worker');pkg=cpu_package(package,tmp_path)
    for j,real in zip(pkg.jobs,package.jobs):j['input_batch_shape'][0]=real['input_batch_shape'][0]
    model=FixtureModel();encoder=FixtureEncoder();model.eval();encoder.eval();caption_batches=[]
    def encode(text):caption_batches.append(len(text));model.caption_calls+=1;return torch.ones(1,len(text),8)
    monkeypatch.setattr(model,'encode_text',encode)
    removed=[];diffusion=SimpleNamespace(num_timesteps=1000,_scale_timesteps=lambda t:t)
    monkeypatch.setattr(worker,'load_model',lambda *a:(model,diffusion,SimpleNamespace(adapters=model.adapter),encoder,torch.nn.Identity().eval(),SimpleNamespace(remove=lambda:removed.append(True)),SimpleNamespace()))
    g=TrainingGuard(dict(version='itm-sufficiency-context-v1',phase='diagnostic',package_sha256=PACKAGE_SHA,
        implementation_sha256='CPUfixture',deadline_monotonic=600.,receipt_path=str(tmp_path/'receipt.json'),expected_artifacts=[]),pkg,clock=lambda:0.)
    worker.diagnose(pkg,SimpleNamespace(device='cpu',output=tmp_path/'native'),g);g.finish()
    assert removed and model.calls==1152 and encoder.calls==432 and model.caption_calls==24
    assert caption_batches.count(5)==caption_batches.count(1)==12
    assert len(list((tmp_path/'native/auxiliary').glob('*.npz')))==576
    for kind,cap in CAPS.items():
        assert g.state['counts'][kind]['completed']==cap
        assert g.state['counts'][kind]['completed_calls']==CALL_CAPS[kind]
    assert all(r['status']=='completed' for r in g.state['checks'])
    assert len(list((tmp_path/'native/predictions').glob('*.npz')))==1152


def test_native_prepare_true_inputs_without_model_or_GPU(tmp_path,monkeypatch):
    parent=script('run_mdm_native_training_sufficiency_bounded');forbid_gpu_and_models(monkeypatch)
    p=parent.prepare(tmp_path/'implementation');x=read(p)
    assert x['package']['sha256']==PACKAGE_SHA and len(x['states'])==192
    assert x['plan']['MDM_examples']==5184 and x['plan']['caption_calls']==24
    assert x['model_forwards']==x['GPU_queries']==0
