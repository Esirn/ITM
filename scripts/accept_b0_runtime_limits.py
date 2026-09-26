#!/usr/bin/env python
"""Owner-02 CPU-only runtime audit; mocked supervisor jobs are never B0 results."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import torch
from audit_caption_intervals import artifact, write_json
import run_mdm_b0_bounded as runner
from itm.models.mdm_run_limits import RunGuard, RunLimitExceeded, atomic_json, atomic_artifact, sha256
from itm.models.mdm_sampling import sample_addressed_ddpm

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'docs/itm_rework/protocol/canonical_method_acceptance_20260925_v1/execution_plan/execution_plan.json'


def supervisor_fixture(directory, scenario):
    """Run the real top-level control flow; stub authorization, GPU, and ALL child jobs."""
    now=[10000.]; contexts=[]; launched=[]
    gpu='GPU-CPU-FIXTURE-NOT-A-DEVICE'
    def child(argv, *, deadline, **kwargs):
        context=json.loads(Path(runner.flag(argv,'--run-context')).read_text())
        contexts.append(context);launched.append(context['phase'])
        output=Path(runner.flag(argv,'--output'))
        sample_index=sum(p=='sample' for p in launched)-1
        if context['phase']=='sample' and scenario=='sample_timeout' and sample_index==1:
            now[0]=deadline+.01
            return {'status':'failed','reason':'wall_timeout_process_group_killed','launched':True,'fixture_only':True}
        receipt={'phase':context['phase'],'status':'completed','updates_attempted':0,'updates_completed':0,
                 'denoising_calls':0,'branch_examples':0,'artifacts':[],'fixture_only':True}
        if context['phase']=='train':
            count=199 if scenario=='partial_training' else 200
            receipt.update(updates_attempted=count,updates_completed=count)
            output.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'fixture_only':True,'execution':{'final_update_200':count==200,'updates_completed':count}},output)
            receipt['artifacts']=[artifact(output.resolve())]
            now[0]+=1200.
        else:
            receipt.update(denoising_calls=1000,branch_examples=4000)
            # Deliberately NOT a valid motion NPZ: this tests publication/hash gates only.
            atomic_artifact(output,lambda f:f.write(b'CPU fixture, not generated motion'))
            atomic_artifact(output.with_suffix('.json'),lambda f:f.write(b'{"fixture_only":true}'))
            receipt['artifacts']=[artifact(output.resolve()),artifact(output.with_suffix('.json').resolve())]
            now[0]+=100.
        atomic_json(context['receipt_path'],receipt)
        return {'status':'completed','reason':None,'launched':True,'returncode':0,'fixture_only':True}
    original_validate=runner.validate_final_checkpoint
    def validate(path,receipt):
        original_validate(path,receipt)
        now[0]+=17. # Parent-side validation time must reduce total remaining time.
    argv=['bounded-fixture','--plan',str(PLAN),'--run-dir',str(directory),'--execute',
          '--authorization-record','NO-AUTHORIZATION-FILE-IS-CREATED','--gpu-uuid',gpu]
    error=None
    with patch.object(sys,'argv',argv),patch.object(runner,'authorization',return_value={
            'authorized':False,'fixture_only':True,'note':'authorization function stubbed for CPU control-flow test; no user approval'}),\
         patch.object(runner,'preflight_gpu',return_value={'fixture_only':True,'gpu_preflight_executed':False}),\
         patch.object(runner,'supervise',side_effect=child),patch.object(runner.time,'monotonic',side_effect=lambda:now[0]),\
         patch.object(runner,'validate_final_checkpoint',side_effect=validate):
        try: result=runner.main()
        except ValueError as exc: result=None;error=str(exc)
    status=json.loads((directory/'runtime/status.json').read_text())
    assert contexts[0]['deadline_monotonic']==13600.
    assert status['total_deadline_monotonic']==24400.
    sample_contexts=[c for c in contexts if c['phase']=='sample']
    if sample_contexts:
        assert {c['deadline_monotonic'] for c in sample_contexts}=={22017.}
        final=str((directory/'model.pt').resolve())
        assert all(any(a['path']==final and a['sha256']==sha256(final) for a in c['expected_artifacts']) for c in sample_contexts)
    if scenario=='success':
        assert result==0 and len(contexts)==37 and status['status']=='completed'
        assert all(s['status']=='completed' for s in status['slots'])
        assert (status['denoising_calls'],status['branch_examples'])==(36000,144000)
    elif scenario=='partial_training':
        assert error and '200' in error and len(contexts)==1
        assert all(s['status']=='not_run_after_stop' for s in status['slots'])
    else:
        assert result==1 and len(contexts)==3 and status['status']=='failed'
        assert [s['status'] for s in status['slots'][:2]]==['completed','failed']
        assert all(s['status']=='not_run_after_stop' for s in status['slots'][2:])
        assert Path(status['slots'][0]['output']).read_bytes()==b'CPU fixture, not generated motion'
        assert status['denoising_calls']==1000 and status['branch_examples']==4000
    return {'scenario':scenario,'fixture_only':True,'real_children_launched':0,
            'simulated_train_jobs':1,'simulated_sample_jobs':len(sample_contexts),
            'status':status,'contexts':contexts,'exception':error}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fixtures=args.fixture_dir.resolve();fixtures.mkdir(parents=True,exist_ok=False)
    write_json(out/'budget.json',{'CPU_supervisor_scenarios':3,'tiny_CPU_DDPM_calls':1000,
        'scalar_GPU_calls':0,'real_model_forwards':0,'B0_training_updates':0,'B0_samples':0,
        'gpu_preflight':False,'user_authorization_record_created':False})
    scenarios=[]
    for scenario in ['success','partial_training','sample_timeout']:
        r=supervisor_fixture(fixtures/scenario,scenario)
        write_json(out/(scenario+'_CPU_FIXTURE.json'),r)
        scenarios.append({'scenario':scenario,'status':'passed','artifact':artifact(out/(scenario+'_CPU_FIXTURE.json'))})
    # Exercise the actual sampler's successful 1000-call boundary on a 2x1x3 tensor.
    class Tiny(torch.nn.Module):njoints,nfeats=2,1
    class Diffusion:
        num_timesteps=1000
        calls=0
        def p_mean_variance(self,model,x,t,**kwargs):
            self.calls+=1
            return {'mean':x*.5,'log_variance':torch.zeros_like(x)-10}
    diffusion=Diffusion()
    g=RunGuard({'phase':'sample','deadline_monotonic':time.monotonic()+30,
                'receipt_path':str(fixtures/'tiny_sampler.receipt.json'),'max_denoising_calls':1000,'max_branch_examples':4000})
    value,plans=sample_addressed_ddpm(Tiny().eval(),diffusion,y={},lengths=torch.tensor([3]),case_ids=['CPU-FIXTURE'],
                                    sample_seed=11,backbone_id='NOT-A-BACKBONE',run_guard=g)
    assert diffusion.calls==g.state['denoising_calls']==1000 and g.state['branch_examples']==4000
    assert torch.isfinite(value).all() and len(plans[0]['noise_draws'])==1000
    refused=False
    try:g.before_denoise('CPU-FIXTURE',0,4)
    except RunLimitExceeded:refused=True
    assert refused and g.state['denoising_calls']==1000
    g.finish()
    # Full sample receipt must still have both published artifacts and correct hashes.
    output=fixtures/'tiny_fixture.npz'
    atomic_artifact(output,lambda f:f.write(b'CPU receipt-validation fixture'))
    short={'status':'completed','denoising_calls':999,'branch_examples':3996,'artifacts':[artifact(output)]}
    try:runner.validate_sample_receipt(short,output)
    except ValueError:short_rejected=True
    else:short_rejected=False
    assert short_rejected
    # Real preparation report from the separate file-only CLI, normalize ONLY run-root paths for comparison.
    prepared=ROOT/'outputs/itm_rework/protocol/b0_runtime_prepare_20260926_v1/runtime/status.json'
    previous=ROOT/'docs/itm_rework/method/b0_limits_preparation_20260926_v2.json'
    state=json.loads(prepared.read_text());old=json.loads(previous.read_text())
    old_root=str(ROOT/'outputs/itm_rework/method/b0_limits_prepare_20260926_v2')
    new_root=str(prepared.parent.parent)
    assert json.loads(json.dumps(state).replace(new_root,old_root))==old
    assert state['authorization'] is None and all(s['status']=='planned' for s in state['slots'])
    shutil.copyfile(prepared,out/'preparation_status.json')
    write_json(out/'acceptance.json',{'read_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'scope':'CPU runtime control-flow and software fault acceptance only','scenarios':scenarios,
        'tiny_sampler':{'calls':1000,'branch_budget_count':4000,'noise_draws':1000,'call_1001_refused':True,
                        'short_completed_receipt_rejected':True,'real_model':False},
        'preparation':{'matches_01_after_only_run_path_substitution':True,'slots_planned':36,'authorization':None,
                       'current':artifact(out/'preparation_status.json'),'previous':artifact(previous)},
        'code':artifact(Path(__file__)),
        'real_B0':{'training_updates':0,'samples':0,'GPU_jobs':0,'GPU_preflight':False},
        'limitations':['Top-level execution fixtures stub authorization/GPU/subprocess jobs; they are not approvals or generation results.',
                       'GPU allocator cap verified only by software mocks; actual CUDA driver/device preflight remains required.',
                       'No user budget authorization has been received or created.']})
    print('3 supervisor scenarios and 1000 tiny CPU DDPM calls passed; preparation matches; no real B0/GPU work')

if __name__=='__main__':main()
