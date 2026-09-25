#!/usr/bin/env python
"""Prepare a non-executing B0 queue; commands remain blocked pending caps/authorization."""
import argparse
import json
from pathlib import Path
import sys
from audit_caption_intervals import artifact, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    data=root/'docs/itm_rework/protocol/canonical_b0_20260925_v3'
    cases=json.loads((data/'sampling_cases.json').read_text())
    budget=json.loads((data/'b0_budget.json').read_text())
    assert len(cases)==36 and len({r['case_id'] for r in cases})==36
    reports=root/'docs/itm_rework/method/canonical_training_input_20260925_v1.json'
    training=json.loads(reports.read_text())['training_args']
    run=root/'outputs/itm_rework/method/canonical_b0_authorized_pilot_v1'
    # This is a path proposal; no output run directory or model is created here.
    assert not run.exists(), 'Choose a fresh proposed run identity'
    train=[sys.executable,str(root/'scripts/train_mdm_imu_control.py')]
    for key in ['canonical_package','manifest','standard_imu_manifest','caption_plan','contract_version',
                'mdm_root','mdm_checkpoint','mdm_args','mean','std']:
        value=training[key]
        if key!='contract_version':value=str((root/value).resolve())
        train += ['--'+key.replace('_','-'),value]
    train += ['--output',str(run/'model.pt'),'--epochs','100','--batch-size','3','--seed','20260925',
              '--learning-rate','5e-5','--sensor-configs','head,wrists','--encoder-variant','ordered_concat',
              '--adapter-variant','additive','--null-mode','strict','--backbone-mode','frozen_eval',
              '--time-encoding','physical_sinusoidal','--text-drop-probability','0.1','--imu-drop-probability','0.1',
              '--diffusion-loss-weight','1','--device','<AUTHORIZED_GPU>']
    for flag in ['trajectory','velocity','jerk','upper-body','upper-body-velocity','text-anchor','upper-body-text-anchor','response']:
        train += ['--'+flag+'-loss-weight','0']
    train += ['--response-proxy-weight','0']
    specs=out/'case_specs';specs.mkdir()
    queue=[]
    for order,case in enumerate(cases):
        path=specs/(case['case_id']+'.json');write_json(path,[case])
        argv=[sys.executable,str(root/'scripts/sample_mdm_imu_control.py'),
            '--canonical-package',str(data/'package.json'),'--control-checkpoint',str(run/'model.pt'),
            '--mdm-args',str((root/training['mdm_args']).resolve()),'--mdm-root',str((root/training['mdm_root']).resolve()),
            '--standard-imu-manifest',str(data/'val_imu.jsonl'),'--motion-manifest',str(data/'val_motion.jsonl'),
            '--caption-plan',str(data/'val_caption_plans.jsonl'),'--spec',str(path.resolve()),
            '--output',str(run/'samples'/(case['case_id']+'.npz')),'--seed','20260926',
            '--text-scale','2.5','--imu-scale','1','--joint-scale','1','--guidance-mode','factorized',
            '--branch-execution','batched','--random-plan','addressed','--time-policy','crop_seconds',
            '--null-mode','strict','--device','<AUTHORIZED_GPU>',
            '--mean',str((root/training['mean']).resolve()),'--std',str((root/training['std']).resolve())]
        queue.append({'order':order,'case_id':case['case_id'],'spec':artifact(path),'argv_template':argv,
            'steps':1000,'maximum_branch_example_forwards':4000,'random_plan_case_id':case['random_plan_case_id'],
            'status':'planned_not_launched','outputs_on_failure':'retain all earlier case files and failed slot reason; no retry'})
    write_json(out/'execution_plan.json',{'version':'canonical-B0-execution-plan-v1','execute':False,
        'budget':artifact(data/'b0_budget.json'),'sampling_plan':artifact(data/'sampling_cases.json'),
        'training_input_report':artifact(reports),'code':artifact(Path(__file__)),
        'authorization_status':'not_authorized','runtime_limits_status':'not_implemented_or_accepted',
        'device_status':'not_selected_or_preflighted','run_directory_proposal':str(run),
        'train_argv_template':train,'queue':queue,
        'required_limit_semantics':{
            'training':'monotonic deadline 3600 seconds from process start including load/save; hard optimizer update cap 200; finite loss/gradients before step',
            'sampling':'one absolute 10800-second deadline shared by all 36 subprocesses, including each load/save; at most 36000 denoising calls and 144000 branch-example forwards',
            'total':'single GPU, absolute active-run cap 14400 seconds; no retries or CPU fallback',
            'memory':'maximum 24 GiB allocated; enforce before model allocation and retain peak/error records; minimum-free check alone is insufficient',
            'partial_artifacts':'write case status before launch and atomic completion status after saved outputs; each case is a separate invocation to preserve completed cases on a later failure',
            'checkpoint':'only completed final update 200 qualifies for planned sampling; partial epoch checkpoints are retained but never silently promoted',
            'denominator':'36 fixed slots; remaining slots after stop remain unexecuted_due_to_budget_or_failure; no slot replacement'},
        'gate_order':['new input replay accepted (owner02)','runtime cap implementation/CPU fault-injection tests accepted',
                      'explicit user budget authorization recorded against budget hash','GPU device/free-memory preflight',
                      'launch exact train queue once; then sample same final checkpoint'],
        'limitations':['Commands are reviewable templates, not an executable approved runner.',
                       'Limit flags/hooks are intentionally not invented; owner01 must implement method-local limits and owner02 coordinate a shared deadline before launch.',
                       'No invocation of these commands has occurred.']})
    print('Prepared one train template and 36 one-case sampling specs; no job launched')

if __name__=='__main__':main()
