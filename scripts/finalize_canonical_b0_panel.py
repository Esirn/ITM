#!/usr/bin/env python
"""Package the explicit GT review and fixed B0 proposal; never authorize a run."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from audit_caption_intervals import artifact, write_json
from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.data.manifest import read_jsonl


def rows(path, value):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    a=p.parse_args();out=a.directory
    if (out/'admission.jsonl').exists():raise FileExistsError('Already packaged')
    review_path=out/'semantic_review.json'
    review={r['motion_id']:r for r in json.loads(review_path.read_text())['cases']}
    verification=json.loads((out/'verification.json').read_text())
    assert verification['pairs']==artifact(a.pairs) and verification['passed']==12
    built=list(read_jsonl(a.pairs));assert set(review)=={r['motion_id'] for r in built}
    admitted=[];all_rows=[];sampling=[]
    for r in built:
        decision=review[r['motion_id']]
        accepted=decision['decision']=='admit'
        all_rows.append({**r,'status':'admitted_synthetic_B0_only' if accepted else decision['decision'],
            'benchmark_admissible':accepted,'training_admissible':accepted and r['split']=='train',
            'semantic_alignment_certified':accepted,'semantic_alignment_scope':'coarse_body_single_model_GT_review',
            'review':decision,'review_artifact':artifact(review_path),
            'physical_mapping_scope':'offline_synthetic_pair_only','deployable_real_IMU_calibration':False})
        if accepted:admitted.append(all_rows[-1])
    assert [r['motion_id'] for r in admitted if r['split']=='train']==['000506','004728','000027']
    assert [r['motion_id'] for r in admitted if r['split']=='val']==['000433','000382','000139']
    rows(out/'admission.jsonl',all_rows)
    shutil.copyfile(a.pairs,out/'built_pairs.jsonl')
    shutil.copyfile(a.pairs.parent/'report.json',out/'build_report.json')
    for split in ['train','val']:
        selected=[r for r in admitted if r['split']==split]
        motion=[];imu=[];plans=[]
        for r in selected:
            motion.append({k:r[k] for k in ['motion_id','split','source_group_id','text_path','joints_path','joint_vec_path']})
            motion[-1].update(num_texts=len(Path(r['text_path']).read_text().splitlines()),mapping_version=r['mapping_version'])
            imu.append({k:r[k] for k in ['motion_id','split','source_group_id','imu_path','schema','mapping_version','num_frames','fps','benchmark_admissible','training_admissible','world_alignment_scope','artifacts']})
            with np.load(r['imu_path']) as d:times=d['timestamps'].copy()
            c=parse_timed_caption([s for s in Path(r['text_path']).read_text().splitlines() if s.strip()][r['caption_index']])
            plan=plan_caption_crop(c,r['motion_frames'],times)
            plans.append({**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in plan.items()},
                'motion_id':r['motion_id'],'split':split,'caption':c.text,'caption_index':r['caption_index'],
                'sensor_configs':['head','wrists'],'status':'admitted_synthetic_B0_only',
                'training_admissible':split=='train','benchmark_admissible':True,
                'world_alignment_certified':True,'semantic_alignment_certified':True,
                'world_alignment_scope':r['world_alignment_scope'],'semantic_alignment_scope':r['semantic_alignment_scope'],
                'mapping_version':r['mapping_version'],'schema':r['schema'],'artifacts':r['artifacts'],
                'admission_artifact':artifact(out/'admission.jsonl')})
            if split=='val':
                for config in ['head','wrists']:
                    for condition in ['paired','text_only','imu_only','backbone_bypass','numeric_zero','physical_static']:
                        sampling.append({'case_id':f"{r['motion_id']}-{config}-{condition}",'motion_id':r['motion_id'],
                            'sensor_config':config,'condition':condition,'caption_index':r['caption_index'],
                            'text':r['caption'] if condition!='imu_only' else '',
                            'text_present':condition!='imu_only','imu_present':condition not in ['text_only','backbone_bypass'],
                            'length':r['motion_frames'],'sampling_seed':20260926,
                            'random_plan_case_id':f"canonical-B0-{r['motion_id']}-{config}",
                            'gt_path':r['joints_path'],'environment':'nonflat_support' if r['motion_id']=='000139' else 'approximately_stationary_floor',
                            'checkpoint_selection':'same_final_checkpoint_for_all_control_conditions',
                            'status':'planned_not_executed'})
        rows(out/f'{split}_motion.jsonl',motion);rows(out/f'{split}_imu.jsonl',imu);rows(out/f'{split}_caption_plans.jsonl',plans)
    write_json(out/'sampling_cases.json',sampling)
    budget={'version':'canonical-B0-budget-v1','authorization_status':'pending_user_budget_authorization',
        'scope':'one tiny synthetic B0 engineering pilot, no hyperparameter search or research-quality claim',
        'data':{'train_groups':3,'val_groups':3,'training_ids':['000506','004728','000027'],'validation_ids':['000433','000382','000139'],
                'manifest_artifacts':[artifact(out/f'{s}_{k}.jsonl') for s in ['train','val'] for k in ['motion','imu','caption_plans']]},
        'training':{'owner':'01','runs':1,'seed':20260925,'epochs':100,'batch_size':3,'samples_per_epoch':6,
            'optimizer_updates':200,'max_examples_in_updates':600,'learning_rate':5e-5,
            'optimizer':'AdamW existing defaults; serialize all resolved settings',
            'sensor_configs':['head','wrists'],'same_checkpoint':True,'backbone':'frozen_eval',
            'encoder':'ordered_concat','adapter':'additive','null_mode':'strict','time_encoding':'physical_sinusoidal',
            'text_dropout':.1,'imu_dropout':.1,'diffusion_loss_weight':1.,'all_other_loss_weights':0.,
            'resume':False,'IMU_normalization':'01 recompute at exact admitted 20Hz input times, unique train clips only, bind to checkpoint',
            'motion_normalization':'retain pretrained MDM Mean/Std and bind hashes; do not refit on 3 clips',
            'checkpoint_selection':'final update 200 only; no validation early-selection'},
        'sampling':{'cases':36,'K':1,'seed':20260926,'sampler':'DDPM','steps':1000,'guidance':'factorized',
            'text_scale':2.5,'imu_scale':1.,'joint_scale':1.,'dtype':'float32','eval_mode':True,
            'max_denoising_step_calls':36000,'max_branch_example_forwards':144000,
            'noise':'existing addressed noise, same original length and random_plan_case_id across six conditions',
            'static':'zero raw acceleration plus first observed sensor orientation held constant; use admitted training normalization',
            'numeric_zero':'all-zero NORMALIZED tensor with sensor presence retained',
            'imu_only':'text presence false, not merely a nonempty text replacement',
            'bypass':'raw frozen MDM path with identical text CFG/noise; strict-null comparison saved',
            'shuffled':'not_estimable: no same-action, near-length donor in the admitted fixed panel; no cross-action substitute',
            'outputs':'full feature tensors, decoded joints, input tensors, masks/timestamps, all noise hashes, config and checkpoint/data hashes, runtime/memory'},
        'resource_caps':{'concurrent_GPUs':1,'maximum_allocated_memory_GiB':24,'training_wall_hours':1,'sampling_wall_hours':3,'total_GPU_wall_hours':4,
            'device':'01 must name one available existing GPU and verify driver/free memory immediately before run; current 02 nvidia-smi cannot reach driver',
            'stop':'first cap reached, NaN/Inf, identity mismatch or OOM; retain partial/failure records, no unbudgeted retry or CPU fallback'},
        'execution_dependencies':['owner-01 explicit canonical_smpl_paired_v1 adapter and provenance binding','owner-02 verify admitted input replay after that concrete adapter exists','user approves this budget','GPU preflight succeeds'],
        'not_dependencies_for_B0':['compatible edit witnesses','generated strict IMU closure','sealed research test set'],
        'sampling_plan':artifact(out/'sampling_cases.json')}
    write_json(out/'b0_budget.json',budget)
    write_json(out/'package.json',{'version':'canonical-B0-delivery-v1','admission':artifact(out/'admission.jsonl'),
        'build':artifact(out/'build_report.json'),'verification':artifact(out/'verification.json'), 'semantic_review':artifact(review_path),
        'budget':artifact(out/'b0_budget.json'),'code':artifact(Path(__file__)),
        'admitted_train':3,'admitted_val':3,'uncertain':5,'conflicting':1,'training_or_generation_executed':False})
    print(json.dumps({'admitted_train':3,'admitted_val':3,'sampling_cases':len(sampling),'budget_authorized':False}))

if __name__=='__main__':main()
