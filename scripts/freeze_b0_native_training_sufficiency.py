#!/usr/bin/env python
"""Native numerical-path amendment, same states/weights/metrics and bounds."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'docs/itm_rework/protocol/b0_training_sufficiency_20261008_v2'
REL='docs/itm_rework/protocol/b0_training_sufficiency_20261008_v3'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def art(p):return dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=sha(p))

def build():
    assert sha(OLD/'package.json')=='471376c4799b80975c09faee4a80e2adfa220a6119a73a325761ed2a89029593'
    files={Path(a['path']).stem:read(ROOT/a['path']) for a in read(OLD/'package.json')['artifacts']}
    states={r['state_id']:r for r in files['states']};seen_text=set();seen_control=set()
    for job in files['jobs']:
        native=states[job['state_id']]['state_family']=='GT_noised_Q';batch=5 if native else 4;text_batch=5 if native else 1
        key=(job['cache_id'],text_batch);control=(job['snapshot_id'],job['cache_id'])
        job.update(input_batch_shape=[batch,263,1,states[job['state_id']]['frames']],scaled_t=[states[job['state_id']]['timestep']]*batch,
                   MDM_examples=batch,caption_batch=text_batch,caption_calls=int(key not in seen_text),
                   caption_examples=text_batch if key not in seen_text else 0,encoder_calls=0 if control in seen_control else 6,
                   main_branch_indices=[1,2,3,4] if native else [0,1,2,3],auxiliary_f00=native)
        seen_text.add(key);seen_control.add(control)
    assert sum(j['MDM_examples'] for j in files['jobs'])==5184
    assert sum(j['caption_calls'] for j in files['jobs'])==24 and sum(j['caption_examples'] for j in files['jobs'])==72
    assert sum(j['encoder_calls'] for j in files['jobs'])==432
    files['design'].update(supersedes=art(OLD/'package.json'),
        revision='Controlled native-path probe showed old Q caption batch5 differs from batch1; restore full native Q MDMbatch5 + CLIPbatch5, trajectories retain batch4 + CLIPbatch1. No tolerance relaxation.',
        cache='Two native caption caches per case (batch5 Q/batch1 traces); 24 calls/72 examples. E controls remain checkpoint/case-specific,6examples per case/snapshot,432total.',
        inference='Original native Q branch order f00/T/P/S/R in MDMbatch5; f00 auxiliary compatibility output, excluded from4608 main predictions/contrasts. Saved U/F trajectories use T/P/S/R batch4. Float32 denoiser/control and native half CLIP.',
        software_bounds='Same atol=rtol=1e-6; same2880 checks, same state bytes, checkpoints, frames and metric rules.')
    files['evaluation'].update(branch_rows=4608,auxiliary_f00_predictions=576,
        amendment='Main T/P/S/R and4608 within/4608 checkpoint contrasts unchanged. Extra f00 only restores old numerical batch path, no semantic/new-motion score.')
    files['budget'].update(MDM_examples=5184,caption_batch1_calls=12,caption_batch5_calls=12,caption_calls=24,caption_examples=72,
                           auxiliary_f00_predictions=576,status='frozen_native_path_amendment',shared_GPU_policy=art(ROOT/'docs/itm_rework/GPU_SHARED_EXECUTION_POLICY_20261008.json'))
    files['validation'].update(MDM_examples=5184,caption_calls=24,caption_examples=72,auxiliary_f00_predictions=576,
                              amendment_model_forwards=0,amendment_GPU_queries=0)
    sources=files['sources']
    proof=ROOT/'outputs/itm_rework/method/caption_path_authorized_20261008_v1'
    for p in (OLD/'package.json',proof/'runtime/status.json',proof/'runtime/receipt.json',proof/'probe/result.json',proof/'probe/captions.npz',*[proof/'probe'/(a+'.npz') for a in 'ABCD'],Path(__file__)):
        sources.append(art(p))
    # Old evidence's mutable runtime source paths are preserved by its archived
    # code; current new implementation is bound separately after this freeze.
    for a in sources:assert sha(ROOT/a['path'])==a['sha256'],a['path']
    out=ROOT/REL;out.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():(out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'package.json').write_text(json.dumps(dict(version='itm-training-sufficiency-native-package-v1',
        artifacts=[dict(path=REL+'/'+p.name,sha256=sha(p)) for p in sorted(out.iterdir())],
        scientific_change='No new states/weights/noise/metrics/tolerances. Explicit native batch-path/counter correction based on controlled probe; no model efficacy claim.'),indent=2,ensure_ascii=False)+'\n')
    return sha(out/'package.json')

if __name__=='__main__':print(build())
