"""CPU inspection only: saved weights/history/input/output arrays; no model forward."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'outputs/itm_rework/method/canonical_b0_authorized_20260926_v1'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def rms(x): return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))

def main():
    torch.set_num_threads(2)
    state=torch.load(RUN/'model.pt',map_location='cpu',weights_only=True)
    first=torch.load(RUN/'model_epoch001.pt',map_location='cpu',weights_only=True)
    history=state['history']
    captions=[json.loads(x) for x in (ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3/train_caption_plans.jsonl').read_text().splitlines()]
    layers=[]
    for key, weight in state['adapters'].items():
        layers.append(dict(name=key,shape=list(weight.shape),l2=float(weight.norm()),max_abs=float(weight.abs().max()),
                           epoch1_to_final_l2=float((weight-first['adapters'][key]).norm())))
    params={group:sum(t.numel() for t in state[group].values()) for group in ('imu_encoder','adapters')}
    counts=history[-1]['condition_counts_cumulative']
    loss={}
    for label, rows in [('first10',history[:10]),('last10',history[-10:]),('all100',history)]:
        v=np.array([r['diffusion_loss'] for r in rows]); loss[label]=dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max()),std=float(v.std()))
    population=[int(c['frames']) for c in captions for _ in range(2)]
    batches=list(itertools.combinations(population,3))
    padded=[1-sum(b)/(3*max(b)) for b in batches]
    examples=[]
    for mid in ('000433','000382','000139'):
        for config in ('head','wrists'):
            with np.load(RUN/f'samples/{mid}-{config}-paired.npz',allow_pickle=False) as p, np.load(RUN/f'samples/{mid}-{config}-physical_static.npz',allow_pickle=False) as s:
                mask=p['sensor_mask'][0].astype(bool)
                x=p['normalized_imu'][0][:,mask,:]; z=s['normalized_imu'][0][:,mask,:]
                examples.append(dict(motion_id=mid,config=config,frames=len(x),
                    active_normalized_input_rms=rms(x),static_normalized_input_rms=rms(z),
                    paired_minus_static_input_rms=rms(x-z),
                    acceleration_difference_rms=rms(x[...,:3]-z[...,:3]),
                    orientation_difference_rms=rms(x[...,3:]-z[...,3:]),
                    paired_static_output_distance_m=float(np.linalg.norm(p['motion']-s['motion'],axis=-1).mean())))
    args=json.loads((ROOT/'outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json').read_text())
    report=dict(read_commit='dec5e93',scope='CPU saved-artifact inspection; zero model forwards, updates, sampling, GPU calls',
        checkpoint_sha256=sha(RUN/'model.pt'),train_captions=[{k:c[k] for k in ('motion_id','frames','caption')} for c in captions],
        unique_train_frames=sum(c['frames'] for c in captions),trainable_parameter_counts=params,
        condition_counts=counts,imu_present_examples=counts['01']+counts['11'],
        loss_summary=loss,loss_caveat='Random timestep/noise/dropout each batch; no matched-probe loss and no generalization diagnosis from these means.',
        adapters=layers,encoder_epoch1_to_final_l2=float(sum((v-first['imu_encoder'][k]).double().square().sum() for k,v in state['imu_encoder'].items()).sqrt()),
        mdm_mask_frames_in_args=args.get('mask_frames','absent; model default false'),
        all_possible_batches_mixed_length=all(len(set(b))>1 for b in batches),
        batch_combinations=len(batches),expected_padding_fraction=float(np.mean(padded)),
        padding_note='Combinatorial expectation under uniform shuffle, not reconstruction of actual batches; loss masks invalid frames, frozen MDM attention does not.',
        input_static_comparisons=examples)
    (OUT/'existing_artifact_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='adapters'},indent=2))
    paths=[RUN/'model.pt',RUN/'model_epoch001.pt',ROOT/'scripts/train_mdm_imu_control.py',ROOT/'src/itm/models/mdm_imu_control.py',ROOT/'outputs/mdm/runtime_root/model/mdm.py',ROOT/'docs/itm_rework/protocol/B0_EVALUATION_20260926.md']
    paths += sorted((RUN/'samples').glob('*-paired.npz'))+sorted((RUN/'samples').glob('*-physical_static.npz'))
    paths += [ROOT/'docs/itm_rework/protocol/canonical_b0_20260925_v3/train_caption_plans.jsonl',ROOT/'outputs/mdm/checkpoints_extracted/humanml_trans_enc_512/args.json']
    (OUT/'sources.json').write_text(json.dumps([dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in paths],indent=2)+'\n')

if __name__=='__main__': main()
