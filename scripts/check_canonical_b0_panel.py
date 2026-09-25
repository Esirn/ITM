#!/usr/bin/env python
"""Independent saved-array checks for the fixed synthetic panel (zero FK calls)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from audit_caption_intervals import artifact, write_json
from itm.data.canonical_pair import SCHEMA
from itm.data.caption_time import parse_timed_caption, plan_caption_crop
from itm.data.humanml_torch import recover_from_ric_torch
from itm.data.manifest import read_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=list(read_jsonl(a.pairs));reports=[];groups={}
    for r in rows:
        groups.setdefault(r['source_group_id'],set()).add(r['split'])
        for f in r['artifacts']+[r['source_artifact'],r['original_caption_artifact']]:
            assert artifact(f['path'])['sha256']==f['sha256'],f['path']
        root=Path(r['imu_path']).parent
        with np.load(root/'source_20hz.npz') as d: s20={k:d[k].copy() for k in d.files}
        with np.load(root/'source_30hz.npz') as d: s30={k:d[k].copy() for k in d.files}
        with np.load(root/'imu.npz') as d: cache={k:d[k].copy() for k in d.files}
        features=np.load(r['joint_vec_path']);target=np.load(r['joints_path'])
        q=np.asarray(r['source_to_canonical_rotation']);origin=np.asarray(r['source_origin'])
        assert str(cache['schema'])==SCHEMA
        np.testing.assert_allclose(q.T@q,np.eye(3),atol=1e-12,rtol=0)
        assert abs(np.linalg.det(q)-1)<1e-12
        np.testing.assert_allclose(cache['timestamps'],np.arange(len(cache['joints']))/30,atol=0,rtol=0)
        assert features.shape==(r['motion_frames'],263) and cache['sensor_mask'].all()
        assert all(np.isfinite(cache[k]).all() for k in ['acceleration','orientation','joints'])
        np.testing.assert_allclose(cache['acceleration'],(s30['acceleration'].astype(float)@q.T).astype(np.float32),atol=0,rtol=0)
        np.testing.assert_allclose(cache['orientation'],np.matmul(q,s30['orientation']).astype(np.float32),atol=0,rtol=0)
        np.testing.assert_allclose(cache['joints'],((s30['joints']-origin)@q.T).astype(np.float32),atol=0,rtol=0)
        np.testing.assert_allclose(target,((s20['joints'][:-1,:22]-origin)@q.T).astype(np.float32),atol=0,rtol=0)
        np.testing.assert_allclose(target[0,0,[0,2]],0,atol=1e-7,rtol=0)
        decoded=recover_from_ric_torch(torch.from_numpy(features)).numpy()
        np.testing.assert_allclose(decoded,target,atol=1e-5,rtol=0)
        # Both FK streams must meet at identical absolute source times every 0.1s.
        x,y=s20['joints'][::2],s30['joints'][::3];n=min(len(x),len(y))
        cross_error=float(np.max(np.abs(x[:n]-y[:n])))
        np.testing.assert_allclose(x[:n],y[:n],atol=1e-5,rtol=0)
        lines=[line for line in Path(r['text_path']).read_text().splitlines() if line.strip()]
        c=parse_timed_caption(lines[r['caption_index']]);assert c.text==r['caption'] and c.start==c.end==0
        plan=plan_caption_crop(c,len(features),cache['timestamps'])
        assert plan['frame_slice']==[0,len(features)]
        reports.append({'motion_id':r['motion_id'],'status':'passed','frames':len(features),
            'RIC_roundtrip_max_abs_m':float(np.max(np.abs(decoded-target))),
            'common_10hz_samples':n,'20hz_30hz_FK_max_abs_m':cross_error,'caption_interval':'complete_full_clip',
            'semantic_review':'not_part_of_this_numeric_check'})
    assert all(len(v)==1 for v in groups.values())
    write_json(a.output,{'pairs':artifact(a.pairs),'cases':reports,'source_groups':len(groups),
        'passed':len(reports),'code':artifact(Path(__file__)),'gpu_jobs':0,'FK_calls':0,
        'generated_IMU_closure_certified':False})
    print(json.dumps({'passed':len(reports),'max_RIC_error_m':max(r['RIC_roundtrip_max_abs_m'] for r in reports),
                      'max_cross_fps_error_m':max(r['20hz_30hz_FK_max_abs_m'] for r in reports)}))

if __name__=='__main__':main()
