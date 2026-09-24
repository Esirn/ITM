#!/usr/bin/env python3
"""Nine-clip clock/representation audit. Oracle yaw is diagnostic only, never input preprocessing."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_itm_rework import artifact,dump
from itm.data.humanml_torch import recover_from_ric_torch
from itm.data.manifest import read_jsonl
from itm.data.physical_time import resample_positions_at_times


def oracle_yaw_residual(source,target):
    # Full-motion GT is deliberately used only to diagnose a frame discrepancy.
    # Translation and initial velocity observability are not evaluated here.
    a=source-source[:,:1];b=target-target[:,:1]
    a2=a[...,[0,2]].reshape(-1,2);b2=b[...,[0,2]].reshape(-1,2)
    u,_,vt=np.linalg.svd(a2.T@b2)
    fix=np.eye(2);fix[-1,-1]=np.linalg.det(u@vt)
    planar=u@fix@vt
    aligned=a.copy();aligned[...,[0,2]]=a[...,[0,2]]@planar
    errors=np.linalg.norm(aligned-b,axis=-1)
    return {"root_relative_raw_mean_m":float(np.linalg.norm(a-b,axis=-1).mean()),
        "oracle_constant_yaw_mean_m":float(errors.mean()),"oracle_constant_yaw_p95_m":float(np.quantile(errors,.95)),
        "head_oracle_mean_m":float(errors[:,15].mean()),"wrists_oracle_mean_m":float(errors[:,[20,21]].mean()),
        "oracle_yaw_rad":float(np.arctan2(planar[1,0],planar[0,0])),
        "fit_uses_complete_GT":True,"permitted_as_deployable_input_transform":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel",type=Path,default=Path("docs/itm_rework/protocol/validation_20260924_v1/panel.jsonl"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(2)
    records={}; caches={}
    for split in ("train","val"):
        records.update({r["motion_id"]:r for r in read_jsonl(f"outputs/manifests_full/{split}.jsonl")})
        caches.update({r["motion_id"]:r for r in read_jsonl(f"outputs/manifests_full/{split}_standard_imu.jsonl")})
    intervals=list(read_jsonl("docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl"))
    group_splits={}
    for r in intervals:group_splits.setdefault(r["source_group_id"],set()).add(r["split"])
    selected=[];seen=set()
    for r in sorted(intervals,key=lambda r:r["motion_id"]):
        if r["split"]=="train" and group_splits[r["source_group_id"]]=={"train"} and r["source_group_id"] not in seen:
            selected.append(r["motion_id"]);seen.add(r["source_group_id"])
            if len(selected)==3:break
    selected += [r["motion_id"] for r in read_jsonl(args.panel)]
    results=[]
    for mid in selected:
        motion=np.load(records[mid]["joint_vec_path"])
        saved=np.load(records[mid]["joints_path"])
        decoded=recover_from_ric_torch(torch.from_numpy(motion)[None])[0].numpy()
        decoded_match=decoded.shape==saved.shape
        decode_diff=float(np.max(np.abs(decoded-saved))) if decoded_match else None
        with np.load(caches[mid]["imu_path"],allow_pickle=False) as f:
            joints=f["joints"][:,:22].astype(np.float64);fps=float(f["fps"])
        source_times=np.arange(len(joints))/fps
        nominal_count=int(np.floor(source_times[-1]*20))+1
        row={"motion_id":mid,"split":records[mid]["split"],"humanml_frames":len(motion),"saved_joint_frames":len(saved),
            "source_30fps_frames":len(joints),"source_supported_20fps_count":nominal_count,
            "source_minus_feature_frames":nominal_count-len(motion),"humanml_decode_max_abs_m":decode_diff,
            "feature_last_time_seconds":(len(motion)-1)/20,"cache_last_time_seconds":float(source_times[-1]),
            "artifacts":[artifact(records[mid]["joint_vec_path"]),artifact(records[mid]["joints_path"]),artifact(caches[mid]["imu_path"])],
            "lag_diagnostics":[]}
        for shift in (-1,0,1):
            requested=(np.arange(len(motion))+shift)/20
            valid=(requested>=source_times[0])&(requested<=source_times[-1])
            times=requested[valid]
            if len(times)<3:continue
            source=resample_positions_at_times(joints,source_times,times)
            row["lag_diagnostics"].append({"source_query_shift_frames_20fps":shift,"compared_frames":int(valid.sum()),
                "excluded_endpoint_frames":int((~valid).sum()),**oracle_yaw_residual(source,decoded[valid])})
        results.append(row)
    dump(args.output/"alignment.json",{"kind":"diagnostic_not_alignment_certification", "cases":results,
        "budget":{"device":"cpu","clips":len(results),"candidate_lags":[-1,0,1],"model_forwards":0},
        "interpretation":["HumanML velocity features drop a terminal frame; T-1 alone does not prove a one-frame start offset",
            "saved new_joints versus RIC decode is independently checked before source comparison",
            "oracle constant yaw uses full GT and is forbidden in deployable head/wrists input processing",
            "HumanML uniform skeleton retargeting changes body scale/offsets; fixed rotation alone may not suffice",
            "no lag or yaw from this report is applied to training or test inputs"]})
    print(json.dumps({"output":str(args.output),"clips":len(results),"source_minus_feature_frames":[r["source_minus_feature_frames"] for r in results],
                      "decode_max_abs_m":[r["humanml_decode_max_abs_m"] for r in results]}))


if __name__=="__main__":main()
