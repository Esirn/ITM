#!/usr/bin/env python3
"""Three train-only kinematic candidates. Always uncertain; never enables B training."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_itm_rework import artifact,dump
from itm.data.dataset import read_caption_records
from itm.data.manifest import read_jsonl
from itm.metrics.arm_attributes import arm_spread_m,shoulder_outward_angle_deg,wrist_displacement_bound_m


def rotate_arm_candidates(joints,angle_degrees=10.):
    """Rotate both arm chains about the shoulder in the current torso frame.

    The sin² envelope is zero at the endpoints. This is a deterministic candidate
    construction, not a claim that any resulting edit is natural/compatible.
    """
    x=np.asarray(joints,dtype=np.float64);out=x.copy()
    left=x[:,16]-x[:,17];left/=np.linalg.norm(left,axis=-1,keepdims=True)
    down=(x[:,1]+x[:,2]-x[:,16]-x[:,17])/2
    down-=np.sum(down*left,-1,keepdims=True)*left;down/=np.linalg.norm(down,axis=-1,keepdims=True)
    axis=np.cross(down,left)
    if not np.isfinite(axis).all():raise ValueError("degenerate torso frame")
    angle=np.deg2rad(angle_degrees)*np.sin(np.linspace(0,np.pi,len(x)))**2
    for shoulder,chain,sign in ((16,[18,20],1),(17,[19,21],-1)):
        delta=x[:,chain]-x[:,shoulder,None]
        k=axis[:,None];theta=(sign*angle)[:,None,None]
        rotated=delta*np.cos(theta)+np.cross(k,delta)*np.sin(theta)+k*np.sum(k*delta,-1,keepdims=True)*(1-np.cos(theta))
        out[:,chain]=rotated+x[:,shoulder,None]
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment",type=Path,default=Path("docs/itm_rework/protocol/alignment_20260924_v1/alignment.json"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audited=json.loads(args.alignment.read_text())
    selected=[r["motion_id"] for r in audited["cases"] if r["split"]=="train"]
    if len(selected)!=3:raise ValueError("expected the predeclared three train-only diagnostic sources")
    motions={r["motion_id"]:r for r in read_jsonl("outputs/manifests_full/train.jsonl")}
    intervals=list(read_jsonl("docs/itm_rework/protocol/source_audit_20260924_v1/source_intervals.jsonl"))
    source={r["motion_id"]:r for r in intervals}; group_splits={}
    for r in intervals:group_splits.setdefault(r["source_group_id"],set()).add(r["split"])
    cases=[];failures=[]
    for mid in selected:
        if group_splits[source[mid]["source_group_id"]]!={"train"}:raise ValueError("train source group overlaps another split")
        original=np.load(motions[mid]["joints_path"]).astype(np.float64)
        captions=read_caption_records(motions[mid]["text_path"])
        caption=next((c.caption for c in captions if c.start==0 and c.end==0 and c.caption),None)
        try:
            if caption is None:raise ValueError("no full-clip base caption")
            edited=rotate_arm_candidates(original)
            base_spread,edit_spread=arm_spread_m(original),arm_spread_m(edited)
            base_angle,edit_angle=shoulder_outward_angle_deg(original),shoulder_outward_angle_deg(edited)
            bone_errors=[]
            for a,b in ((16,18),(18,20),(17,19),(19,21)):
                before=np.linalg.norm(original[:,a]-original[:,b],axis=-1)
                after=np.linalg.norm(edited[:,a]-edited[:,b],axis=-1)
                bone_errors.append(float(np.max(np.abs(before-after))))
            name=args.output/f"train_{mid}_head_arm_spread.npz"
            np.savez_compressed(name,base_joints=original,edited_joints=edited,timestamps_seconds=np.arange(len(original))/20,
                representation=np.array("humanml22_renderable_joints_not_full_HumanML263"))
            checks={"finite":bool(np.isfinite(edited).all()),"arm_bone_length_max_abs_change_m":max(bone_errors),
                "endpoint_max_abs_change_m":float(np.max(np.abs(edited[[0,-1]]-original[[0,-1]]))),
                "max_applied_shoulder_rotation_deg":10.,"shoulder_outward_angle_base_deg":base_angle,
                "shoulder_outward_angle_edit_deg":edit_angle,"arm_spread_base_m":base_spread,"arm_spread_edit_m":edit_spread,
                "requested_direction_observed":edit_spread>base_spread,"naturalness_review":"pending",
                "collision_and_joint_limit_certification":"not_performed"}
            cases.append({"case_id":f"candidate-train-{mid}-head-spread","source_group_id":source[mid]["source_group_id"],
                "source_motion_id":mid,"sensor_config":"head","base_caption_id":f"{mid}-full-clip",
                "base_caption_text":caption,"edit_caption_id":f"{mid}-arm-spread-candidate",
                "edit_caption_text":caption.rstrip(".")+", with the arms held farther apart.","edited_attribute":"arm_spread_m",
                "compatibility_label":"uncertain","evidence_type":"analytic_joint_proxy_candidate_only",
                "witness_motion_path":str(name.resolve()),"witness_artifact":artifact(name),
                "witness_observation_residuals":{"head_joint_max_abs_m":float(np.max(np.abs(edited[:,15]-original[:,15]))),
                    "wrist_joint_position_displacement_bound_m":wrist_displacement_bound_m(original,edited),
                    "sensor_acceleration":None,"sensor_orientation":None},"quality_checks":checks,
                "tolerances":{"geometric_float64_regression_atol":1e-10,"application_sensor_tolerance":None},
                "text_information_role":"candidate_additional_attribute_not_certified_nonredundant",
                "timestamps":{"fps":20.,"frames":len(original),"clock":"HumanML_clip_local_seconds"},
                "calibration_id":"none_joint_proxy_only","split_role":"train","construction_version":"arm-spread-candidate-v1",
                "reviewer_or_rule_version":"analytic-invariants-v1-naturalness-pending",
                "training_admissible":False,"blockers":["no rendered SMPL measurement closure","independent naturalness and caption review pending",
                    "training input/world canonicalization unresolved","application thresholds not frozen"]})
        except ValueError as error:
            failures.append({"motion_id":mid,"reason":str(error)})
    with (args.output/"candidates.jsonl").open("w") as f:
        for row in cases:f.write(json.dumps(row)+"\n")
    dump(args.output/"summary.json",{"attempted":3,"candidates":len(cases),"failures":failures,"compatible_count":0,
        "training_admissible_count":0,"head_proxy_invariant_only":True,"budget":{"device":"cpu","clips":3,"angle_degrees":10,"parameter_search":False},
        "wrists_limitation":"Fixing both wrist positions fixes their horizontal separation; this is not an IMU incompatibility proof."})
    print(json.dumps({"output":str(args.output),"candidates":len(cases),"compatible":0,"training_admissible":0}))


if __name__=="__main__":main()
