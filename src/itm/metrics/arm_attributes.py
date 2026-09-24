"""Geometric arm attributes and independent diagnostic checks.

No metric here alone certifies semantic intent or motion naturalness. Wrist
spread matches the training attribute; shoulder angle is an additional check.
"""
import numpy as np


def _joints(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=3 or values.shape[1:]!=(22,3) or len(values)<3 or not np.isfinite(values).all():
        raise ValueError("finite (T>=3,22,3) joints required")
    return values


def arm_spread_m(joints):
    values=_joints(joints)
    return float(np.linalg.norm((values[:,20]-values[:,21])[:,[0,2]],axis=-1).mean())


def shoulder_outward_angle_deg(joints):
    """Mean signed upper-arm abduction in the torso's lateral/down plane.

    Uses shoulders/elbows/hips rather than the wrist endpoint distance trained
    by B. Degenerate torso/upper-arm projections are rejected, not filled with 0.
    """
    v=_joints(joints)
    left=v[:,16]-v[:,17]
    down=(v[:,1]+v[:,2]-v[:,16]-v[:,17])/2
    norms=np.linalg.norm(left,axis=-1,keepdims=True)
    if np.any(norms<1e-6):raise ValueError("degenerate shoulder width")
    left=left/norms
    down=down-(down*left).sum(-1,keepdims=True)*left
    norms=np.linalg.norm(down,axis=-1,keepdims=True)
    if np.any(norms<1e-6):raise ValueError("degenerate torso frame")
    down=down/norms
    angles=[]
    for shoulder,elbow,outward in ((16,18,left),(17,19,-left)):
        arm=v[:,elbow]-v[:,shoulder]
        lateral=(arm*outward).sum(-1); vertical=(arm*down).sum(-1)
        if np.any(np.hypot(lateral,vertical)<1e-6):raise ValueError("degenerate arm projection")
        angles.append(np.degrees(np.arctan2(lateral,vertical)))
    return float(np.stack(angles,-1).mean())


def wrist_displacement_bound_m(base,edit):
    """Triangle-inequality bound for a *joint-position* proxy, not an IMU bound."""
    a,b=_joints(base),_joints(edit)
    if a.shape!=b.shape:raise ValueError("matching physical time axes required")
    return float(np.linalg.norm(b[:,[20,21]]-a[:,[20,21]],axis=-1).sum(-1).mean())
