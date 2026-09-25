"""Offline synthetic pair coordinates; never a head/wrist deployment calibration.

HumanML263 field layout follows the local MDM motion_process implementation.
The source SMPL body proportions are retained (no uniform_skeleton retarget).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

SCHEMA = "canonical_smpl_paired_v1"
MAPPING_VERSION = "canonical-smpl-pair-v1"
CHAINS = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15],
          [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def canonical_transform(joints):
    """A constant proper yaw and translation using full GT OFFLINE only.

    Heading uses anatomical right-minus-left hips AND shoulders, with the same
    20-frame Gaussian smoothing as the encoder; its first value defines +Z.
    This makes the omitted initial root quaternion exactly the decoder identity.
    """
    p = np.asarray(joints, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (22, 3) or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Finite (T,22,3), T>=3 required")
    across = p[:, 2] - p[:, 1] + p[:, 17] - p[:, 16]
    norms = np.linalg.norm(across, axis=-1, keepdims=True)
    if np.min(norms) < 1e-8:
        raise ValueError("Degenerate body heading")
    forward = np.cross(np.array([0., 1., 0.]), across / norms)
    forward = gaussian_filter1d(forward, 20, axis=0, mode="nearest")
    if np.linalg.norm(forward[0, [0, 2]]) < 1e-8:
        raise ValueError("Degenerate smoothed heading")
    angle = -np.arctan2(forward[0, 0], forward[0, 2])
    matrix = Rotation.from_rotvec([0, angle, 0]).as_matrix()
    origin = np.array([p[0, 0, 0], np.min(p[..., 1]), p[0, 0, 2]])
    return matrix, origin


def transform_points(points, matrix, origin):
    return (np.asarray(points, dtype=np.float64) - origin) @ matrix.T


def encode_humanml263(joints, *, mdm_root=None):
    """HumanML263 layout with float64 yaw/IK arithmetic and no quaternion bias.

    Rotation channels retain the reference HumanML chain-root IK convention;
    they are not SMPL segment orientations. No body-size uniformization occurs.
    """
    p = np.asarray(joints, dtype=np.float64)
    across = p[:, 2] - p[:, 1] + p[:, 17] - p[:, 16]
    across /= np.linalg.norm(across, axis=-1, keepdims=True)
    forward = gaussian_filter1d(np.cross([0., 1., 0.], across), 20, axis=0, mode="nearest")
    if np.min(np.linalg.norm(forward[:, [0, 2]], axis=-1)) < 1e-8:
        raise ValueError("Degenerate smoothed heading")
    yaw = np.unwrap(-np.arctan2(forward[:, 0], forward[:, 2]))
    if abs(yaw[0]) > 1e-6:
        raise ValueError("Canonical initial root is not decoder identity")
    root = Rotation.from_rotvec(np.column_stack([yaw*0, yaw, yaw*0])).as_matrix()
    delta = np.diff(yaw)
    if np.any(np.abs(delta) >= np.pi):
        raise ValueError("Ambiguous root angular increment")
    velocity = np.einsum('tij,tj->ti', root[1:], p[1:, 0]-p[:-1, 0])
    root_data = np.concatenate([delta[:, None]/2, velocity[:, [0,2]], p[:-1, 0, 1:2]], axis=-1)
    local = p.copy()
    local[..., 0] -= p[:, 0:1, 0]
    local[..., 2] -= p[:, 0:1, 2]
    local = np.einsum('tij,tkj->tki', root, local)
    local_velocity = np.einsum('tij,tkj->tki', root[:-1], p[1:]-p[:-1])
    raw = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,-1,0],
        [0,1,0],[0,-1,0],[0,-1,0],[0,1,0],[0,0,1],[0,0,1],[0,1,0],
        [1,0,0],[-1,0,0],[0,0,1],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0]], dtype=float)
    rotations = np.zeros((len(p),22,3,3))
    rotations[:,0]=root
    for chain in CHAINS:
        parent = root
        for first, second in zip(chain[:-1],chain[1:]):
            bone = p[:,second]-p[:,first]
            length = np.linalg.norm(bone,axis=-1,keepdims=True)
            if np.min(length)<1e-8: raise ValueError("Degenerate bone")
            bone /= length
            quat = np.column_stack([np.cross(raw[second],bone), 1+bone@raw[second]])
            antiparallel = np.linalg.norm(quat,axis=-1)<1e-8
            if antiparallel.any():
                axis=np.cross(raw[second],np.eye(3)[np.argmin(np.abs(raw[second]))])
                quat[antiparallel,:3]=axis/np.linalg.norm(axis)
                quat[antiparallel,3]=0
            global_rotation=Rotation.from_quat(quat).as_matrix()
            rotations[:,second]=np.matmul(parent.transpose(0,2,1),global_rotation)
            parent=global_rotation
    cont6d=np.concatenate([rotations[...,0],rotations[...,1]],axis=-1)
    feet = ((p[1:,[7,10,8,11]]-p[:-1,[7,10,8,11]])**2).sum(-1)<.002
    features = np.concatenate([root_data,local[:-1,1:].reshape(len(p)-1,63),
        cont6d[:-1,1:].reshape(len(p)-1,126),local_velocity.reshape(len(p)-1,66),feet.astype(float)],axis=-1)
    if features.shape != (len(p)-1,263) or not np.isfinite(features).all():
        raise ValueError("Invalid HumanML263 encoding")
    return features.astype(np.float32)
