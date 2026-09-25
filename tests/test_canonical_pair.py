from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from itm.data.canonical_pair import canonical_transform, transform_points, encode_humanml263


def fixture_joints():
    # Non-collinear limb bones and turning/translating root exercise integration.
    rest=np.array([[0,1,0],[.12,.95,0],[-.12,.95,0],[0,1.15,0],
        [.12,.55,.04],[-.12,.55,.04],[0,1.3,0],[.12,.12,0],[-.12,.12,0],[0,1.45,0],
        [.12,.06,.15],[-.12,.06,.15],[0,1.6,0],[.08,1.48,0],[-.08,1.48,0],[0,1.8,0],
        [.2,1.5,0],[-.2,1.5,0],[.45,1.35,.03],[-.45,1.35,.03],[.65,1.2,.08],[-.65,1.2,.08]])
    t=np.arange(45)/20
    q=Rotation.from_rotvec(np.column_stack([t*0,.8*t,t*0])).as_matrix()
    return np.einsum('tij,kj->tki',q,rest)+np.stack([t,.08*np.sin(t),t*t*.1],-1)[:,None]


def test_canonical_coordinates_invariant_to_constant_yaw_and_translation():
    p=fixture_joints();q=Rotation.from_rotvec([0,1.7,0]).as_matrix()
    a,b=canonical_transform(p);c,d=canonical_transform(p@q.T+[5,2,-4])
    np.testing.assert_allclose(transform_points(p,a,b),transform_points(p@q.T+[5,2,-4],c,d),atol=1e-12)
    assert np.linalg.det(a)==pytest.approx(1.)


def test_turning_root_roundtrip_without_oracle_alignment():
    mdm=Path('/home/a200/0relatedworks/motion-diffusion-model')
    if not mdm.exists():pytest.skip('local pinned MDM dependency unavailable')
    import torch
    from itm.data.humanml_torch import recover_from_ric_torch
    p=fixture_joints();q,o=canonical_transform(p);target=transform_points(p,q,o)
    features=encode_humanml263(target.copy(),mdm_root=mdm)
    recovered=recover_from_ric_torch(torch.from_numpy(features)).numpy()
    np.testing.assert_allclose(recovered,target[:-1],atol=1e-5,rtol=0)
    np.testing.assert_allclose(recovered[0,0,[0,2]],0,atol=1e-7)


def test_degenerate_heading_rejected():
    with pytest.raises(ValueError,match='Degenerate'):canonical_transform(np.zeros((8,22,3)))


def test_double_precision_diagnostic_preserves_dtype_and_quadratic():
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_acceleration_numerics import differentiate64
    t=np.arange(20,dtype=np.float64)/30
    p=np.broadcast_to((t*t)[:,None,None],(20,6,3)).copy()+.123456789012
    a=differentiate64(p)
    assert a.dtype==np.float64
    np.testing.assert_allclose(a[3:-3],2.,atol=1e-12,rtol=0)
