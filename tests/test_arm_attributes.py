import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from itm.metrics.arm_attributes import arm_spread_m,shoulder_outward_angle_deg,wrist_displacement_bound_m


def fixture():
    x=np.zeros((21,22,3))
    for j,position in {1:[.1,0,0],2:[-.1,0,0],16:[.2,.6,0],17:[-.2,.6,0],18:[.3,.3,0],19:[-.3,.3,0],20:[.3,0,0],21:[-.3,0,0],15:[0,.9,0]}.items():
        x[:,j]=position
    return x


def test_spread_is_not_swing_and_angle_uses_other_joints():
    x=fixture(); wider=x.copy();wider[:,20,0]+=.2;wider[:,21,0]-=.2
    assert arm_spread_m(wider)>arm_spread_m(x)
    assert shoulder_outward_angle_deg(wider)==shoulder_outward_angle_deg(x)
    assert arm_spread_m(wider)-arm_spread_m(x)<=wrist_displacement_bound_m(x,wider)+1e-12


def test_geometric_candidate_preserves_head_bones_and_endpoints(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/"scripts"))
    spec=importlib.util.spec_from_file_location("candidate_builder",root/"scripts/build_arm_spread_candidates.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    x=fixture();edited=module.rotate_arm_candidates(x)
    np.testing.assert_array_equal(edited[:,15],x[:,15])
    np.testing.assert_allclose(edited[[0,-1]],x[[0,-1]],atol=1e-12)
    for a,b in ((16,18),(18,20),(17,19),(19,21)):
        np.testing.assert_allclose(np.linalg.norm(edited[:,a]-edited[:,b],axis=-1),np.linalg.norm(x[:,a]-x[:,b],axis=-1),atol=1e-12)
    assert arm_spread_m(edited)>arm_spread_m(x)
    assert shoulder_outward_angle_deg(edited)>shoulder_outward_angle_deg(x)


def test_degenerate_torso_does_not_produce_semantic_success():
    with pytest.raises(ValueError,match="degenerate"):
        shoulder_outward_angle_deg(np.zeros((3,22,3)))
