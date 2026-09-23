from pathlib import Path
import sys

import numpy as np
import pytest

from itm.backbones import HYMotionBackbone, MotionConditions


def test_hymotion_lengths_and_decode(tmp_path):
    backbone = HYMotionBackbone(tmp_path, tmp_path)
    assert backbone._integer_lengths(np.asarray([60])) == [60]
    joints = np.zeros((1, 60, 52, 3), dtype=np.float32)
    assert backbone.decode_motion({"keypoints3d": joints}).shape == (1, 60, 22, 3)


def test_hymotion_initial_adapter_rejects_batches(tmp_path):
    backbone = HYMotionBackbone(tmp_path, tmp_path)
    conditions = MotionConditions(text=["walk", "run"], lengths=[60, 60])
    with pytest.raises(ValueError, match="exactly one"):
        backbone.sample(conditions, seed=1)


def test_hymotion_explicit_python_runs_sampler(tmp_path):
    (tmp_path / "workspace.json").write_text("{}")
    sampler = tmp_path / "sampler.py"
    sampler.write_text(
        "import json, sys\n"
        "import numpy as np\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "assert args['--text'] == 'a person walks'\n"
        "assert args['--device'] == 'cpu'\n"
        "assert args['--seed'] == '1234'\n"
        "n = int(args['--frames'])\n"
        "np.savez(args['--output'], latent_denorm=np.zeros((1,n,201)),\n"
        " keypoints3d=np.zeros((1,n,52,3)), rot6d=np.zeros((1,n,22,6)),\n"
        " transl=np.zeros((1,n,3)), root_rotations_mat=np.zeros((1,n,3,3)),\n"
        " global_rotations_mat=np.zeros((1,n,22,3,3)),\n"
        " metadata=np.array(json.dumps({'executable': sys.executable})))\n"
    )
    backbone = HYMotionBackbone(
        tmp_path, tmp_path, device="cpu", sampler_script=sampler,
        python_executable=Path(sys.executable),
    )
    result = backbone.sample(MotionConditions(text=["a person walks"], lengths=[60]), seed=1234)
    assert result["keypoints3d"].shape == (1, 60, 52, 3)
    assert result["metadata"]["executable"] == sys.executable


def test_hymotion_python_preserves_venv_symlink(tmp_path):
    executable = tmp_path / "python"
    executable.symlink_to(sys.executable)
    backbone = HYMotionBackbone(tmp_path, tmp_path, python_executable=executable)
    assert backbone.python_executable == executable
