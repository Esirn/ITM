import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from itm.data.random_plan import VERSION,sampling_noise


def test_noise_plan_replay_checks_hashes_and_complete_draws(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/"scripts"))
    spec=importlib.util.spec_from_file_location("handoff_acceptance",root/"scripts/accept_method_handoff.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    draws=[sampling_noise((1,3,1,4),backbone_id="fixture",case_id="a",sample_seed=1,step=step)[1] for step in (-1,2,1)]
    digest=hashlib.sha256(json.dumps(draws,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    plan={"version":VERSION,"steps":3,"sampler":"ddpm","noise_draws":draws,"random_plan_id":digest,"timestep_map":[0,1,2]}
    assert module.verify_random_plan(plan)["draws_recomputed"]==3
    plan["noise_draws"][1]["noise_sha256"]="altered"
    with pytest.raises(ValueError,match="digest"):
        module.verify_random_plan(plan)


def test_method_training_rejects_uncertain_candidates(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/"scripts"))
    spec=importlib.util.spec_from_file_location("response_training_interface",root/"scripts/train_mdm_imu_control.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=root/"docs/itm_rework/protocol/arm_spread_candidates_20260924_v1/candidates.jsonl"
    with pytest.raises(ValueError,match="compatible train"):
        module.load_response_manifest(path)
