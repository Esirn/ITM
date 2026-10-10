"""Tensor fixtures for caption path construction; no pretrained/GPU calls."""
import importlib.util
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('path_probe',ROOT/'scripts/run_b0_caption_path_probe.py')
module=importlib.util.module_from_spec(spec)


@pytest.fixture(scope='module')
def probe():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'));spec.loader.exec_module(module);return module


def test_paths_preserve_native_slots_and_strict_text_null(probe):
    single=torch.ones(1,1,8);native=torch.arange(40,dtype=torch.float32).reshape(1,5,8)
    broadcast,five=probe.text_paths(single,native,torch.tensor([False,True,True,True,True]))
    assert not broadcast[:,0].any() and not five[:,0].any()
    assert torch.equal(five[:,1:],native[:,1:]) and torch.equal(broadcast[:,1:],single.repeat(1,4,1))


@pytest.mark.parametrize('presence',[[True]*5,[False]*5,[False,True,True,True]])
def test_wrong_native_presence_or_batch_rejected(probe,presence):
    with pytest.raises(ValueError):probe.text_paths(torch.ones(1,1,8),torch.ones(1,5,8),torch.tensor(presence))


def test_exact_frozen_caps_and_denominators(probe):
    assert probe.CAPS==dict(mdm=(4,18),encoder=(6,6),caption=(2,6))
    assert len(probe.initial_ledger()['jobs'])==4 and len(probe.initial_ledger()['checks'])==9
