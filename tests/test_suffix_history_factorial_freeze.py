"""Reuse immutable old-noise/native-terminal gates; no model or new real noise."""
import pytest

from freeze_b0_suffix_history_factorial import suffix_noise,validate_terminal
from test_suffix_switch_freeze import test_terminal_full_native_identity_required


def test_preserve_original_suffix_noise_boundary():
    records=[dict(step=t) for t in [-1]+list(range(999,0,-1))]
    assert suffix_noise(records)==records[101:]
    records[101]['step']=900
    with pytest.raises(ValueError):suffix_noise(records)
