import pytest

from freeze_b0_suffix_history_factorial_v2 import select_text_trace
from test_suffix_history_factorial_freeze import test_preserve_original_suffix_noise_boundary,test_terminal_full_native_identity_required


def test_select_text_trace_even_when_paired_first():
    records=[dict(case_id='paired',timestep=500,trace_id='P'),dict(case_id='text',timestep=500,trace_id='T')]
    assert select_text_trace(records,dict(case_id='text',condition='text_only'),500)['trace_id']=='T'
    with pytest.raises(ValueError):select_text_trace(records,dict(case_id='paired',condition='paired'),500)
    with pytest.raises(ValueError):select_text_trace(records+[records[1]],dict(case_id='text',condition='text_only'),500)
