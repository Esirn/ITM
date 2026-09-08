import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_control_strata.py"
SPEC = importlib.util.spec_from_file_location("analyze_control_strata", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_semantic_tags_are_multilabel_and_word_bounded():
    assert MODULE.semantic_tags("a person turns while walking and waves both hands") == [
        "locomotion", "turning", "upper_body"
    ]
    assert MODULE.semantic_tags("a person understands a signal") == ["other"]


def test_length_bins_are_stable_at_boundaries():
    assert MODULE.length_bin(80) == "short_1_80"
    assert MODULE.length_bin(81) == "medium_81_140"
    assert MODULE.length_bin(140) == "medium_81_140"
    assert MODULE.length_bin(141) == "long_141_196"
