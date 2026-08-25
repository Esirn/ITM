from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
DRAFT = ROOT / "paper" / "论文草稿.md"
BIBLIOGRAPHY = ROOT / "paper" / "references.bib"


def citation_keys(text: str) -> set[str]:
    return set(re.findall(r"\[([a-zA-Z][a-zA-Z0-9]+)\]", text))


def bibliography_keys(text: str) -> set[str]:
    return set(re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE))


def test_paper_citations_resolve_and_entries_are_used():
    cited = citation_keys(DRAFT.read_text(encoding="utf-8"))
    bibliography = bibliography_keys(BIBLIOGRAPHY.read_text(encoding="utf-8"))
    assert cited
    assert cited == bibliography


def test_bibliography_has_balanced_braces_and_unique_keys():
    text = BIBLIOGRAPHY.read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", text, flags=re.MULTILINE)
    assert text.count("{") == text.count("}")
    assert len(keys) == len(set(keys))


def test_figure_manifest_covers_main_and_preliminary_figures():
    text = (ROOT / "paper" / "FIGURE_MANIFEST.md").read_text(encoding="utf-8")
    for figure in ("Figure 1", "Figure 2", "Figure 3", "Figure 4"):
        assert figure in text
