"""The Applied Energy submission's front matter, held to the journal's rules.

These are the checks an editorial office makes before a reviewer sees the
paper, so a manuscript that fails one is returned unread: 3 to 5 highlights of
at most 85 characters each, supplied as their own file; an abstract within the
word limit (guides quote 200 or 250, so 200); 4 to 6 keywords. And the
manuscript must open on its title page -- elsarticle prints a highlights
environment on a page of its own *before* the title, which is how the front
page went missing once.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ELS = ROOT / "docs/paper_elsevier"


def _body(name: str) -> str:
    s = (ELS / "main.tex").read_text()
    m = re.search(rf"\\begin\{{{name}\}}(.*?)\\end\{{{name}\}}", s, re.S)
    assert m, f"no {name} environment"
    return "\n".join(l for l in m.group(1).splitlines() if not l.lstrip().startswith("%"))


def test_manuscript_opens_on_its_title_page():
    s = (ELS / "main.tex").read_text()
    live = "\n".join(l.split("%")[0] for l in s.splitlines())
    assert "\\begin{highlights}" not in live
    assert "\\begin{graphicalabstract}" not in live


def test_highlights_file_meets_the_journal_rules():
    lines = [l.strip() for l in (ELS / "highlights.txt").read_text().splitlines()]
    bullets = [l.lstrip("•").strip() for l in lines if l.startswith("•")]
    assert 3 <= len(bullets) <= 5
    for b in bullets:
        assert len(b) <= 85, (len(b), b)


def test_abstract_within_the_word_limit():
    text = re.sub(r"\\[a-zA-Z]+|[{}$]", " ", _body("abstract"))
    assert len(text.split()) <= 200


def test_keyword_count():
    kws = [k for k in _body("keyword").split("\\sep") if k.strip()]
    assert 4 <= len(kws) <= 6
