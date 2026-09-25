"""Which horizon-risk result files are the study's windows, and which are
calibration variants of them.

A model directory's tag is "@<June>" for one of the I-BLEND windows, and a
trailing letter marks a variant of that same window calibrated differently
(@2016k is the 2016 window with the adaptive step stated as a fraction of
interval width; @g and @k are the Phoenix office at each step). A variant
changes how the forecast is calibrated, never which dates it was trained on,
so it belongs in the comparison table that is about the calibration and
nowhere else. Every table and figure that reports "the windows" filters
through here, so adding another variant cannot quietly double the panel.
"""
from __future__ import annotations

from pathlib import Path

PREFIX = "horizon_risk_"


def tag_of(path: Path) -> str:
    """'' for an untagged series, '2016' or '2016k' for a tagged window."""
    _, _, tag = path.stem[len(PREFIX):].partition("@")
    return tag


def is_primary(path: Path) -> bool:
    """A window of the study proper: no tag, or a tag that is only a year."""
    tag = tag_of(path)
    return tag == "" or tag.isdigit()


def primary_results(results: Path, pattern: str = "horizon_risk_IIITD_*.json") -> list[Path]:
    return sorted(p for p in results.glob(pattern) if is_primary(p))
