"""Which result files count as the study's windows.

A calibration variant (@2016k, @g, @k) is the same window with a different
adaptive step, and the panel that reports "fourteen windows" must not silently
become twenty-seven when one is added. This pins the rule.
"""
from __future__ import annotations

from pathlib import Path

from eval.windows import is_primary, primary_results, tag_of

R = Path("results")


def test_tag_and_primary():
    cases = {
        "horizon_risk_Fox_office_Gaylord.json": ("", True),
        "horizon_risk_IIITD_Academic@2016.json": ("2016", True),
        "horizon_risk_IIITD_Academic@2016k.json": ("2016k", False),
        "horizon_risk_Fox_office_Gaylord@g.json": ("g", False),
        "horizon_risk_Fox_office_Gaylord@k.json": ("k", False),
    }
    for name, (tag, primary) in cases.items():
        p = R / name
        assert tag_of(p) == tag, name
        assert is_primary(p) is primary, name


def test_primary_results_filters_a_real_directory(tmp_path):
    for name in ("horizon_risk_IIITD_A@2016.json", "horizon_risk_IIITD_A@2016k.json",
                 "horizon_risk_IIITD_B@2017.json", "horizon_risk_IIITD_B@2017k.json"):
        (tmp_path / name).write_text("{}")
    got = [p.name for p in primary_results(tmp_path)]
    assert got == ["horizon_risk_IIITD_A@2016.json", "horizon_risk_IIITD_B@2017.json"]
