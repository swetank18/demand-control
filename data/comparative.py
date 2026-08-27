"""Build the comparative panel: the same rule, applied across climates and countries.

The repo's evidence comes from four buildings at one site in Tempe, Arizona, and
`data/buildings.json` already admits what that costs: "It is not India, and the
load shapes are American." This closes half of that gap — not by finding Indian
data, which BDG2 does not contain, but by establishing *how far the method
travels* across the climates and building stocks it can be tested on.

Two arms, because "comparative" without a controlled factor is just more numbers.

**Climate arm.** Demographic held fixed, climate and country varied. Education
is the one usage class that survives the selection rule at every site on the
ladder, so it is the control. Seven sites spanning cooling-degree-day share 0.87
(Phoenix) to 0.00 (Dublin), across the USA, the UK and Ireland. Office is run as
a second, narrower ladder (five sites, three countries) to check that whatever we
find is a property of climate and not an artefact of one usage class.

**Demographic arm.** Climate held fixed, demographic varied. Rat (Washington DC)
is the only site with enough passing buildings to populate eight usage classes,
including Lodging/residential — which is the demographic Samanvay actually
targets and which the current evidence base does not contain at all.

Selection inside a (site, usage) cell is by the repo's own stated criterion for
why `Fox_office_Gaylord` is the primary building: the sharpest demand-charge
exposure, p99/median. Applied uniformly, never hand-picked.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

#: cooling-degree-day share, from data/sites.py. Ordered hot -> cold.
CLIMATE_LADDER = ["Fox", "Bull", "Rat", "Hog", "Bear", "Robin", "Wolf"]
OFFICE_LADDER = ["Fox", "Rat", "Hog", "Robin", "Wolf"]
DEMOGRAPHIC_SITE = "Rat"

CLIMATE_USAGE = "Education"


def pick(p: pd.DataFrame, site: str, usage: str) -> dict | None:
    """Sharpest demand-charge exposure in the cell. The repo's own criterion."""
    g = p[(p.site == site) & (p.usage == usage)]
    if g.empty:
        return None
    b = g.sort_values("p99_over_median", ascending=False).iloc[0]
    return {
        "id": b.building_id,
        "role": "primary",
        "label": f"{site} {usage}",
        "site": site,
        "usage": usage,
        "country": b.country,
        "why": (
            f"sharpest demand-charge exposure among {len(g)} {usage} building(s) "
            f"passing the rule at {site}: p99/median = {b.p99_over_median:.1f}, "
            f"median {b.median_kw:.0f} kW, load-temp corr {b.corr_temp:.2f}"
        ),
    }


def main() -> None:
    r = pd.read_csv(ROOT / "site_survey.csv")
    p = r[r.passes_rule]

    panel: dict[str, list[dict]] = {"climate": [], "office": [], "demographic": []}

    for site in CLIMATE_LADDER:
        b = pick(p, site, CLIMATE_USAGE)
        if b:
            panel["climate"].append(b)
    for site in OFFICE_LADDER:
        b = pick(p, site, "Office")
        if b:
            panel["office"].append(b)
    for usage in sorted(p[p.site == DEMOGRAPHIC_SITE].usage.unique()):
        b = pick(p, DEMOGRAPHIC_SITE, usage)
        if b:
            panel["demographic"].append(b)

    spec = {
        "_": __doc__.strip().splitlines()[0],
        "selection_rule": (
            "identical to data/buildings.json: no district chilled-water meter, "
            ">97% coverage in the split window, median load 40-800 kW, positive "
            "load-vs-outdoor-temperature correlation. Tie-break inside a "
            "(site, usage) cell: highest p99/median."
        ),
        "climate_usage": CLIMATE_USAGE,
        "arms": panel,
    }
    (ROOT / "comparative.json").write_text(json.dumps(spec, indent=2))

    # one prepare config per site, since prepare.py works a site at a time
    everything: dict[str, list[dict]] = {}
    for arm, items in panel.items():
        for b in items:
            everything.setdefault(b["site"], [])
            if b["id"] not in [x["id"] for x in everything[b["site"]]]:
                everything[b["site"]].append(b)

    cfg_dir = ROOT / "configs"
    cfg_dir.mkdir(exist_ok=True)
    for site, blds in everything.items():
        (cfg_dir / f"{site}.json").write_text(json.dumps(
            {"site": site, "site_note": f"comparative panel, {site}", "buildings": blds}, indent=2))

    print("comparative panel")
    for arm, items in panel.items():
        print(f"\n--- {arm} arm ({len(items)}) ---")
        for b in items:
            print(f"  {b['site']:7} {b['country']:8} {b['usage']:30} {b['id']}")
    print(f"\n{len(everything)} site configs -> {cfg_dir}")
    print(f"{sum(len(v) for v in everything.values())} unique buildings to prepare")


if __name__ == "__main__":
    main()
