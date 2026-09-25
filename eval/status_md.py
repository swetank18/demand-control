"""Refresh the live block of ../ieee_paper_status.md from results/.

The narrative in that file is written by hand. Everything between the
``<!-- live:start -->`` and ``<!-- live:end -->`` markers is replaced by this
script from the result files, so the numbers in the status file can never drift
from what the run actually produced.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.windows import primary_results

STATUS = ROOT.parent / "ieee_paper_status.md"
RESULTS = ROOT / "results"
FOX_H64 = 0.405   # the paper's original single-building figure, for reference


def india_benchmark() -> str:
    p = RESULTS / "comparative_india.json"
    if not p.exists():
        return "_benchmark not started_\n"
    d = json.loads(p.read_text())
    rows = []
    for v in d.values():
        if "models" not in v:
            continue
        o, sn = v["models"]["lightgbm_quantile"], v["models"]["seasonal_naive"]
        rows.append((v["usage"] != "campus", v["id"].replace("IIITD_", ""), v["test_june"], v["train_months"],
                     o["pinball_mean"], sn["pinball_mean"], o["skill_vs_seasonal"], o["coverage_90"]))
    rows.sort()
    out = ["| level | series | June | train | ours | seas. naive | skill | cov90 |", "|---|---|---|---|---|---|---|---|"]
    for b, s, y, m, o, sn, sk, c in rows:
        out.append(f"| {'building' if b else '**campus**'} | {s} | {y} | {m} mo | {o:.3f} | {sn:.3f} | {sk:+.3f} | {c:.3f} |")
    bl = [r for r in rows if r[0]]
    if bl:
        import statistics as st
        out.append("")
        out.append(f"Building rows: n={len(bl)}, mean skill {st.mean(r[6] for r in bl):+.3f}, ours wins "
                   f"{sum(r[6] > 0 for r in bl)}/{len(bl)}, coverage {min(r[7] for r in bl):.3f}–{max(r[7] for r in bl):.3f} "
                   f"(mean {st.mean(r[7] for r in bl):.3f}).")
        camp = [r for r in rows if not r[0]]
        if camp:
            out.append(f"Aggregation ladder, coverage of nominal 0.90: buildings {st.mean(r[7] for r in bl):.3f} → "
                       f"campus {camp[0][7]:.3f} → Delhi city 0.762.")
    return "\n".join(out) + "\n"


def india_horizon() -> str:
    files = primary_results(RESULTS)
    if not files:
        return "_horizon stage not started_\n"
    out = ["| series | June | per-step α̂ | realised H=64 | 95% CI (day-block) | independence | Boole | copula | lag-1 ρ |",
           "|---|---|---|---|---|---|---|---|---|"]
    vals = []
    for f in files:
        d = json.loads(f.read_text())
        r = [x for x in d["marginal_vs_joint"] if x["H"] == 64][0]
        name, june = d["building"].replace("IIITD_", ""), d["tag"].lstrip("@")
        boole = min(1.0, 64 * r["per_step_exceedance"])
        ci = r.get("empirical_horizon_ci")
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "—"
        out.append(f"| {name} | {june} | {r['per_step_exceedance']:.3f} | **{r['empirical_horizon']:.3f}** | {ci_s} | "
                   f"{r['independence_bound']:.3f} | {boole:.2f} | {r['copula_predicted']:.3f} | {d['copula']['corr_lag1']:.3f} |")
        vals.append(r["empirical_horizon"])
    import statistics as st
    out.append("")
    out.append(f"{len(vals)} windows measured. Realised H=64 exceedance: min {min(vals):.3f}, median {st.median(vals):.3f}, "
               f"max {max(vals):.3f} (Fox_office_Gaylord, the paper's original single figure: {FOX_H64}). "
               f"Nominal level of the constraint: 0.05.")
    return "\n".join(out) + "\n"


def run_state() -> str:
    alive = subprocess.run(["pgrep", "-f", "reproduce_india.sh"], capture_output=True).returncode == 0
    log = RESULTS / "india_arm.log"
    last = ""
    if log.exists():
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        last = lines[-1][:110] if lines else ""
    return f"Run `reproduce_india.sh`: **{'RUNNING' if alive else 'finished / not running'}**. Last log line: `{last}`\n"


def main() -> None:
    live = "\n".join([
        f"_Auto-generated {datetime.now():%Y-%m-%d %H:%M} by `demand-control/eval/status_md.py` from `results/`._",
        "",
        run_state(),
        "#### India arm — benchmark (five forecasters, identical calibration layer)",
        india_benchmark(),
        "#### India arm — horizon gap, open loop (per window)",
        india_horizon(),
    ])
    text = STATUS.read_text()
    new = re.sub(r"<!-- live:start -->.*?<!-- live:end -->",
                 "<!-- live:start -->\n" + live + "\n<!-- live:end -->", text, flags=re.S)
    STATUS.write_text(new)
    print(f"refreshed {STATUS} ({datetime.now():%H:%M})")


if __name__ == "__main__":
    main()
