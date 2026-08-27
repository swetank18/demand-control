"""Ship the comparative benchmark into the Next.js app as data.

Same contract as eval/export_web.py: the web app never recomputes anything. It
renders what the benchmark produced, so the deployed page and the study document
cannot drift apart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import data.sites as S

RESULTS = ROOT / "results"
CACHE = ROOT / "data/cache"
APP = Path("/home/swetank/hackit/ampcast")
OURS = "lightgbm_quantile"


def merge_all() -> pd.DataFrame:
    """Read every result file without writing to any of them.

    The benchmark runs write results/comparative.json and
    results/comparative_national.json incrementally, one series at a time, and
    may still be running. This only ever reads them; an earlier version briefly
    overwrote comparative.json to reuse the report loader, which would have
    raced the live writer and could have dropped completed series.
    """
    merged: dict = {}
    for name in ("comparative", "comparative_climate", "comparative_office",
                 "comparative_demographic", "comparative_national"):
        path = RESULTS / f"{name}.json"
        if not path.exists():
            continue
        try:
            merged.update(json.loads(path.read_text()))
        except json.JSONDecodeError:
            # a partial write caught mid-flight; skip this pass rather than fail
            print(f"  ! {path.name} was mid-write, skipped this pass")
    if not merged:
        raise SystemExit("no results yet")
    (RESULTS / "comparative_merged.json").write_text(json.dumps(merged, indent=2, default=str))

    rows = []
    for tag, r in merged.items():
        if "models" not in r:
            rows.append(dict(tag=tag, id=r.get("id"), arm=r.get("arm"), error=r.get("error")))
            continue
        base = dict(tag=tag, id=r["id"], arm=r["arm"], site=r["site"], usage=r["usage"],
                    country=r["country"], tier=r["tier"], median_load=r["median_load"],
                    n_test=r["n_test"])
        for k, d in r["models"].items():
            base[f"{k}_pinball"] = d["pinball_mean"]
            base[f"{k}_skill"] = d.get("skill_vs_seasonal")
            base[f"{k}_cov90"] = d.get("coverage_90")
            base[f"{k}_pct"] = d.get("pinball_pct_of_median")
        rows.append(base)
    return pd.DataFrame(rows)


def covariates() -> pd.DataFrame:
    clim = S.climate_table(S.load_weather(), S.load_meta())
    rows = {}
    man = CACHE / "manifest_comparative.json"
    if man.exists():
        for bid, p in json.loads(man.read_text())["buildings"].items():
            site = bid.split("_")[0]
            rows[bid] = dict(
                cdd_share=float(clim.loc[site, "cdd_share"]) if site in clim.index else np.nan,
                t_mean=float(clim.loc[site, "t_mean"]) if site in clim.index else np.nan,
                hvac_share=p["hvac_share_of_meter"],
                tau_h=p["thermal"]["time_constant_h"],
                median_kw=None,
                ua_flagged="HAND-PICKED" in p["thermal"]["ua_source"],
                corr_temp=np.nan,
            )
    nat = CACHE / "manifest_national.json"
    if nat.exists():
        for name, m in json.loads(nat.read_text()).items():
            rows[name] = dict(cdd_share=m["cdd_share"], t_mean=m["t_mean"],
                              hvac_share=np.nan, tau_h=np.nan, median_kw=None,
                              ua_flagged=False, corr_temp=m["corr_temp"],
                              unit=m["unit"], native_min=m["native_resolution_min"],
                              note=m["note"], window=m["window"],
                              p99_over_median=m["p99_over_median"])
    return pd.DataFrame(rows).T.rename_axis("id").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=APP)
    args = ap.parse_args()

    df = merge_all()
    df = df[df.get("error").isna()] if "error" in df.columns else df
    cov = covariates()
    df = df.merge(cov, on="id", how="left")

    def rows_for(arm: str) -> list[dict]:
        g = df[df.arm == arm].copy()
        out = []
        for _, r in g.iterrows():
            out.append({
                "id": r["id"], "site": r.get("site"), "usage": r.get("usage"),
                "country": r.get("country"), "tier": int(r.get("tier", 1)),
                "cddShare": None if pd.isna(r.get("cdd_share")) else float(r["cdd_share"]),
                "tMean": None if pd.isna(r.get("t_mean")) else float(r["t_mean"]),
                "hvacShare": None if pd.isna(r.get("hvac_share")) else float(r["hvac_share"]),
                "corrTemp": None if pd.isna(r.get("corr_temp")) else float(r["corr_temp"]),
                "medianLoad": None if pd.isna(r.get("median_load")) else float(r["median_load"]),
                "uaFlagged": bool(r.get("ua_flagged", False)),
                "note": r.get("note") if isinstance(r.get("note"), str) else None,
                "nativeMin": None if pd.isna(r.get("native_min")) else int(r["native_min"]),
                "models": {
                    k: {
                        "pinball": None if pd.isna(r.get(f"{k}_pinball")) else float(r[f"{k}_pinball"]),
                        "skill": None if pd.isna(r.get(f"{k}_skill")) else float(r[f"{k}_skill"]),
                        "cov90": None if pd.isna(r.get(f"{k}_cov90")) else float(r[f"{k}_cov90"]),
                        "pct": None if pd.isna(r.get(f"{k}_pct")) else float(r[f"{k}_pct"]),
                    }
                    for k in ("static_margin", "persistence", "climatology", "seasonal_naive", OURS)
                    if f"{k}_pinball" in r.index and not pd.isna(r.get(f"{k}_pinball"))
                },
            })
        return out

    # the survey result, which stands on its own before any model runs
    surv = pd.read_csv(ROOT / "data/site_survey.csv")
    clim = S.climate_table(S.load_weather(), S.load_meta())
    sites = []
    for site, g in surv.groupby("site"):
        if site not in clim.index:
            continue
        sites.append({
            "site": site, "country": g.country.iloc[0],
            "metered": int(len(g)), "passes": int(g.passes_rule.sum()),
            "heating": int(g.heating_dominated.sum()),
            "cddShare": float(clim.loc[site, "cdd_share"]),
            "tMean": float(clim.loc[site, "t_mean"]),
        })
    sites.sort(key=lambda s: -s["cddShare"])

    def corr(a: str, b: str, arms: tuple[str, ...]) -> dict | None:
        g = df[df.arm.isin(arms)].dropna(subset=[a, b])
        if len(g) < 3:
            return None
        return {"r": float(np.corrcoef(g[a].astype(float), g[b].astype(float))[0, 1]), "n": int(len(g))}

    payload = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "sites": sites,
        "arms": {a: rows_for(a) for a in ("climate", "office", "demographic", "national")},
        "correlations": {
            "cdd_vs_hvac": corr("cdd_share", "hvac_share", ("climate", "office", "demographic")),
            "cdd_vs_skill": corr("cdd_share", f"{OURS}_skill", ("climate", "office")),
            "hvac_vs_skill": corr("hvac_share", f"{OURS}_skill", ("climate", "office", "demographic")),
        },
        "counts": {
            "passing": int(surv.passes_rule.sum()),
            "heatingOnly": int(surv.heating_dominated.sum()),
            "sitesWithPasses": int(surv[surv.passes_rule].site.nunique()),
            "countries": int(surv[surv.passes_rule].country.nunique()),
        },
    }

    dest = args.out / "src/lib/comparative.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=1))
    n = sum(len(v) for v in payload["arms"].values())
    print(f"wrote {dest}  ({len(json.dumps(payload))/1024:.1f} KB, {n} series, {len(sites)} sites)")
    for a, v in payload["arms"].items():
        print(f"  {a:12} {len(v)}")


if __name__ == "__main__":
    main()
