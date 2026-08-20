"""Tariff order text -> tariff JSON, with a human in the loop.

The plan lists an LLM parser here and lists it first on the cut list. It is cut.
What replaced it is a deterministic extractor: regexes over the order text,
every hit reported with the snippet it came from, and a confirmation step that
refuses to emit a tariff nobody looked at.

That is not a downgrade. A tariff is the cost function of the whole system; a
hallucinated peak multiplier silently changes every number downstream and
nothing would fail loudly. Determinism plus a diff a human signs off on is the
right shape for this particular file, and it keeps the portability story
concrete: point it at another state's order, check the fields, rerun.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .schema import Tariff


@dataclass
class Extraction:
    field: str
    value: object
    confidence: str          # "high" | "low"
    evidence: str


NUM = r"(?:\d{1,3}(?:,\d{2,3})*|\d+)(?:\.\d+)?"


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def extract(text: str) -> list[Extraction]:
    out: list[Extraction] = []
    t = re.sub(r"[ \t]+", " ", text)

    pats = {
        "demand_charge_per_kva": [
            rf"(?:demand charge|demand charges)[^\n]{{0,80}}?(?:Rs\.?|₹|INR)\s*({NUM})\s*(?:per|/)\s*kVA",
            rf"(?:Rs\.?|₹|INR)\s*({NUM})\s*(?:per|/)\s*kVA\s*(?:per|/)\s*month",
        ],
        "energy_rate": [
            rf"(?:energy charge|energy charges|unit rate)[^\n]{{0,80}}?(?:Rs\.?|₹|INR)\s*({NUM})\s*(?:per|/)\s*(?:kWh|unit)",
            rf"(?:Rs\.?|₹|INR)\s*({NUM})\s*(?:per|/)\s*(?:kWh|unit)",
        ],
        "contract_demand_kva": [rf"contract(?:ed)? demand[^\n]{{0,40}}?({NUM})\s*kVA"],
        "billing_demand_floor_pct": [
            rf"(?:billing demand|recorded demand)[^\n]{{0,120}}?({NUM})\s*%\s*of\s*(?:the\s*)?contract",
            rf"({NUM})\s*%\s*of\s*(?:the\s*)?contract(?:ed)? demand",
        ],
        "electricity_duty_pct": [rf"electricity (?:duty|tax)[^\n]{{0,40}}?({NUM})\s*%"],
        "billing_interval_minutes": [rf"({NUM})\s*[- ]?minute\s*(?:block|interval|integration)"],
    }
    for field, plist in pats.items():
        for p in plist:
            m = re.search(p, t, re.I)
            if m:
                v = _f(m.group(1))
                if field == "billing_interval_minutes":
                    v = int(v)
                out.append(Extraction(field, v, "high" if plist.index(p) == 0 else "low",
                                      t[max(0, m.start() - 60): m.end() + 60].strip()))
                break

    # ToD windows: "18:00 to 22:00 ... 25%" or "1.25 times"
    windows = []
    for m in re.finditer(
        rf"(\d{{1,2}}[:.]\d{{2}})\s*(?:hrs?|hours)?\s*(?:to|-|–|until)\s*(\d{{1,2}}[:.]\d{{2}})"
        rf"[^\n]{{0,120}}?(?:(\d{{1,3}}(?:\.\d+)?)\s*%|({NUM})\s*times)", t, re.I,
    ):
        start, end = m.group(1).replace(".", ":"), m.group(2).replace(".", ":")
        seg = m.group(0).lower()
        if m.group(3):
            pct = float(m.group(3)) / 100.0
            mult = 1.0 - pct if any(w in seg for w in ("rebate", "less", "below", "discount", "solar")) else 1.0 + pct
        else:
            mult = float(m.group(4))
        name = ("peak" if "peak" in seg else "solar" if "solar" in seg
                else "night" if "night" in seg else "normal")
        windows.append({"name": name, "start": start, "end": end, "multiplier": round(mult, 4)})
    if windows:
        windows, filled = _complete_day(windows)
        note = "ToD windows are the field most often mis-parsed; check every row."
        if filled:
            note += (" Gaps at " + ", ".join(filled) +
                     " were not stated in the order and were filled with a 'normal' window at"
                     " multiplier 1.0, which is what a ToD schedule expressed as deltas implies.")
        out.append(Extraction("tod_windows", windows, "low", note))
    return out


def _mins(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _complete_day(windows: list[dict]) -> tuple[list[dict], list[str]]:
    """Fill any part of the day the order did not name.

    Orders state peak/solar/night as deltas and leave the ordinary hours implicit.
    The schema requires a partition, so the implicit remainder is made explicit
    at multiplier 1.0 -- and reported, because a silently invented window is
    exactly the failure mode this module exists to prevent.
    """
    covered = [False] * 1440
    for w in windows:
        a, b = _mins(w["start"]), _mins(w["end"])
        rng = range(a, b) if a < b else list(range(a, 1440)) + list(range(0, b))
        for m in rng:
            covered[m] = True
    gaps, start = [], None
    for m in range(1440):
        if not covered[m] and start is None:
            start = m
        elif covered[m] and start is not None:
            gaps.append((start, m)); start = None
    if start is not None:
        gaps.append((start, 1440))
    filled = []
    for a, b in gaps:
        s_, e_ = f"{a // 60:02d}:{a % 60:02d}", f"{(b % 1440) // 60:02d}:{(b % 1440) % 60:02d}"
        windows.append({"name": "normal", "start": s_, "end": e_, "multiplier": 1.0})
        filled.append(f"{s_}-{e_}")
    return windows, filled


def compile_tariff(
    text: str, base: dict, confirm: bool = True, assume_yes: bool = False
) -> tuple[dict, list[Extraction]]:
    """Merge extractions into ``base`` and refuse to emit unconfirmed output."""
    ext = extract(text)
    draft = dict(base)
    for e in ext:
        draft[e.field] = e.value

    print("\nExtracted from the order text:\n")
    for e in ext:
        print(f"  [{e.confidence:>4}] {e.field} = {json.dumps(e.value)}")
        print(f"         evidence: ...{e.evidence[:150]}...")
    missing = [k for k in ("energy_rate", "demand_charge_per_kva", "tod_windows") if k not in {e.field for e in ext}]
    if missing:
        print(f"\n  NOT FOUND, kept from the template: {missing}")

    if confirm and not assume_yes:
        ans = input("\nEvery number above changes every rupee downstream. Confirmed against the order? [y/N] ")
        if ans.strip().lower() not in ("y", "yes"):
            raise SystemExit("not confirmed; nothing written")

    try:
        Tariff.from_dict(draft).validate()
    except ValueError as exc:
        raise SystemExit(
            f"\nrefusing to write an invalid tariff: {exc}\n"
            "Fix the order text or the template, then rerun. A tariff that does not "
            "partition the day makes the bill ambiguous, and every number downstream "
            "inherits the ambiguity."
        ) from exc
    return draft, ext


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile a tariff order into tariff JSON.")
    ap.add_argument("order_text", type=Path, help="plain-text extract of the tariff order")
    ap.add_argument("--template", type=Path,
                    default=Path(__file__).resolve().parent / "orders/tnerc_2026.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt (CI only)")
    args = ap.parse_args()

    base = json.loads(args.template.read_text())
    draft, _ = compile_tariff(args.order_text.read_text(), base, assume_yes=args.yes)
    args.out.write_text(json.dumps(draft, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
