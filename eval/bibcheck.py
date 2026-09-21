"""Check every entry in docs/paper/refs.bib against Crossref.

A wrong author list in a reference is the cheapest way to lose a reviewer's
trust, and it is the kind of error that survives proofreading because nobody
re-reads a bibliography. This asks Crossref for the best bibliographic match
on each entry's title and reports, per entry, whether the title, first
author, year and venue agree, and the DOI it found. Nothing is written; the
output is for a human to act on. A "~" row is an entry Crossref does not
index (NeurIPS, ICML, ICLR, TMLR); check those against OpenAlex or the
proceedings page by hand.

    ../.venv/bin/python eval/bibcheck.py            # every entry
    ../.venv/bin/python eval/bibcheck.py --key foo  # one entry, full record
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "docs/paper/refs.bib"
UA = "demand-control-bibcheck (mailto:dynamicsanvil@gmail.com)"


def parse(text: str) -> list[dict]:
    """Enough of BibTeX to get the fields this check compares."""
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        f = {"kind": kind, "key": key}
        for fm in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body):
            f[fm.group(1).lower()] = fm.group(2).strip()
        out.append(f)
    return out


def norm(s: str) -> str:
    s = re.sub(r"[{}\\$]", "", s or "").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_family(author: str) -> str:
    a = (author or "").split(" and ")[0].strip()
    return norm(a.split(",")[0] if "," in a else a.split()[-1] if a else "")


def crossref(title: str, rows: int = 2) -> list[dict]:
    q = urllib.parse.urlencode({
        "query.bibliographic": title, "rows": rows,
        "select": "DOI,title,container-title,volume,issue,page,article-number,issued,author,type"})
    req = urllib.request.Request(f"https://api.crossref.org/works?{q}", headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))["message"]["items"]


def crossref_doi(doi: str) -> dict:
    if doi.lower().startswith("10.5281/"):
        # Zenodo mints through DataCite, which Crossref does not mirror
        req = urllib.request.Request(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}",
                                     headers={"User-Agent": UA, "Accept": "application/vnd.api+json"})
        a = json.load(urllib.request.urlopen(req, timeout=30))["data"]["attributes"]
        return {"DOI": doi, "title": [t["title"] for t in a.get("titles", [])][:1],
                "author": [{"family": c.get("familyName", c.get("name", ""))} for c in a.get("creators", [])],
                "issued": {"date-parts": [[a.get("publicationYear")]]},
                "container-title": [a.get("publisher", "")], "type": "dataset"}
    req = urllib.request.Request(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}",
                                 headers={"User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=30))["message"]


def arxiv_abs(eprint: str) -> dict:
    """Title, authors and date from the abstract page's citation meta tags.
    The export API answers 406 for some recent identifiers; the page does not."""
    req = urllib.request.Request(f"https://arxiv.org/abs/{eprint}",
                                 headers={"User-Agent": "Mozilla/5.0 " + UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    meta = re.findall(r'<meta name="citation_(\w+)" content="([^"]*)"', html)
    out = {"authors": []}
    for k, v in meta:
        if k == "author":
            out["authors"].append(v)
        else:
            out[k] = v
    return out


def check(e: dict) -> dict:
    title = e.get("title", "")
    # An arXiv preprint is checked against arXiv, which is the record.
    if e.get("eprint"):
        a = arxiv_abs(e["eprint"])
        sim = SequenceMatcher(None, norm(title), norm(a.get("title", ""))).ratio()
        fam = norm(a["authors"][0].split(",")[0]) if a["authors"] else ""
        problems = []
        if sim < 0.9:
            problems.append(f"title ({sim:.2f}): {a.get('title', '')[:70]}")
        if fam and fam != first_family(e.get("author", "")):
            problems.append(f"first author: {fam} vs {first_family(e.get('author', ''))}")
        yr = int(a.get("date", "0")[:4] or 0)
        if yr and int(e.get("year", 0) or 0) != yr:
            problems.append(f"year: {yr} vs {e.get('year')}")
        return {"key": e["key"], "status": "ok" if not problems else "CHECK", "sim": sim,
                "doi": f"arXiv:{e['eprint']}", "year": yr, "problems": problems, "has_doi": True}
    # A DOI is an exact key, so it is the record to compare against when present;
    # a title search is the fallback for everything else, taking the best
    # *title* match among the top hits because Crossref's relevance order puts
    # an IFAC abstract above the journal paper with the identical title.
    if e.get("doi"):
        hits = [crossref_doi(e["doi"])]
    else:
        hits = crossref(title, rows=5)
    if not hits:
        return {"key": e["key"], "status": "no match"}

    def _sim(h):
        return SequenceMatcher(None, norm(title), norm((h.get("title") or [""])[0])).ratio()
    h = max(hits, key=_sim)
    ht = (h.get("title") or [""])[0]
    sim = _sim(h)
    hyr = (h.get("issued", {}).get("date-parts") or [[None]])[0][0]
    yr = int(e.get("year", 0) or 0)
    fam = norm((h.get("author") or [{}])[0].get("family", ""))
    venue = (h.get("container-title") or [""])[0]
    mine_venue = e.get("journal") or e.get("booktitle") or ""
    problems = []
    if sim < 0.9:
        problems.append(f"title ({sim:.2f}): {ht[:70]}")
    # Crossref sometimes files "Masahiro Ono" whole in the family field, so a
    # match is the surname appearing in it, not string equality
    author_ok = not fam or first_family(e.get("author", "")) in fam.split()
    if not author_ok:
        problems.append(f"first author: {fam} vs {first_family(e.get('author', ''))}")
    # online-first can put Crossref's "issued" three years before the volume
    if hyr and yr and abs(hyr - yr) > 3:
        problems.append(f"year: {hyr} vs {yr}")
    if h.get("volume") and e.get("volume") and h["volume"] != e["volume"]:
        problems.append(f"volume: {h['volume']} vs {e['volume']}")
    hp = h.get("page") or h.get("article-number") or ""
    if hp and e.get("pages") and norm(hp) != norm(e["pages"].replace("--", "-")):
        problems.append(f"pages: {hp} vs {e['pages']}")
    status = "ok" if sim >= 0.9 and not problems else "CHECK"
    if status == "CHECK" and not e.get("doi") and sim < 0.9 and not author_ok:
        # A different title by a different author is not a disagreement about
        # this entry; it is Crossref having nothing for it -- a NeurIPS/ICML
        # paper, most likely. "Verify elsewhere", not "wrong". The case that
        # stays flagged is the dangerous one: the same title on other authors.
        status = "unindexed"
        problems = []
    return {"key": e["key"], "status": status,
            "sim": sim, "doi": h.get("DOI"), "year": hyr, "venue": venue[:50],
            "mine_venue": mine_venue[:50], "problems": problems, "has_doi": bool(e.get("doi"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    entries = parse(BIB.read_text())
    if args.key:
        e = next(x for x in entries if x["key"] == args.key)
        print(json.dumps(crossref(e["title"], rows=3), indent=1)[:6000])
        return
    for e in entries:
        try:
            r = check(e)
        except Exception as ex:  # network hiccup: say so and move on
            r = {"key": e["key"], "status": "error", "problems": [str(ex)]}
        flag = {"ok": " ", "CHECK": "!", "no match": "?", "error": "x", "unindexed": "~"}[r["status"]]
        line = f"{flag} {r['key']:<26} {r.get('year') or '':<5} {str(r.get('doi') or '')[:38]:<38}"
        if r["status"] == "ok" and not r.get("has_doi"):
            line += "  (no doi in bib)"
        print(line)
        for pr in r.get("problems", []):
            print(f"      - {pr}")
        time.sleep(0.6)


if __name__ == "__main__":
    main()
