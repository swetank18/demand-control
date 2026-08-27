"""Assemble a self-contained folder that uploads to Overleaf as-is.

The paper's source of truth stays in `docs/paper/`, inside the repo, where the
table generator writes and where git tracks it. This copies that into a
standalone `paper/` directory beside the repo and zips it, so the upload step is
"drag one thing into Overleaf" rather than "remember which files and what
structure".

Because it copies rather than edits, the two cannot silently disagree: rerun
this and the bundle is current. If you edit inside the bundle instead of inside
docs/paper, the next run overwrites your edit -- which is the correct direction
for a build artefact.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/paper"
DEFAULT_DEST = ROOT.parent / "paper"

UPLOAD_README = """# Upload this folder to Overleaf

Everything the paper needs is in here. Nothing else is required.

## The fastest route

Upload **paper.zip** directly: Overleaf's *New Project -> Upload Project* accepts
a zip and preserves the folder structure, which is what matters -- `main.tex`
expects to find the tables under `tables/`.

If you would rather upload loose files, upload all of them and keep `tables/` as
a folder rather than flattening it.

## Settings

- Compiler: **pdfLaTeX**
- Main document: **main.tex**

The packages used -- booktabs, natbib, hyperref, microtype, amsmath, geometry --
are all in Overleaf's default TeX Live installation, so nothing needs installing.

## Contents

| File | What it is |
| --- | --- |
| `main.tex` | the paper |
| `refs.bib` | bibliography, 10 entries |
| `tables/*.tex` | seven tables, all machine-generated from the run output |

## Regenerating

This folder is a build artefact. The source lives in the repository at
`demand-control/docs/paper/`, and the tables are written by
`eval/paper_tables.py` from `results/`. To refresh everything after a new run:

```bash
cd demand-control
python eval/paper_tables.py      # rewrite the tables from results/
python eval/paper_bundle.py      # rebuild this folder and paper.zip
```

Edit the paper in `demand-control/docs/paper/main.tex`, not here -- a rebuild
overwrites this copy.

## One thing to check before you submit anywhere formal

The bibliography entries are real, well-known papers, but the volume and page
numbers were written from memory rather than pulled from a citation database.
For a hackathon or a demo this is fine. For a submission, verify them.
"""


def build(dest: Path, make_zip: bool = True) -> None:
    if not (SRC / "main.tex").exists():
        raise SystemExit(f"no paper source at {SRC}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    shutil.copy2(SRC / "main.tex", dest / "main.tex")
    shutil.copy2(SRC / "refs.bib", dest / "refs.bib")

    tables = SRC / "tables"
    if tables.exists():
        shutil.copytree(tables, dest / "tables")

    # any figures the paper grows later come along automatically
    figs = SRC / "figures"
    if figs.exists():
        shutil.copytree(figs, dest / "figures")

    (dest / "README.md").write_text(UPLOAD_README)

    files = sorted(p for p in dest.rglob("*") if p.is_file())
    print(f"bundle -> {dest}")
    for p in files:
        print(f"  {p.relative_to(dest)}  ({p.stat().st_size:,} B)")

    if make_zip:
        zpath = dest / "paper.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                if p.name == "README.md":
                    continue          # upload instructions are not part of the project
                z.write(p, p.relative_to(dest))
        print(f"\nzip    -> {zpath}  ({zpath.stat().st_size:,} B)")
        print("        upload this to Overleaf: New Project -> Upload Project")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()
    build(args.dest, make_zip=not args.no_zip)


if __name__ == "__main__":
    main()
