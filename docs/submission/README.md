# Submitting this paper

Two documents build from one study:

| | path | what it is |
|---|---|---|
| Technical report | `docs/paper/main.tex` | the full study, ~23 pp, single column |
| Conference paper | `docs/paper_ieee/main.tex` | the six-page IEEE cut, currently 4 pp |

The conference paper reads its tables from `eval/paper_tables.py --ieee` and
its figures and bibliography from the report's directory, so no number in it
is typed by hand and the two cannot disagree.

## What still needs the author

1. **Affiliation and co-authors.** `docs/paper_ieee/main.tex`, marked
   `TODO(author)`. IEEE wants department, organisation, city, country and
   email per author. The report (`docs/paper/main.tex`) has the same block.
2. **Venue.** The conference build targets the six-page limit most IEEE
   conferences use and lands at four, so a shorter limit needs cutting rather
   than restructuring and a longer one needs nothing. Nothing else in the
   build depends on the venue.
3. **Repository URL** for the reproducibility note, once the repo is public.
   Marked `TODO(author)` in both documents.
4. **IEEE PDF eXpress** — the venue issues a conference ID; run both PDFs
   through it and submit the file it returns, not the local one.
5. **Copyright form** — after acceptance, through the venue's system.

## Before you submit

    ../.venv/bin/python -m pytest tests -q          # 98 tests
    ../.venv/bin/python eval/paper_tables.py        # report tables
    ../.venv/bin/python eval/paper_tables.py --ieee # conference tables
    ../.venv/bin/python eval/paper_figures.py
    ../.venv/bin/python eval/bibcheck.py            # every reference against its index
    ../.venv/bin/python eval/paper_bundle.py        # Overleaf bundle for the report

Then build each document three times (pdflatex, bibtex, pdflatex, pdflatex)
and check the log reports no overfull boxes and no undefined references.

## Abstract

`abstract_200w.md` is the 200-word version for the submission form.
