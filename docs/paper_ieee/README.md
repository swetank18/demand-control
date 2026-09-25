# Conference version

Six-page IEEE conference cut of the technical report in `../paper/`. Scope:
Contributions 1 and 2 — what the marginal constraint delivers over a horizon,
and how to make that level settable — with the calibration work kept only
where Contribution 1 leans on it.

It is **not** a separate document in the sense that matters: the tables come
from the same generator as the report's, so no number in it is typed by hand
and none can drift from the study.

    ../../.venv/bin/python eval/paper_tables.py --ieee   # tables/*.tex
    # figures and refs.bib are the report’s, referenced not copied
    cd docs/paper_ieee && pdflatex main && bibtex main && pdflatex main && pdflatex main

Two things need the author before submission, both marked `TODO(author)` in
`main.tex`: the affiliation block, and the repository URL in the
reproducibility note. The page limit is a `\documentclass` matter — this
builds to 4 pages against the 6 most IEEE conferences allow, so a venue with
a different limit needs no restructuring.
