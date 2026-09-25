# Elsevier version — for Applied Energy or Energy and Buildings

`elsarticle` in `preprint` style, which is what Elsevier wants at submission:
single column, generously leaded, 40 pages. That is not the published length —
rebuild with `[3p,twocolumn,authoryear]` and it is 22, which is the figure to
judge it by.

Same body as the report and the IEEE build; all three `\input`
`../paper/sections/*.tex`. This file owns the class, Elsevier's front matter
(highlights, keywords, structured affiliations) and single-column figure
placement.

    ../../.venv/bin/python eval/paper_tables.py --fit --out docs/paper_elsevier/tables
    cd docs/paper_elsevier && pdflatex main && bibtex main && pdflatex main && pdflatex main

`--fit` scales each tabular to the measure: Elsevier's text block is narrower
than the report's a4 one, so tables that fit there overrun here by a few picas.

## Why this exists

The paper is 18 pages in IEEE Transactions format. Transactions on Smart Grid
allows 10 before over-length charges, and cutting the cross-country sections
saves only two pages, because their tables float and stay. A paper of this
length wants a journal that takes papers of this length: Applied Energy is the
closest fit by audience and takes them, and Energy and Buildings is the
fallback. `\journal{}` at the top of `main.tex` needs confirming either way.
