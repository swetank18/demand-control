# Elsevier version — the Applied Energy submission (chosen 2026-10-02)

`elsarticle` in `preprint` style, which is what Elsevier wants at submission:
single column, generously leaded, ~43 pages. That is not the published length —
rebuild with `[3p,twocolumn]` for an estimate of that. References are numbered
(`elsarticle-num-names`, so `\citet` still prints the authors' names), which is
Applied Energy's style.

Same body as the report and the IEEE build; all three `\input`
`../paper/sections/*.tex`. This file owns the class, Elsevier's front matter
(highlights, keywords, structured affiliations) and single-column figure
placement.

    ../../.venv/bin/python eval/paper_tables.py --fit --out docs/paper_elsevier/tables
    cd docs/paper_elsevier && pdflatex main && bibtex main && pdflatex main && pdflatex main
    ../../.venv/bin/python eval/paper_bundle.py --build elsevier   # -> ~/hackit/paper_elsevier/, Overleaf zip

Applied Energy's front-matter rules this file is held to: 3–5 highlights of at
most 85 characters each; an abstract within the journal's word limit; 4–6
keywords; and the data-availability, CRediT, competing-interest, generative-AI
and funding statements after the body. The statements only the author can make
print as red **TODO(author)** so none can be submitted blank.

`--fit` scales each tabular to the measure: Elsevier's text block is narrower
than the report's a4 one, so tables that fit there overrun here by a few picas.

## Why this exists

The paper is 18 pages in IEEE Transactions format. Transactions on Smart Grid
allows 10 before over-length charges, and cutting the cross-country sections
saves only two pages, because their tables float and stay. A paper of this
length wants a journal that takes papers of this length: Applied Energy is the
closest fit by audience and takes them, and Energy and Buildings is the
fallback.
