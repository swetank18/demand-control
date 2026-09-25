# Journal version — the submission document

IEEEtran two-column, currently 18 pages. Built on the **same body** as the
technical report: both `\input` the section files in `../paper/sections/`, so
the prose has one source and the two documents cannot disagree. What differs
is the class, the front matter a journal expects (nomenclature, index terms,
running head), the sectioning — four of the report's sections are demoted to
subsections to give a Transactions reader the structure they expect — and the
tables, which are emitted two-column.

    ../../.venv/bin/python eval/paper_tables.py --ieee --out docs/paper_journal/tables
    cd docs/paper_journal && pdflatex main && bibtex main && pdflatex main && pdflatex main

Figures and `refs.bib` are the report's, referenced not copied.
`IEEEtranN.bst` is the natbib-aware IEEE style, which is what keeps the
report's `\citet{...}` sentences grammatical under numeric citation.

## Length

18 pages is long for IEEE Transactions on Smart Grid, whose regular-paper
allowance is 10 with over-length charges beyond it. Three ways out, in the
order they are worth considering:

1. **Cut to the two central contributions** — the horizon gap and the dial —
   and carry the cross-country calibration study only where those lean on it.
   That is roughly 11 pages and is the strongest version of the paper.
2. **Keep the full study and pay the over-length charge.**
3. **Submit somewhere the length is not a problem** — IEEE Access (no limit,
   open-access fee), Applied Energy or Energy and Buildings (Elsevier, so a
   different template, which the shared-sections layout makes cheap to add).

This is an editorial decision, not a formatting one, so it is left open here.
