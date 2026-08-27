# The write-up

`main.tex` is the technical report. It is the artefact for the novelty question:
it states what is *not* new first, positions against prior art with citations,
and then makes three measurements.

## No number in it is typed by hand

`tables/*.tex` are generated. `main.tex` `\input{}`s them, so the update path is:

```bash
python eval/comparative.py --arms climate office demographic national
python eval/horizon_risk.py --building Fox_office_Gaylord
python eval/paper_tables.py        # regenerates every table in the paper
```

If a table in the PDF disagrees with `results/`, the paper is stale, not the
results. That is deliberate and it is the same contract the rest of the repo
runs under.

## Compiling

There is no TeX toolchain in this environment and installing one costs a couple
of gigabytes, so the source is written to compile elsewhere.

**Overleaf** (what you almost certainly want): upload `main.tex`, `refs.bib` and
the `tables/` folder, keeping the directory structure. Set the compiler to
pdfLaTeX. It needs `booktabs`, `natbib`, `hyperref`, `microtype`, `amsmath` and
`geometry`, all of which Overleaf has by default.

**Locally**, if you ever want it:

```bash
sudo apt install texlive-latex-recommended texlive-bibtex-extra
cd docs/paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## What is deliberately not in it

The closed-loop acceptance test for the scenario MPC. That section
measures the horizon gap and describes the copula fix as conservative and
directionally right, and explicitly declines to claim the problem is solved
until that test lands. If it passes, that becomes a fourth contribution and the
abstract changes. If it fails, the paper is still correct as written.
