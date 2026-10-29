# Stochastic Methods — NTUA

Lecture decks and notes for the stochastic methods course at NTUA School of Civil Engineering.
The `sampling` deck uses the `ntuabeamer` class and needs `lualatex` and the Libertinus fonts.

Topic ownership is tracked in `TOPICS.toml`: each concept has one owning deck or note,
and others cross-reference it instead of re-explaining.

Books, lecture notes, assignments and grading material are not in this repo.

## Slides and notes

All lecture decks and notes are published as PDFs on GitHub Releases:

[slides](https://github.com/constatza/stochastic-methods/releases)

Each item is a single PDF. Download whichever you need — no clone required.

## Suggested reading order

1. Sampling — acceptance-rejection, importance sampling/ESS, MCMC, Metropolis–Hastings (presentation)
2. Bayesian inference — Bayes rule, likelihood from forward model, analytic posteriors, MH verification (note)
3. Model selection — Bayesian model selection (note)

## For instructors & developers (clone & build)

Requires TeX Live with `latexmk`, `lualatex` and the Libertinus fonts (sampling deck),
and [`uv`](https://docs.astral.sh/uv/) for the Python demos.

```sh
git clone <this-repo> && cd stochastic-methods
curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh   # once; writes to TEXMFHOME
tools/build-all                  # or: cd presentations/<deck> && latexmk main.tex
```

Install the `ntuabeamer` class before building. The install script respects
`NTUA_BEAMER_REF`; this course is pinned to `v0.1.0-alpha.1`:

```sh
NTUA_BEAMER_REF=v0.1.0-alpha.1 curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh
```

Each deck builds into its own `.build/`. Built PDFs are not committed to git.
