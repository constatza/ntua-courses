# Stochastic Methods — NTUA

Lecture decks and notes for the stochastic methods course at NTUA School of Civil Engineering.
The `sampling` deck uses the `ntuabeamer` class and needs `lualatex`.

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

Requires TeX Live with `latexmk` and `lualatex` (sampling deck),
and [`uv`](https://docs.astral.sh/uv/) for the Python demos.

```sh
git clone <ntua-courses repo> && cd ntua-courses
tools/build-course stochastic-course      # PDFs end up in stochastic-course/dist/
```

This course lives in the `ntua-courses` repository next to the shared `tools/` and the `ntuabeamer` class, so nothing needs installing. CI runs the same build in a pinned TeX Live image. Pushing the tag `stochastic-course/vX.Y.Z` publishes that course's PDFs as a GitHub release (answer PDFs are withheld). See the repository README for details.

Each target builds into its own `.build/`. Built PDFs are not committed to git.
