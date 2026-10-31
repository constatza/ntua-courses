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
git clone <this-repo> && cd stochastic-course
ref=$(sed -n 's/.*course-ci.yml@//p' .github/workflows/ci.yml)   # beamer-ntua tag this course is pinned to
curl -fsSL -o /tmp/install-class.sh "https://raw.githubusercontent.com/constatza/beamer-ntua/$ref/scripts/install-class.sh"
NTUA_BEAMER_REF=$ref sh /tmp/install-class.sh   # once: class -> TEXMFHOME, build-course/release-course -> ~/.local/bin
build-course                                    # PDFs end up in dist/
```

CI runs the same `build-course` through the shared workflow in beamer-ntua, so a local `ok` means a CI `ok`.
Pushing a `v*` tag publishes `dist/` as a GitHub release (answer PDFs are withheld).

Each target builds into its own `.build/`. Built PDFs are not committed to git.
