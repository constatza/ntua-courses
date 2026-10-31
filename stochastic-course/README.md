# Stochastic Methods — NTUA

Lecture decks and notes for the stochastic methods course at the NTUA School of Civil Engineering.

## Get the PDFs

All slides and notes are published as PDFs on the repository's releases page, filtered to this course:

**[ntua-courses releases — stochastic-course](https://github.com/constatza/ntua-courses/releases?q=stochastic-course)**

Pick the newest `stochastic-course/v…` release and download whichever PDFs you need; no clone or LaTeX is required. Files are named by kind and topic, for example `presentation_sampling.pdf` and `note_model-selection.pdf`.

## Suggested reading order

1. Sampling — acceptance-rejection, importance sampling/ESS, MCMC, Metropolis–Hastings (presentation)
2. Bayesian inference — Bayes rule, likelihood from forward model, analytic posteriors, MH verification (note)
3. Model selection — Bayesian model selection (note)

## Maintaining the course

Building, releasing and editing this course is covered in the [repository README](../README.md). The `sampling` deck is built with LuaLaTeX. Topic ownership is tracked in [`TOPICS.toml`](TOPICS.toml): each concept has one owning deck or note, and others cross-reference it instead of re-explaining. Books, lecture notes, assignments and grading material are not in this repository.
