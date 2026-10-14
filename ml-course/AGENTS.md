# Agent guidelines — machine-learning

Setup and build instructions are in `README.md`; do not duplicate them here. Remaining work is in `ROADMAP.md`.

## Layout
- One directory per deck: `presentations/<slug>/main.tex` (always `main.tex`), with `assets/` for hand-made figures, `assets/generated/` for figures produced by `src/`, and `.latexmkrc` that only sets `.build/`.
- `exercises/<slug>/` holds the exercise sheet and our own answers, never student work.
- Python generators live in the deck's `src/`, are run with `uv run`, and compute output paths from `__file__` (no hardcoded directories), with explicit RNG seeds.

## Topics
- `TOPICS.toml` names one owning deck per topic. Check it before adding content; cross-reference the owner instead of re-explaining, and update the registry when a topic moves.
- Do not change a deck's content while reorganizing; path-only edits unless the task says otherwise.

## Rules
- Verify with `tools/build-all`; every target must report `ok`.
- Never commit `.build/`, built PDFs, gifs, or anything from `../_archive` (books, slides, datasets, assignments, grades, student work). Do not copy that material into the repo.
- Commits: Conventional Commit header, bulleted body, no AI attribution.
