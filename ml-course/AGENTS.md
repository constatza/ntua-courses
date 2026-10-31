# Agent guidelines — ml-course

Setup and build instructions are in `README.md`; do not duplicate them here. Remaining work is in `ROADMAP.md`.

## Layout
- One directory per deck: `presentations/<slug>/main.tex` (always `main.tex`), with `assets/` for hand-made figures, `assets/generated/` for figures produced by `src/`, plus a one-line `.latexmkrc` only when the engine is not pdflatex (`$pdf_mode = 4;` for lualatex). `build-course` sets the output directory.
- `exercises/<slug>/` holds `main.tex` (the sheet), optional `answers.tex` (our solutions, never student work), and its own `src/` and `assets/generated/`. An exercise may cover several topics; list it under each in `TOPICS.toml`. Targets never read files from a sibling directory.
- Python generators live in the deck's `src/`, are run with `uv run`, and compute output paths from `__file__` (no hardcoded directories), with explicit RNG seeds.

## Topics
- `TOPICS.toml` names one owning deck per topic. Check it before adding content; cross-reference the owner instead of re-explaining, and update the registry when a topic moves.
- Do not change a deck's content while reorganizing; path-only edits unless the task says otherwise.

## Rules
- Verify with `build-course` (see README); every target must report `ok`.
- Never commit `.build/`, built PDFs, gifs, or anything from `../_archive` (books, slides, datasets, assignments, grades, student work). Do not copy that material into the repo.
- Commits: Conventional Commit header, bulleted body, no AI attribution.
