# machine-learning
Decks live in `presentations/<slug>/main.tex` (LaTeX + `assets/` + optional `src/` Python generators run with `uv run`),
exercises in `exercises/`. Topic ownership: `TOPICS.toml`. Build everything: `tools/build-all`.
Template decks need the `ntuabeamer` class installed (github:constatza/beamer-ntua, `scripts/install-class.sh`).
Books, slides, datasets and assignments live outside git in `../_archive`.
`neural-nets` and `pdf-distances` still use their original beamer themes (not yet ported to `ntuabeamer`).
