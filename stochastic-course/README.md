# stochastic-methods
Decks live in `presentations/<slug>/main.tex` (LaTeX + `assets/` + optional `src/` Python generators run with `uv run`),
exercises in `exercises/`. Topic ownership: `TOPICS.toml`. Build everything: `tools/build-all`.
Needs the [beamer-ntua](https://github.com/constatza/beamer-ntua) clone as a sibling `../beamer-ntua` (or `NTUA_BEAMER_HOME`);
copy `beamer-ntua/templates/deck.latexmkrc` to a new deck's `.latexmkrc`. Books, slides, datasets, assignments live outside git in `../_archive`.
`sampling` still uses its original metropolis theme (needs lualatex) (not yet ported to `ntuabeamer`).
