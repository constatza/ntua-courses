# machine-learning
Decks live in `presentations/<slug>/main.tex` (LaTeX + `assets/` + optional `src/` Python generators run with `uv run`),
exercises in `exercises/`. Topic ownership: `TOPICS.toml`.
All decks use the `ntuabeamer` class.
Books, slides, datasets and assignments are not in this repo.

## Setup (this repo only)
Requires TeX Live with `latexmk`, and [`uv`](https://docs.astral.sh/uv/) for the Python demos.
Install the `ntuabeamer` class before building:

```sh
git clone <this-repo> && cd machine-learning
curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh   # once; writes to TEXMFHOME
tools/build-all                  # or: cd presentations/<deck> && latexmk -pdf main.tex
```

To pin a template version, set `NTUA_BEAMER_REF=<tag>` for the install script. Each deck builds into its own `.build/`.
