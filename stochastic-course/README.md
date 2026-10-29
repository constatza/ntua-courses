# stochastic-methods
Decks live in `presentations/<slug>/main.tex` (LaTeX + `assets/` + optional `src/` Python generators run with `uv run`),
notes in `notes/`. Topic ownership: `TOPICS.toml`.
The `sampling` deck uses the `ntuabeamer` class and needs lualatex.
Books, lecture notes, assignments and grading material are not in this repo.

## Setup (this repo only)
Requires TeX Live with `latexmk`, `lualatex` and the Libertinus fonts (sampling deck), and [`uv`](https://docs.astral.sh/uv/) for the Python demos.
Install the `ntuabeamer` class before building:

```sh
git clone <this-repo> && cd stochastic-methods
curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh   # once; writes to TEXMFHOME
tools/build-all                  # or: cd presentations/<deck> && latexmk main.tex
```

The install script respects `NTUA_BEAMER_REF`; this course is pinned to `v0.1.0-alpha.1`:

```sh
NTUA_BEAMER_REF=v0.1.0-alpha.1 curl -fsSL https://raw.githubusercontent.com/constatza/beamer-ntua/main/scripts/install-class.sh | sh
```

Each deck builds into its own `.build/`.
