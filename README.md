# ntua-courses
Meta-repo that pins independent repos as submodules. Each one builds on its own; you do not need all of them.

| Submodule | What | Use it alone |
|---|---|---|
| `beamer-ntua` | Beamer class `ntuabeamer` (own repo) | its README |
| `machine-learning` | decks, notes, exercises | its README → Setup |
| `stochastic-methods` | decks, notes | its README → Setup |

## Everything together
```sh
git clone --recurse-submodules <this-repo> ntua-courses && cd ntua-courses
sh beamer-ntua/scripts/install-class.sh      # once; writes to TEXMFHOME
machine-learning/tools/build-all
stochastic-methods/tools/build-all
```

## Just one course
```sh
git clone <this-repo> ntua-courses && cd ntua-courses
git submodule update --init machine-learning   # only what you ask for
```
or clone that course's repo directly and follow its Setup section.

Non-source material (books, slides, datasets, grades) lives in an untracked `_archive/` and is not part of any repo.
