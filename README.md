# ntua-courses
Meta-repo that pins the course repos as submodules. Each course is independent and builds on its own;
template decks need the `ntuabeamer` class installed (see `beamer-ntua/scripts/install-class.sh`).
- `beamer-ntua` — Beamer class (own repo)
- `machine-learning`, `stochastic-methods` — presentations, notes, exercises
Non-source material (books, slides, datasets, grades) lives in an untracked `_archive/`.
Clone: `git clone --recurse-submodules <url> ntua-courses`
