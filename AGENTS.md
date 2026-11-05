# Agent guidelines — ntua-courses

One repository holds the courses (`ml-course/`, `stochastic-course/`), the shared tooling (`tools/`) and the `ntuabeamer` class (`beamer-ntua/`). Each course has its own `AGENTS.md`, `README.md` and `ROADMAP.md`; follow the guidelines of the course you are working in.

- Build and verify with `tools/build-course [course]`; every target must report `ok`. Run `prek run --all-files` before committing.
- `beamer-ntua/` is a one-way copy of its public repository. Never edit it here and never `git subtree push`; fix the class in its own repository and `git subtree pull`.
- Tooling changes in `tools/` affect every course; keep `tests/tools-smoke.sh` passing.
- Releases are per-course tags: `<course>/vX.Y.Z`.
- Cross-course work items live in `ROADMAP.md`; course-specific items live in that course's `ROADMAP.md`.
- `_archive/` (books, slides, datasets, assignments, grades, student submissions) is untracked. Never add it to git.
- Variants: a translation or other variant of a target is a sibling `main-<variant>.tex` (e.g. `main-greek.tex`); answers have none. Mirror every change to `main.tex` into its variants in the same commit; `tools/check-variants` enforces matching questions, sections, labels and links. Built as `<kind>_<slug>_<variant>.pdf`.
- Commits: Conventional Commit header, bulleted body, no AI attribution.
