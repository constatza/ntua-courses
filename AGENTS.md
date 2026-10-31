# Agent guidelines — ntua-courses (meta-repo)

This repo only pins `beamer-ntua`, `ml-course` and `stochastic-course` as submodules. Each is independent with its own `AGENTS.md`, `README.md` and `ROADMAP.md`; work inside the child repo and follow its guidelines.

- Edit and commit in the child repo first; commit here only to move submodule pointers or to change this README, `AGENTS.md` and `ROADMAP.md`.
- Cross-repo work items live in `ROADMAP.md`. Course-specific items live in that course's `ROADMAP.md`; do not repeat them here.
- `_archive/` (books, slides, datasets, assignments, grades, student submissions) is untracked and outside every repo. Never add it to git.
- Do not assume a user wants every course: instructions must work for cloning a single repo.
- Commits: Conventional Commit header, bulleted body, no AI attribution.
