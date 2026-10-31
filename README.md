# ntua-courses

All NTUA courses, their build tooling and CI in one repository. Each course is a directory of decks, notes and exercise sheets; the shared `tools/` and the `ntuabeamer` class (`beamer-ntua/`) live next to them, so every commit is a complete, buildable snapshot and tooling can never drift from content.

| Path | What |
|---|---|
| `ml-course/` | machine learning: decks, notes, exercises |
| `stochastic-course/` | stochastic methods: decks, notes, exercises |
| `tools/` | `build-course`, `release-course`, `check-course-layout`, `ci`, `courses` |
| `beamer-ntua/` | the Beamer class, a one-way copy of [constatza/beamer-ntua](https://github.com/constatza/beamer-ntua) |

## Build
Requires TeX Live with `latexmk` (and `lualatex` for the `sampling` deck) and [`uv`](https://docs.astral.sh/uv/) for the Python demos. Nothing is installed globally; the class is found through `TEXINPUTS` inside this repository.
```sh
git clone <this-repo> ntua-courses && cd ntua-courses
tools/build-course                  # every course; PDFs end up in <course>/dist/
tools/build-course ml-course        # one course
```
A failed target leaves no `dist/` and no stale PDF behind, and prints the tail of its LaTeX log.

## Just one course
`git sparse-checkout` gives a single-course view (it is a convenience, not an access boundary): `git clone --filter=blob:none --sparse <this-repo> && cd ntua-courses && git sparse-checkout set ml-course tools beamer-ntua`.

## Checks
`prek install -t pre-commit -t commit-msg` once per clone ([prek](https://github.com/j178/prek) runs the standard `.pre-commit-config.yaml`). Third-party hooks (ruff, shellcheck, actionlint, conventional commits) are remote; the course checks (layout, no built PDFs, no AI attribution) are local scripts. CI runs the same hooks plus the build.

## Releases
Content is versioned per course with path-prefixed tags: `git tag ml-course/v1.2.0 && git push origin ml-course/v1.2.0`. CI builds only that course and publishes its PDFs as a release (answer PDFs are withheld unless `RELEASE_ANSWERS=1`). `tools/release-course` is safe to re-run after a failure: a stale draft is replaced, an already-published tag is refused. Older per-repo releases (`v0.1.0-alpha.1`) stay in the archived `ml-course` and `stochastic-course` repositories.

## Updating the class
`beamer-ntua/` is ordinary files, synced one way from its public repository, which stays the source of truth:
```sh
git subtree pull --prefix=beamer-ntua git@github.com:constatza/beamer-ntua.git main --squash
```
Never `git subtree push`. Make class fixes in the class repository and pull them here.

## TeX Live
CI builds inside `texlive/texlive:latest@sha256:…` (full TeX Live 2026, pinned by digest in `.github/workflows/ci.yml`), so the build needs no network and cannot change under you. To move to a newer TeX Live: `docker pull texlive/texlive:latest`, copy the digest from `docker inspect`, update `ci.yml`, and run `tools/ci` in that image.

## Adding a course
Copy an existing course directory, rename it, adjust its `README.md`, `AGENTS.md`, `ROADMAP.md` and `TOPICS.toml`. `tools/courses` discovers it automatically (any top-level directory with `presentations/`, `notes/` or `exercises/`), and the `<course>/v*` tag convention works without editing CI.

## Access and splitting
Access is all-or-nothing for the whole repository. If a co-lecturer, TA or student needs only one course, or one course should become public, split it out with `git subtree split` or `git filter-repo`.

Non-source material (books, slides, datasets, grades) lives in an untracked `_archive/` and is not part of the repository.
