# ntua-courses
Meta-repo that pins independent repos as submodules. Each one builds on its own; you do not need all of them.

| Submodule | What | Use it alone |
|---|---|---|
| `ml-course` | decks, notes, exercises | its README → Setup |
| `stochastic-course` | decks, notes, exercises | its README → Setup |

## Everything together
```sh
git clone --recurse-submodules <this-repo> ntua-courses && cd ntua-courses
sh scripts/bootstrap.sh                      # submodules, then the class and tools at the tag the courses pin
(cd ml-course && build-course)
(cd stochastic-course && build-course)
```

## Just one course
```sh
git clone <this-repo> ntua-courses && cd ntua-courses
git submodule update --init ml-course   # only what you ask for
```
or clone that course's repo directly and follow its Setup section.

## Local checks
Each repo has a `.pre-commit-config.yaml` for [prek](https://github.com/j178/prek). Run `prek install -t pre-commit -t commit-msg` once per clone. The course-specific hooks (layout, no built PDFs, no AI attribution) are defined once in `beamer-ntua` and referenced by tag, so all courses run the same ones.

`beamer-ntua` (the `ntuabeamer` class, build and release scripts, shared CI and hooks) is an independent repo that the courses consume by tag. It is not tracked here; see its [README](https://github.com/constatza/beamer-ntua). Course content tags are separate from its tags.

## Committing
Child first, then parent. Where you run `git commit` decides which repo is committed.
```sh
cd ml-course && build-course && git add <paths> && git commit   # work and commit in the course repo
cd .. && git add ml-course && git commit -m "chore(repo): bump ml-course"   # optional snapshot in the parent
```
- The parent stores only a pointer to a commit of each child; bump it when you want a snapshot of versions that go together.
- Push the child before the parent, or other clones will point at a commit that is not on GitHub.
- `git submodule status` shows the pinned commits; a leading `+` means the child is ahead of the pin.
- Commit messages: Conventional Commit header (`type(scope): summary`) with a bulleted body.

Non-source material (books, slides, datasets, grades) lives in an untracked `_archive/` and is not part of any repo.
