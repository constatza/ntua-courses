# ntua-courses
Meta-repo that pins independent repos as submodules. Each one builds on its own; you do not need all of them.

| Submodule | What | Use it alone |
|---|---|---|
| `beamer-ntua` | Beamer class `ntuabeamer` (own repo) | its README |
| `ml-course` | decks, notes, exercises | its README → Setup |
| `stochastic-course` | decks, notes, exercises | its README → Setup |

## Everything together
```sh
git clone --recurse-submodules <this-repo> ntua-courses && cd ntua-courses
sh scripts/bootstrap.sh                      # submodules + class + build-course (or: sh beamer-ntua/scripts/install-class.sh)
(cd ml-course && build-course)
(cd stochastic-course && build-course)
```

## Just one course
```sh
git clone <this-repo> ntua-courses && cd ntua-courses
git submodule update --init ml-course   # only what you ask for
```
or clone that course's repo directly and follow its Setup section.

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
