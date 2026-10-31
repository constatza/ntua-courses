# Roadmap — cross-course

Done: courses, tooling and CI in one repository; the class is a one-way subtree copy; per-course prefixed release tags; local and CI checks via prek.

## Next
1. Rebase or merge the pushed branch `feat/numbered-neural-curriculum` (a rebuilt neural-network curriculum from the old `ml-course` repo, 2 commits). It predates the `presentations/generative/` split, so expect conflicts in the decks it touches.
2. Review `_archive/_pre-reorg/` and `_archive/old-releases/` (the old `v0.1.0-alpha.1` PDFs) and `_archive/old-ml-course-all-branches.bundle` (full backup of the deleted `ml-course` repo) (old trees kept for safety), then delete it. Extend `_archive/MANIFEST.sha256` to cover the `admin/` folder if wanted.
3. Optional: Renovate for the TeX Live image digest and hook revs; branch protection with required CI checks; a backup mirror remote for this repository.
4. Optional: scope CI to changed courses (path filters) once build time hurts.

## Open questions
- `pdf-distances` is in `ml-course` (ML-engineering applications) but could belong in `stochastic-course` (KDE, optimal transport are statistics); changing it means moving the directory and updating both `TOPICS.toml` files.
- Whether a GitLab pipeline is ever needed; the README sketch is untested.
