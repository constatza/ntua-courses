# Roadmap — cross-repo

Done: course repos with shared build/release/CI tooling in `beamer-ntua`, topics mapped, meta-repo with submodules, setup docs.

## Next
1. Push `beamer-ntua` (`main` and tag `v0.2.0`) so the course CI can resolve the reusable workflow; push each course next, then this repo.
2. Rename the GitHub repos `machine-learning` → `ml-course` and `stochastic-methods` → `stochastic-course`; add the topics `course`, `ntua` and pin them on the profile. `.gitmodules` already points at the new names.
3. Test the documented setup from a clean machine or container: clone with `--recurse-submodules`, run `scripts/bootstrap.sh`, run `build-course` in each course.
4. Push a `v0.1.0` tag in one course on a fork and confirm the draft-then-publish release (see `beamer-ntua/scripts/release-course`).
5. Review `_archive/_pre-reorg/` (old trees kept for safety), then delete it. Extend `_archive/MANIFEST.sha256` to cover the `admin/` folder if wanted.
6. Refresh the pinned `texlive/texlive:latest-medium` digest in `beamer-ntua/.github/workflows/course-ci.yml` when a course needs a newer TeX Live (`docker pull`, then `docker inspect`), and tag a new `beamer-ntua` release.
7. When a third course exists, extract a `course-template` GitHub template repo (thin `ci.yml`, `.gitignore`, `AGENTS.md`, one sample deck).

## Open questions
- `pdf-distances` is in `ml-course` (ML-engineering applications) but could belong in `stochastic-course` (KDE, optimal transport are statistics); changing it means moving the directory and updating both `TOPICS.toml` files.
- `beamer-ntua/scripts/install-class.ps1` does not install the course tools yet (Windows).
