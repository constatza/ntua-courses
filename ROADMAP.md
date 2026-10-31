# Roadmap — cross-repo

Done: course repos with shared build/release/CI tooling and prek hooks in `beamer-ntua`, topics mapped, meta-repo with submodules, setup docs.

## Next
1. Test the documented setup from a clean machine or container: clone with `--recurse-submodules`, run `scripts/bootstrap.sh`, run `build-course` in each course.
2. Review `_archive/_pre-reorg/` (old trees kept for safety), then delete it. Extend `_archive/MANIFEST.sha256` to cover the `admin/` folder if wanted.
3. Refresh the pinned `texlive/texlive:latest-medium` digest in `beamer-ntua/.github/workflows/course-ci.yml` when a course needs a newer TeX Live (`docker pull`, then `docker inspect`), and tag a new `beamer-ntua` release. Then bump the tag in each course's `ci.yml` and `.pre-commit-config.yaml` together (`course-layout` fails if they differ).
4. When a third course exists, extract a `course-template` GitHub template repo (thin `ci.yml`, `.gitignore`, `AGENTS.md`, one sample deck).

## Open questions
- `pdf-distances` is in `ml-course` (ML-engineering applications) but could belong in `stochastic-course` (KDE, optimal transport are statistics); changing it means moving the directory and updating both `TOPICS.toml` files.
- `beamer-ntua/scripts/install-class.ps1` does not install the course tools yet (Windows).
