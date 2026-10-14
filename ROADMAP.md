# Roadmap — cross-repo

Done: course repos created, topics mapped, `ntuabeamer` consumed as an installed class, meta-repo with submodules, setup docs.

## Next
1. Create GitHub repos `ntua-courses`, `machine-learning`, `stochastic-methods`; push each course first, then the parent.
2. Point the course submodules at their GitHub URLs (`git submodule set-url`); today `.gitmodules` uses local `./` paths and a fresh clone cannot resolve them.
3. Tag a release of `beamer-ntua`, then note the tag in each course README that uses the class (`NTUA_BEAMER_REF=<tag>`).
4. Test the documented setup from a clean machine or container: clone, run `install-class.sh`, run `tools/build-all`. The installer and the tag pinning have not been tested.
5. Review `_archive/_pre-reorg/` (old trees kept for safety), then delete it. Extend `_archive/MANIFEST.sha256` to cover the `admin/` folder if wanted.
6. Decide where publishable PDFs go (GitHub Releases or Pages); built PDFs are ignored in git.

## Open questions
- `pdf-distances` is in `machine-learning` (ML-engineering applications) but could belong in `stochastic-methods` (KDE, optimal transport are statistics); changing it means moving the directory and updating both `TOPICS.toml` files.
