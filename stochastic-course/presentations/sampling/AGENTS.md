# Repository Guidelines

## Project Structure & Module Organization
The Beamer deck lives at `sampling-techniques.tex` alongside its compiled artifacts (`.pdf`, `.aux`, `.log`, etc.). Python sampling demos are grouped under `scripts/`, one module per technique, and write both PNG and GIF outputs to `figures/`. Keep generated media versioned for reproducibility; regenerate assets via the scripts instead of editing them manually. Font configuration depends on Libertinus families, so ensure they are installed before compiling the slides.

## Build, Test, and Development Commands
- `latexmk -lualatex sampling-techniques.tex` &mdash; rebuild the presentation using LuaLaTeX/Fontspec. Run from the repository root; add `-pvc` during longer editing sessions.
- `latexmk -c` &mdash; clean auxiliary files after a build.
- `uv run python scripts/acceptance_rejection_pi.py` (and the analogous scripts for importance sampling and Metropolis--Hastings) &mdash; regenerate figures and animations in `figures/`. Always execute Python entries through `uv run` so the pinned dependency set is respected; ensure your `uv` project targets Python 3.10 with `numpy`, `matplotlib`, and `imageio` available.

## Coding Style & Naming Conventions
Python modules follow PEP 8 with four-space indentation, type annotations, frozen dataclasses for results, and pure helper functions above the `main()` entry point. Use descriptive file names that mirror existing patterns (e.g., `ars_pi_final.png`, `<method>_<topic>_{final|evolution}.gif`). Prefer f-strings, NumPy vectorization, and explicit RNG seeds passed through `np.random.default_rng`. Keep plotting aesthetics consistent with the existing color palette and layout (paired subplots, readable font sizes, tight layout).

## Testing Guidelines
There is no automated test suite yet; treat each script execution as a smoke test. Verify that regenerated media appears in `figures/`, matches the expected filenames, and that GIFs play smoothly. When adjusting stochastic logic, inspect summary statistics (acceptance rates, ESS, running means) and add quick assertions or logging while iterating. If you introduce reusable utilities, consider wrapping them with lightweight `pytest` cases under a future `tests/` directory.

## Commit & Pull Request Guidelines
Use concise, imperative commit subjects that describe the change scope (e.g., `Update IS demo weighting diagnostics`). Group code and artifact updates in a single commit so outputs stay tied to their generators. Pull requests should include: a short overview, references to related slides/frames, regenerated figure thumbnails if visuals changed, and notes on toolchain versions (`uv` + Python, LaTeX engine) used. Surface any deviations from deterministic seeds or asset naming so reviewers can reproduce the results quickly.
