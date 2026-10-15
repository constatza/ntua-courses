# Roadmap — stochastic-methods

1. Overlap trims recorded in `TOPICS.toml`: shorten the sampling deck's Bayes intro frames to a recap that cites the Bayes note; replace the note's Metropolis–Hastings derivation with a pointer to the sampling deck (keep the verification examples).
2. Fold `notes/model-selection` into `notes/bayesian-inference-mh` as a section, then update `TOPICS.toml`.
3. Run the sampling generators (`presentations/sampling/run.sh`) in the new location and confirm PNGs land in `assets/generated/`. The new build is 37 pages while the old `sampling-techniques.pdf` had 4; check the old file was a stale or handout build.
4. Topics with no deck yet, covered only by the archived lecture notes: probability, stochastic processes, KL/spectral/PCE expansions, SSFEM, reliability, sensitivity. Write decks as needed and register owners in `TOPICS.toml`.
5. Create `exercises/` for our own solutions (assignments and MATLAB code stay in `../_archive/stochastic-methods`).
6. `notes/bayesian-inference-mh/src/bayesian_update.py` came from the old top-level `code/`; check it still runs.
