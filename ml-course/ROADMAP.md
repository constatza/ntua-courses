# Roadmap — machine-learning

1. Port `neural-nets` and `pdf-distances` to the `ntuabeamer` class (they still use plain beamer and the Madrid theme); preserve content, one deck per commit.
2. Run every Python generator in its new location (`generative/src`, `pca-rom/src/pca_rom_mre.py`, `neural-nets/src/skorch_example.py`) and confirm outputs land in `assets/`; only the `stochastic-methods` Bayes generator has been verified so far. The generative exercise generators now write to `exercises/toy-diffusion-flow/assets/generated`.
3. Move `pca-rom/pca-rom.xopp` into `assets/` (it sits in the deck root).
4. `neural-nets`: keep the "advanced architectures" frame as a signpost to `gnns` and `generative` only (topic owner stays `neural-nets`).
5. Write the LSTM mechanical-system material (assignment is in `../_archive/machine-learning/assignments`): solution code under `exercises/lstm-mechanical-system/`, optional deck; then register it in `TOPICS.toml` (currently only attributed to `neural-nets`).
6. Decide whether `pdf-distances` stays here (see parent `ROADMAP.md`).
