# Roadmap — machine-learning

1. Run every Python generator in its new location (`generative/src`, `pca-rom/src/pca_rom_mre.py`, `neural-nets/src/skorch_example.py`) and confirm outputs land in `assets/`; only the `stochastic-methods` Bayes generator has been verified so far. The generative exercise generators now write to `exercises/toy-diffusion-flow/assets/generated`.
2. Move `pca-rom/pca-rom.xopp` into `assets/` (it sits in the deck root).
3. `neural-nets`: keep the "advanced architectures" frame as a signpost to `gnns` and `generative` only (topic owner stays `neural-nets`).
4. Write the LSTM mechanical-system material (assignment is in `../_archive/machine-learning/assignments`): solution code under `exercises/lstm-mechanical-system/`, optional deck; then register it in `TOPICS.toml` (currently only attributed to `neural-nets`).
5. Decide whether `pdf-distances` stays here (see parent `ROADMAP.md`).
