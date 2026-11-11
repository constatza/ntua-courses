#!/usr/bin/env -S uv run --script
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: jupyter,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.12.0", "keras==3.15.1", "matplotlib==3.11.2", "numpy==2.5.4"]
# ///

# %% [markdown]
# # Training Neural Networks in Python
#
# **A surrogate-modelling tutorial with Keras**
#
# *Data-Driven Methods in Engineering, NTUA School of Civil Engineering*
#
# This is the Python sibling of the MATLAB Live Script `nn_training_tutorial.m` in the same folder: same sections, same data.
#
# ## How to use this notebook
#
# Run it **cell by cell** and look at each result before moving on.
#
# - We build one neural-network surrogate and refine it step by step.
# - **Try it** questions ask you to change one line, rerun a cell, and explain what you see.
# - Networks are [Keras](https://keras.io/) models, computed by [JAX](https://docs.jax.dev/).
# - Utilities (the data split, plotting) are in collapsed cells; you do not need to read them to follow the text.
#
# To run it: `uv run nn_training_tutorial.py` as a script, or `tools/sync-notebooks` from the repository and open the `.ipynb`.

# %% jupyter={"source_hidden": true}
# Utilities: imports and problem description types.
from __future__ import annotations

import os
from dataclasses import dataclass, replace

os.environ["KERAS_BACKEND"] = "jax"  # must be set before Keras is imported

import keras
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Column:
    length: float  # m
    inertia: float  # m^4


@dataclass(frozen=True, slots=True)
class InputDistribution:
    e_mean: float  # Pa
    e_std: float  # Pa
    k_limits: tuple[float, float]


# %% [markdown]
# # Settings
#
# All knobs live here. Change one, rerun the cells below, and compare.
#
# - `SEED` fixes the random number generators, so every run is reproducible.
# - `EPOCHS`, `BATCH_SIZE`, `LEARN_RATE` and `HIDDEN` define the reference training run.

# %%
# Data
SEED = 20261111
N_SAMPLES = 1000
SPLIT = (0.6, 0.2, 0.2)  # train / validation / test fractions

# Reference training run
EPOCHS = 200
BATCH_SIZE = 32
LEARN_RATE = 1e-2
HIDDEN = 8  # neurons in the hidden layer

MC_SAMPLES = 10_000  # fresh samples for the Monte Carlo comparison

# The physical problem: fields are read as COLUMN.length, INPUTS.e_mean, ...
COLUMN = Column(length=4.0, inertia=5e-6)
INPUTS = InputDistribution(e_mean=200e9, e_std=10e9, k_limits=(0.6, 1.0))

# %% [markdown]
# # The problem: a buckling-load surrogate
#
# A **surrogate** is a cheap learned imitation of an expensive model. Here the "expensive model" is a formula, so we always know the right answer.
#
# ## The formula
#
# A slender column under axial compression buckles at the Euler critical load
#
# $$
# P_{cr} = \frac{\pi^2 E I}{(K L)^2}.
# $$
#
# | Symbol | Meaning | Value |
# |---|---|---|
# | $E$ | Young's modulus | uncertain |
# | $K$ | effective-length factor | uncertain, between 0.6 and 1.0 |
# | $L$ | length | 4 m |
# | $I$ | second moment of area | $5\times10^{-6}$ m$^4$ |
#
# `buckling_load` returns MN. The plots show the load against each input, with the other one fixed.


# %%
def buckling_load(e: Array | float, k: Array | float, column: Column = COLUMN) -> Array:
    """Euler critical load in MN."""
    # NumPy operators work element by element, so e and k may be whole arrays.
    return np.pi**2 * np.asarray(e) * column.inertia / (np.asarray(k) * column.length) ** 2 / 1e6


k_line = np.linspace(*INPUTS.k_limits, 100)  # *k_limits unpacks the tuple into (start, stop)
e_line = np.linspace(INPUTS.e_mean - 3 * INPUTS.e_std, INPUTS.e_mean + 3 * INPUTS.e_std, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))  # axes[0] and axes[1] are the two panels

axes[0].plot(k_line, buckling_load(INPUTS.e_mean, k_line), lw=1.5)
axes[0].set(xlabel="K", ylabel="$P_{cr}$ (MN)", title="E = 200 GPa")  # $...$ renders as math
axes[0].grid(True)

axes[1].plot(e_line / 1e9, buckling_load(e_line, 0.8), lw=1.5)  # E plotted in GPa
axes[1].set(xlabel="E (GPa)", ylabel="$P_{cr}$ (MN)", title="K = 0.8")
axes[1].grid(True)


# %% [markdown]
# - $K$ changes the load by a factor of almost 3, following $1/K^2$.
# - $E$ changes it by a few percent, linearly.
#
# ## The uncertain inputs
#
# - $K \sim \mathcal U(0.6, 1.0)$: real supports are only partially rigid.
# - $E$ is lognormal with mean 200 GPa and standard deviation 10 GPa, so it is always positive.
#
# A lognormal $E = e^{Z}$ comes from a normal $Z \sim \mathcal N(m, s^2)$. Matching the mean and standard deviation of $E$ gives
#
# $$
# s = \sqrt{\ln\left(1 + \frac{\sigma_E^2}{\mu_E^2}\right)}, \qquad m = \ln \mu_E - \frac{s^2}{2}.
# $$
#
# `sample_inputs` returns $n$ rows $[E\;K]$: rows are observations, columns are features.
#
# - $E$ comes from [`Generator.lognormal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.lognormal.html), which takes $m$ and $s$.
# - $K$ comes from [`Generator.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html).


# %%
def sample_inputs(rng: np.random.Generator, n: int, inputs: InputDistribution = INPUTS) -> Array:
    """n-by-2 array [E, K]: lognormal E with the requested mean and std, uniform K."""
    s = np.sqrt(np.log1p((inputs.e_std / inputs.e_mean) ** 2))  # std of the underlying normal; log1p(x) = log(1 + x)
    m = np.log(inputs.e_mean) - s**2 / 2  # mean of the underlying normal

    e = rng.lognormal(m, s, n)
    k = rng.uniform(*inputs.k_limits, n)
    return np.column_stack((e, k))  # two length-n arrays side by side: shape (n, 2)


# A Generator created from a seed always produces the same numbers.
preview = sample_inputs(np.random.default_rng(SEED), 5000)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

axes[0].hist(preview[:, 0] / 1e9, bins=40)  # column 0 is E
axes[0].set(xlabel="E (GPa)", ylabel="count", title="Lognormal E")

axes[1].hist(preview[:, 1], bins=40)  # column 1 is K
axes[1].set(xlabel="K", title="Uniform K")

# %% [markdown]
# # One input: every model is a curve
#
# $E$ is fixed at its mean, so the network is a function of $K$ only and we can draw it on top of the true curve.
#
# ## Generate and split the data
#
# The rows are split **once** into three sets:
#
# | Set | Share | Used for |
# |---|---|---|
# | training | 60% | fitting the weights |
# | validation | 20% | comparing settings |
# | test | 20% | one final error estimate, at the very end |

# %% jupyter={"source_hidden": true}
# Utilities: the split.
@dataclass(frozen=True, slots=True)
class Dataset:
    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array
    x_test: Array
    y_test: Array


def split_dataset(rng: np.random.Generator, x: Array, y: Array) -> Dataset:
    """Shuffle the rows once and cut them into training, validation and test sets."""
    order = rng.permutation(len(x))
    n_train, n_val = round(SPLIT[0] * len(x)), round(SPLIT[1] * len(x))
    train, val, test = order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]
    return Dataset(x[train], y[train], x[val], y[val], x[test], y[test])


# %%
rng = np.random.default_rng(SEED)

# [:, [1]] keeps column 1 (K) as a 2-D column of shape (N, 1): Keras expects (rows, features).
# [:, 1] would give a flat array of shape (N,).
k = sample_inputs(rng, N_SAMPLES)[:, [1]]
pcr = buckling_load(INPUTS.e_mean, k[:, 0])  # targets, with E fixed at its mean

# split_dataset (collapsed above) returns data1.x_train, data1.y_train, data1.x_val, ...
data1 = split_dataset(rng, k, pcr)

# Try it: keep only 15 training rows (uncomment), then rerun the cells below.
# data1 = replace(data1, x_train=data1.x_train[:15], y_train=data1.y_train[:15])

plt.figure()
plt.scatter(data1.x_train, data1.y_train, s=8, label="training")  # label feeds the legend
plt.scatter(data1.x_val, data1.y_val, s=8, label="validation")
plt.scatter(data1.x_test, data1.y_test, s=8, label="test")
plt.xlabel("K")
plt.ylabel("$P_{cr}$ (MN)")
plt.legend()
plt.grid(True)

# %% [markdown]
# ## Define the network
#
# A network with one hidden layer computes
#
# $$
# \hat y = W_2 \tanh(W_1 x' + b_1) + b_2.
# $$
#
# In Keras it is a stack of layers:
#
# - [`Normalization`](https://keras.io/api/layers/preprocessing_layers/numerical/normalization/): standardises the inputs, $x' = (x-\mu)/\sigma$.
# - [`Dense(h, activation="tanh")`](https://keras.io/api/layers/core_layers/dense/): the affine map $Wx+b$ with $h$ outputs, then the nonlinearity.
# - A final `Dense(1)` with no activation, so the output can take any value.
#
# `make_network` builds a fresh network, so every run starts from the same initial weights.


# %%
def make_network(data: Dataset, widths: tuple[int, ...] = (HIDDEN,), activation: str = "tanh") -> keras.Sequential:
    # A Keras layer owns its weights, so every training run needs a freshly built network.
    keras.utils.set_random_seed(SEED)  # same initial weights every time

    normalise = keras.layers.Normalization()
    normalise.adapt(data.x_train)  # computes mu and sigma from the training rows only

    return keras.Sequential(
        [
            keras.Input((data.x_train.shape[1],)),  # number of input features
            normalise,  # Try it (two inputs): comment out this line
            # One Dense layer per entry of widths; *[...] unpacks them into this list.
            # widths=() gives no hidden layer at all: a linear model.
            *[keras.layers.Dense(width, activation=activation) for width in widths],
            keras.layers.Dense(1),  # linear output
        ]
    )


model1 = make_network(data1)
model1.summary()  # 25 trainable parameters: (1+1)*8 + (8+1)*1


# %% [markdown]
# ## Train
#
# Training minimises the mean squared error with **mini-batch gradient descent**: each update uses $b$ random rows, and one **epoch** is one pass over the training set.
#
# $$
# \mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat y_i)^2
# $$
#
# `train` is two Keras calls:
#
# | Call | Argument | Meaning |
# |---|---|---|
# | `compile` | `optimizer` | e.g. `keras.optimizers.Adam(learning_rate)` |
# | `compile` | `loss="mse"` | the loss above |
# | `fit` | `epochs`, `batch_size` | training budget and rows per update |
# | `fit` | `validation_data` | evaluated after every epoch, never used for updates |
# | `fit` | `TerminateOnNaN` | stops a run whose loss overflows |
#
# `fit` returns a `History`: `history.history["loss"]` and `["val_loss"]` hold one value per epoch.


# %%
def train(model: keras.Sequential, data: Dataset, optimizer: keras.optimizers.Optimizer, epochs: int = EPOCHS) -> keras.callbacks.History:
    model.compile(optimizer=optimizer, loss="mse")
    return model.fit(
        data.x_train,
        data.y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_data=(data.x_val, data.y_val),
        callbacks=[keras.callbacks.TerminateOnNaN()],
        verbose=0,  # no progress bar per epoch
    )


history1 = train(model1, data1, keras.optimizers.Adam(LEARN_RATE))

# history.history is a dict of lists; [-1] is the last epoch.
print(f"final validation MSE: {history1.history['val_loss'][-1]:.2e} MN^2")

# %% [markdown]
# ## Loss history
#
# `plot_loss` draws the training loss (solid, the mean over the epoch's mini-batches) and the validation loss (dashed) on a **log axis**.
#
# How to read it:
#
# - both still falling: train longer;
# - both flat: converged;
# - training falls, validation rises: **overfitting**;
# - spikes or `NaN`: the learning rate is too large.

# %% jupyter={"source_hidden": true}
# Utilities: prediction and plots.
def predict(model: keras.Sequential, x: Array) -> Array:
    """Network output as a flat array (model.predict returns shape (rows, 1))."""
    return model.predict(x, verbose=0)[:, 0]


def plot_loss(history: keras.callbacks.History, name: str) -> None:
    """Adds to the current axes: training (solid) and validation (dashed) MSE per epoch, log y-axis."""
    epochs = np.arange(1, len(history.history["loss"]) + 1)
    (line,) = plt.semilogy(epochs, history.history["loss"], "-", lw=2, label=f"{name} train")
    plt.semilogy(epochs, history.history["val_loss"], "--", lw=2, color=line.get_color(), label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (MN$^2$)")
    plt.legend()
    plt.grid(True)


def plot_fit_1d(model: keras.Sequential, data: Dataset) -> None:
    """Formula, network and training samples over K (E at its mean), in the current axes."""
    k_grid = np.linspace(INPUTS.k_limits[0] - 0.1, INPUTS.k_limits[1] + 0.1, 300)
    plt.plot(k_grid, buckling_load(INPUTS.e_mean, k_grid), "k-", lw=1.5, label="formula")
    plt.plot(k_grid, predict(model, k_grid[:, None]), lw=1.5, label="network")
    plt.scatter(data.x_train[:, 0], data.y_train, s=20, label="training samples")
    for limit in INPUTS.k_limits:
        plt.axvline(limit, ls=":", color="gray")
    plt.ylim(0, 3)
    plt.xlabel("K")
    plt.ylabel("$P_{cr}$ (MN)")
    plt.grid(True)


def plot_parity(y: Array, y_hat: Array, name: str) -> None:
    """New figure: predicted vs true (with the diagonal) and residuals vs true."""
    fig, (parity, residual) = plt.subplots(1, 2, figsize=(12, 4.5))
    parity.scatter(y, y_hat, s=8, alpha=0.5)
    parity.plot([y.min(), y.max()], [y.min(), y.max()], "k--")
    parity.set(xlabel="true $P_{cr}$ (MN)", ylabel="predicted (MN)", title=f"Parity: {name}", aspect="equal")
    residual.scatter(y, y_hat - y, s=8, alpha=0.5)
    residual.axhline(0, color="k", ls="--")
    residual.set(xlabel="true $P_{cr}$ (MN)", ylabel="residual (MN)", title=f"Residuals: {name}")
    for ax in (parity, residual):
        ax.grid(True)


# %%
plt.figure()
plot_loss(history1, "reference")
plt.title("Loss history")

# %% [markdown]
# ## Predict and compare with the formula
#
# `model.predict` evaluates the network. The curve extends beyond the sampled range (dotted lines), where the network **extrapolates**.

# %%
k_grid = np.linspace(0.5, 1.1, 300)  # wider than the sampled 0.6-1.0

# k_grid[:, None] adds a feature axis: shape (300,) becomes (300, 1).
# predict returns shape (300, 1); [:, 0] flattens it back for plotting.
p_net = model1.predict(k_grid[:, None], verbose=0)[:, 0]

plt.figure()
plt.plot(k_grid, buckling_load(INPUTS.e_mean, k_grid), "k-", lw=1.5, label="formula")
plt.plot(k_grid, p_net, lw=1.5, label="network")
plt.scatter(data1.x_train[:, 0], data1.y_train, s=10, alpha=0.3, label="training samples")
plt.axvline(INPUTS.k_limits[0], ls=":", color="gray")  # sampled range of K
plt.axvline(INPUTS.k_limits[1], ls=":", color="gray")
plt.xlabel("K")
plt.ylabel("$P_{cr}$ (MN)")
plt.legend()
plt.grid(True)

# %% [markdown]
# **Try it** (rerun from *Define the network*):
#
# - Set `LEARN_RATE = 1e-4` in Settings. What does the loss history look like after 200 epochs? And the curve?
# - Set `LEARN_RATE = 1`. What happens to the loss?
# - Keep only 15 training rows (*Generate and split the data*). Why does the fit get worse, and why does raising `EPOCHS` help only partly?

# %% [markdown]
# ## Underfitting and overfitting
#
# Real data are noisy. We keep only **10** training rows, add 20% noise, and train two extreme networks for 1000 epochs:
#
# - a **linear** model, with no hidden layer: too little capacity, so it is **biased** and cannot follow the curvature;
# - a **large** network, two hidden layers of 64: enough capacity to chase the noise.
#
# The validation targets stay noise-free, so validation loss measures the error against the true curve.

# %%
NOISE = 0.20
N_NOISY = 10

# replace() copies data1 with new training rows; data1 itself is not changed.
noise_rng = np.random.default_rng(SEED)
noisy = replace(
    data1,
    x_train=data1.x_train[:N_NOISY],
    y_train=data1.y_train[:N_NOISY] * (1 + NOISE * noise_rng.standard_normal(N_NOISY)),  # multiplicative noise
)

linear = make_network(noisy, widths=())  # no hidden layer: a straight line in K
large = make_network(noisy, widths=(64, 64))

# With 10 rows, one epoch is a single update, so 1000 epochs are only 1000 updates.
history_linear = train(linear, noisy, keras.optimizers.Adam(LEARN_RATE), epochs=1000)
history_large = train(large, noisy, keras.optimizers.Adam(LEARN_RATE), epochs=1000)

plt.figure(figsize=(16, 4))

plt.subplot(1, 3, 1)  # 1 row, 3 columns, panel 1
plot_fit_1d(linear, noisy)
plt.title("Linear: underfits")

plt.subplot(1, 3, 2)
plot_fit_1d(large, noisy)
plt.title("64 + 64: overfits")

plt.subplot(1, 3, 3)
plot_loss(history_linear, "linear")
plot_loss(history_large, "64+64")
plt.title("Loss histories")

# %% [markdown]
# **Try it:**
#
# - Which network has a large gap between training and validation loss? Which has none, but a high level?
# - Find the epoch where the large network's validation loss is lowest (`np.argmin(history_large.history["val_loss"])`). What would early stopping return?
# - Train `make_network(noisy)` (8 neurons) on `noisy` too. Where does it land between the two?

# %% [markdown]
# # Two inputs: every model is a surface
#
# Now both $E$ and $K$ vary, and the surrogate is a surface $\hat P(E, K)$.
#
# ## Generate the data

# %%
rng = np.random.default_rng(SEED)
x2 = sample_inputs(rng, N_SAMPLES)  # shape (N, 2): column 0 is E, column 1 is K
y2 = buckling_load(x2[:, 0], x2[:, 1])

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))

axes[0].scatter(x2[:, 0] / 1e9, x2[:, 1], s=4)
axes[0].set(xlabel="E (GPa)", ylabel="K", title="Inputs")

axes[1].scatter(x2[:, 1], y2, s=4)
axes[1].set(xlabel="K", ylabel="$P_{cr}$ (MN)", title="Load against K")

axes[2].hist(y2, bins=30)
axes[2].set(xlabel="$P_{cr}$ (MN)", title="Load distribution")

# %% [markdown]
# ## The target in 3D
#
# `np.meshgrid` builds a grid, `plot_surface` draws the formula, `scatter` adds the samples.

# %%
# e_grid and k_grid2 are 100-by-100 arrays: every combination of e_line and k_line.
e_grid, k_grid2 = np.meshgrid(e_line, k_line)

fig = plt.figure(figsize=(7, 5))
ax3 = fig.add_subplot(projection="3d")
ax3.plot_surface(e_grid / 1e9, k_grid2, buckling_load(e_grid, k_grid2), cmap="viridis", alpha=0.8)
ax3.scatter(x2[:, 0] / 1e9, x2[:, 1], y2, s=3, c="k")
ax3.set(xlabel="E (GPa)", ylabel="K", zlabel="$P_{cr}$ (MN)")
ax3.view_init(30, -35)  # elevation, azimuth in degrees

# %% [markdown]
# ## Split and train the reference network
#
# The same network as before, now with **two** input features.

# %%
data2 = split_dataset(rng, x2, y2)

model2 = make_network(data2)  # Keras reads the number of inputs (2) from data2.x_train
history2 = train(model2, data2, keras.optimizers.Adam(LEARN_RATE))

plt.figure()
plot_loss(history2, "reference")
plt.title("Reference network")

# %% [markdown]
# **Try it:** comment out the `normalise` line in `make_network`, then rerun that cell and this one. What happens to the loss, and why?
#
# - Hint: $E \approx 2\times10^{11}$ while $K \approx 0.8$.
# - With raw inputs, $E$ dominates $W_1 x$, `tanh` saturates at $\pm 1$, and its gradient vanishes.
# - `Normalization` rescales each input with the **training** mean and standard deviation, stored in the layer.

# %% [markdown]
# ## Evaluate on the validation set
#
# Three numbers, from `regression_metrics`:
#
# $$
# \mathrm{RMSE} = \sqrt{\mathrm{MSE}}, \qquad R^2 = 1 - \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}.
# $$
#
# - MSE in MN$^2$, RMSE in MN.
# - $R^2 = 1$ is a perfect fit.
#
# Two plots, from `plot_parity`:
#
# - **Parity:** predicted against true; a perfect model lies on the diagonal.
# - **Residuals:** $\hat y - y$ against true; look for trends, not just size.


# %%
def regression_metrics(y: Array, y_hat: Array) -> dict[str, float]:
    mse = float(np.mean((y - y_hat) ** 2))
    r2 = 1 - float(np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"MSE": mse, "RMSE": mse**0.5, "R2": r2}


y_hat2 = predict(model2, data2.x_val)

print(regression_metrics(data2.y_val, y_hat2))
plot_parity(data2.y_val, y_hat2, "reference, validation")

# %% [markdown]
# ## Refine: learning rate
#
# Adam at a small, the reference and a large learning rate. Only the learning rate changes.

# %%
# make_network resets the seed, so all three runs start from the same weights.
model_slow = make_network(data2)
history_slow = train(model_slow, data2, keras.optimizers.Adam(1e-4))

model_fast = make_network(data2)
history_fast = train(model_fast, data2, keras.optimizers.Adam(1.0))

# history2 (the reference, learning rate 1e-2) is reused as the medium rate.
plt.figure()
plot_loss(history_slow, r"$\eta = 10^{-4}$")  # r"..." keeps the backslash for the math
plot_loss(history2, r"$\eta = 10^{-2}$")
plot_loss(history_fast, r"$\eta = 1$")
plt.title("Adam: learning rate")

# %% [markdown]
# **Try it:**
#
# - Which rate is too small, and how can you tell from the curve?
# - What does the large rate do to the validation loss?

# %% [markdown]
# ## Refine: optimiser
#
# Plain **SGD** takes the same step size for every weight:
#
# $$
# \theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}.
# $$
#
# **Adam** adapts the step per weight, using running averages of the gradient and of its square.
# We train SGD at the same three rates.

# %%
model_sgd_slow = make_network(data2)
history_sgd_slow = train(model_sgd_slow, data2, keras.optimizers.SGD(1e-4))  # momentum defaults to 0

model_sgd = make_network(data2)
history_sgd = train(model_sgd, data2, keras.optimizers.SGD(1e-2))

model_sgd_fast = make_network(data2)
history_sgd_fast = train(model_sgd_fast, data2, keras.optimizers.SGD(1.0))

plt.figure()
plot_loss(history_sgd_slow, r"SGD $\eta = 10^{-4}$")
plot_loss(history_sgd, r"SGD $\eta = 10^{-2}$")
plot_loss(history_sgd_fast, r"SGD $\eta = 1$")
plt.title("SGD: learning rate")

# %% [markdown]
# **Try it:**
#
# - Which optimiser is more sensitive to the learning rate? A curve that stops early hit `NaN`.
# - Use `keras.optimizers.SGD(1e-2, momentum=0.9)`. Does the medium rate improve?

# %% [markdown]
# ## Refine: a wider and deeper network
#
# Two hidden layers of 16 neurons, Adam at the reference rate.

# %%
model_wide = make_network(data2, widths=(16, 16))  # Try it: activation="relu"
history_wide = train(model_wide, data2, keras.optimizers.Adam(LEARN_RATE))

plt.figure()
plot_loss(history2, "8")
plot_loss(history_wide, "16-16")
plt.title("Network size")

y_hat_wide = predict(model_wide, data2.x_val)
print(regression_metrics(data2.y_val, y_hat_wide))
plot_parity(data2.y_val, y_hat_wide, "16-16, validation")

# %% [markdown]
# **Try it:** did the wider network help? Compare its validation MSE with the reference, and `model_wide.summary()` with `model2.summary()`.
# A bigger network has more parameters to fit, and is not automatically better.
#
# ## Choose a model
#
# Pick the network with the lowest **validation** MSE above. The test set stays untouched until the next cell.

# %%
chosen = model2  # Try it: replace with your choice

# %% [markdown]
# ## Final test, once
#
# - The test set played no part in any choice, so its error is an honest estimate for new columns.
# - If you change the model after seeing it, the test set becomes a second validation set.

# %%
y_test_hat = predict(chosen, data2.x_test)

print(regression_metrics(data2.y_test, y_test_hat))
plot_parity(data2.y_test, y_test_hat, "chosen, test")

# %% [markdown]
# # Using the surrogate
#
# ## Monte Carlo
#
# Propagate the input uncertainty to the load with `MC_SAMPLES` fresh inputs:
#
# - the same bin edges for formula and network, so the histograms compare;
# - `density=True` so both have area one;
# - the 95th percentile describes the upper tail that matters for design.

# %%
x_mc = sample_inputs(np.random.default_rng(SEED + 1000), MC_SAMPLES)  # a different seed: these inputs are new
y_true = buckling_load(x_mc[:, 0], x_mc[:, 1])
y_net = predict(chosen, x_mc)

edges = np.linspace(y_true.min(), y_true.max(), 50)  # 50 edges = 49 shared bins

plt.figure()
plt.hist(y_true, edges, density=True, alpha=0.5, label="formula")
plt.hist(y_net, edges, density=True, histtype="step", lw=1.5, label="network")
plt.xlabel("$P_{cr}$ (MN)")
plt.ylabel("probability density")
plt.legend()

print(f"p95: formula {np.percentile(y_true, 95):.4f} MN, network {np.percentile(y_net, 95):.4f} MN")

# %% [markdown]
# ## Response curves
#
# Vary one input, hold the other at its mean. The network should reproduce a straight line in $E$ and a $1/K^2$ curve in $K$.

# %%
k_mean = float(np.mean(INPUTS.k_limits))

# The network needs both inputs: pair each swept value with the fixed one.
# np.full_like(v, c) is an array shaped like v, filled with c.
sweep_e = np.column_stack((e_line, np.full_like(e_line, k_mean)))
sweep_k = np.column_stack((np.full_like(k_line, INPUTS.e_mean), k_line))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(e_line / 1e9, buckling_load(e_line, k_mean), "k-", lw=1.5, label="formula")
axes[0].plot(e_line / 1e9, predict(chosen, sweep_e), "--", lw=1.5, label="network")
axes[0].set(xlabel="E (GPa)", ylabel="$P_{cr}$ (MN)", title="K = 0.8")
axes[0].legend()

axes[1].plot(k_line, buckling_load(INPUTS.e_mean, k_line), "k-", lw=1.5, label="formula")
axes[1].plot(k_line, predict(chosen, sweep_k), "--", lw=1.5, label="network")
axes[1].set(xlabel="K", ylabel="$P_{cr}$ (MN)", title="E = 200 GPa")
axes[1].legend()

# %% [markdown]
# # Takeaways
#
# - Split once; choose by validation error; test once.
# - Standardise the inputs with training statistics.
# - Read loss histories on a log axis before reading final numbers.
# - Too little capacity underfits (high, flat losses); too much overfits (a growing gap).
#   Refine one setting at a time and look at parity and residual plots, not only at one number.

# %%
plt.show()
