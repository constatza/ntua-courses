#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2", "numpy==2.5.4", "scikit-learn==1.9.1"]
# ///

"""Reproducible instructor baseline for the cantilever surrogate exercise."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPRegressor

Array = NDArray[np.float64]
Solver = Literal["adam", "sgd"]
Activation = Literal["tanh", "relu"]
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "generated"
SEED = 20261104
DATA_SIZE = 3000
MC_SIZE = 10000
EPOCHS = 1500
# The sheet asks for the baseline MSE at this epoch, read from the same 1500-epoch history.
CHECKPOINT_EPOCH = 500
BATCH_SIZE = 64
BASELINE_ARCHITECTURE = (4,)
BASELINE_LEARNING_RATE = 1e-4
P_MEAN, P_STD = 10000.0, 700.0
E_MEAN, E_STD = 30e9, 3e9
L_LIMITS = (2.4, 3.6)
I_LIMITS = (4e-5, 1.2e-4)
FEATURE_NAMES = ("P (N)", "E (Pa)", "L (m)", "I (m⁴)")


@dataclass(frozen=True, slots=True)
class TargetScaler:
    """z = (delta - mean) / std; fitted on training targets only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, y: Array) -> TargetScaler:
        return cls(float(y.mean()), float(y.std()))

    def forward(self, y: Array) -> Array:
        return (y - self.mean) / self.std

    def inverse(self, z: Array) -> Array:
        return self.mean + self.std * z


# Keeping the target in metres is the identity transform, so no run needs an Optional scaler.
KEEP_METRES = TargetScaler(0.0, 1.0)


@dataclass(frozen=True, slots=True)
class Trial:
    """One Part E configuration; everything not listed stays as in the baseline."""

    hidden_layers: tuple[int, ...]
    learning_rate: float
    activation: Activation
    scale_target: bool


# At most 6 configurations, as the sheet allows, each changing one setting of the previous
# row, so the table itself shows which change helped (rate, target scaling, width, rate,
# depth, activation).
SEARCH = (
    Trial((4,), 1e-3, "tanh", False),
    Trial((4,), 1e-3, "tanh", True),
    Trial((8,), 1e-3, "tanh", True),
    Trial((8,), 3e-3, "tanh", True),
    Trial((8, 8), 3e-3, "tanh", True),
    Trial((8,), 3e-3, "relu", True),
)


@dataclass(frozen=True, slots=True)
class RunResult:
    name: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    activation: Activation
    solver: Solver
    model: MLPRegressor
    target_scaler: TargetScaler
    train_mse: tuple[float, ...]
    val_mse: tuple[float, ...]

    def predict(self, features: Array) -> Array:
        """Deflection in metres, whatever the training target scale was."""
        return self.target_scaler.inverse(self.model.predict(features))

    @property
    def scales_target(self) -> bool:
        return self.target_scaler != KEEP_METRES

    @property
    def parameter_count(self) -> int:
        return sum(w.size for w in self.model.coefs_) + sum(b.size for b in self.model.intercepts_)


@dataclass(frozen=True, slots=True)
class Moments:
    mean: float
    std: float
    p95: float


@dataclass(frozen=True, slots=True)
class Metrics:
    test_mse: float
    test_rmse: float
    test_r2: float
    tail_rmse: float
    mc_analytical: Moments
    mc_surrogate: Moments
    response_max_absolute_error: dict[str, float]


@dataclass(frozen=True, slots=True)
class SplitData:
    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array


def sample_inputs(rng: np.random.Generator, size: int) -> Array:
    p = rng.normal(P_MEAN, P_STD, size)
    log_variance = np.log1p((E_STD / E_MEAN) ** 2)
    e = rng.lognormal(np.log(E_MEAN) - log_variance / 2, np.sqrt(log_variance), size)
    length = rng.uniform(*L_LIMITS, size)
    inertia = rng.uniform(*I_LIMITS, size)
    return np.column_stack((p, e, length, inertia))


def deflection(x: Array) -> Array:
    return x[:, 0] * x[:, 2] ** 3 / (3 * x[:, 1] * x[:, 3])


def mse(actual: Array, predicted: Array) -> float:
    return float(np.mean((actual - predicted) ** 2))


def split_indices(rng: np.random.Generator) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    order = rng.permutation(DATA_SIZE)
    return order[:1800], order[1800:2400], order[2400:]


def scale_inputs(x: Array, mean: Array, std: Array) -> Array:
    return (x - mean) / std


def run_experiment(
    name: str,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    data: SplitData,
    solver: Solver = "adam",
    activation: Activation = "tanh",
    target_scaler: TargetScaler = KEEP_METRES,
) -> RunResult:
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        momentum=0.0,
        alpha=0.0,
        batch_size=BATCH_SIZE,
        learning_rate_init=learning_rate,
        random_state=SEED,
        early_stopping=False,
        shuffle=True,
    )
    scaled_y_train = target_scaler.forward(data.y_train)
    train_history: list[float] = []
    val_history: list[float] = []
    # fit() would stop once the loss improves by less than tol, so the epoch budget is driven
    # by hand: one partial_fit call is one pass over the training set.
    for _ in range(EPOCHS):
        model.partial_fit(data.x_train, scaled_y_train)
        # MSE is always measured on deflection in metres so runs stay comparable.
        train_loss = mse(data.y_train, target_scaler.inverse(model.predict(data.x_train)))
        val_loss = mse(data.y_val, target_scaler.inverse(model.predict(data.x_val)))
        if not np.isfinite(train_loss + val_loss):
            raise ValueError(f"Run {name} produced non-finite MSE")
        train_history.append(train_loss)
        val_history.append(val_loss)
    return RunResult(name, hidden_layers, learning_rate, activation, solver, model, target_scaler, tuple(train_history), tuple(val_history))


def fit_trial(job: tuple[str, Trial, SplitData]) -> RunResult:
    """Top-level so ProcessPoolExecutor can pickle it; every trial is seeded and deterministic."""
    name, trial, data = job
    scaler = TargetScaler.fit(data.y_train) if trial.scale_target else KEEP_METRES
    return run_experiment(name, trial.hidden_layers, trial.learning_rate, data, activation=trial.activation, target_scaler=scaler)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_inputs(x: Array, y: Array) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 2.8))
    for index, label in enumerate(FEATURE_NAMES):
        axes[index].hist(x[:, index], bins=30, color="#4477aa")
        axes[index].set_xlabel(label)
        axes[index].set_ylabel("count")
    axes[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[4].hist(y, bins=30, color="#228833")
    axes[4].set_xlabel("deflection (m)")
    axes[4].set_ylabel("count")
    fig.tight_layout()
    save_figure(fig, "input_histograms.png")


def draw_history(ax: plt.Axes, run: RunResult, title: str) -> None:
    epochs = np.arange(1, len(run.train_mse) + 1)
    ax.plot(epochs, run.train_mse, label="training MSE")
    ax.plot(epochs, run.val_mse, label="validation MSE")
    ax.set(xlabel="epoch", ylabel="deflection MSE (m²)", yscale="log", title=title)


def plot_baseline(run: RunResult, no_skill: float) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    draw_history(ax, run, "Baseline: 4 tanh neurons, Adam, lr=1e-4")
    ax.axhline(no_skill, color="black", linestyle=":", label="no-skill MSE₀")
    ax.axvline(CHECKPOINT_EPOCH, color="grey", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "loss_baseline.png")


def plot_probes(baseline: RunResult, probes: list[tuple[RunResult, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    epochs = np.arange(1, len(baseline.val_mse) + 1)
    for ax, (run, title) in zip(axes.flat, probes, strict=True):
        draw_history(ax, run, title)
        ax.plot(epochs, baseline.val_mse, color="grey", linestyle="--", label="baseline validation")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, "loss_probes.png")


def plot_selected(run: RunResult) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    draw_history(ax, run, f"Selected {run.name}: {describe(run)}")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "loss_selected.png")


def plot_test(actual: Array, predicted: Array) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    limits = (float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max())))
    axes[0].scatter(actual, predicted, s=9, alpha=0.35)
    axes[0].plot(limits, limits, color="black", label="perfect prediction")
    axes[0].set(xlabel="analytical deflection (m)", ylabel="predicted deflection (m)", title="Test parity")
    axes[0].legend()
    axes[1].scatter(actual, predicted - actual, s=9, alpha=0.35)
    axes[1].axhline(0, color="black")
    axes[1].set(xlabel="analytical deflection (m)", ylabel="prediction error (m)", title="Test signed errors")
    fig.tight_layout()
    save_figure(fig, "test_diagnostics.png")


def plot_mc(actual: Array, predicted: Array) -> None:
    pooled = np.concatenate((actual, predicted))
    grid = np.linspace(float(pooled.min()), float(pooled.max()), 400)
    bandwidth = float(pooled.std()) * len(actual) ** (-1 / 5)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for values, label, color, style in (
        (actual, "analytical", "#1f77b4", "-"),
        (predicted, "network", "#d62728", "--"),
    ):
        kde = KernelDensity(bandwidth=bandwidth).fit(values[:, None])
        ax.plot(grid, np.exp(kde.score_samples(grid[:, None])), color=color, linestyle=style, linewidth=2, label=label)
    ax.set(xlabel="deflection (m)", ylabel="density (1/m)", title="Fresh Monte Carlo response densities")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "mc_distribution.png")


def plot_responses(predict: Callable[[Array], Array], x_mean: Array) -> dict[str, float]:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    ranges = (
        (P_MEAN - 2 * P_STD, P_MEAN + 2 * P_STD),
        (E_MEAN - 2 * E_STD, E_MEAN + 2 * E_STD),
        L_LIMITS,
        I_LIMITS,
    )
    errors: dict[str, float] = {}
    for index, (ax, bounds, label) in enumerate(zip(axes.flat, ranges, FEATURE_NAMES, strict=True)):
        grid = np.linspace(*bounds, 101)
        points = np.tile(x_mean, (len(grid), 1))
        points[:, index] = grid
        analytical = deflection(points)
        network = predict(points)
        ax.plot(grid, analytical, label="analytical")
        ax.plot(grid, network, label="network")
        ax.set(xlabel=label, ylabel="deflection (m)")
        if index == 3:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend()
        errors[label] = float(np.max(np.abs(network - analytical)))
    fig.tight_layout()
    save_figure(fig, "response_curves.png")
    return errors


def tex_number(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def describe(run: RunResult) -> str:
    widths = "+".join(map(str, run.hidden_layers))
    scaling = "z" if run.scales_target else "m"
    return f"{widths} {run.activation}, lr={run.learning_rate:g}, target {scaling}"


def search_rows(runs: list[RunResult], baseline: RunResult) -> list[str]:
    return [
        rf"{run.name} & {', '.join(map(str, run.hidden_layers))} & {run.activation} & "
        rf"{tex_number(run.learning_rate, 0)} & {'standardized' if run.scales_target else 'metres'} & "
        rf"{run.parameter_count} & {tex_number(run.train_mse[-1], 2)} & {tex_number(run.val_mse[-1], 2)} & "
        rf"{baseline.val_mse[-1] / run.val_mse[-1]:.0f} \\"
        for run in runs
    ]


def write_results_tex(
    baseline: RunResult,
    no_skill: float,
    probes: dict[str, RunResult],
    search: list[RunResult],
    selected: RunResult,
    metrics: Metrics,
    x: Array,
    y: Array,
) -> None:
    sample_rows = [
        rf"{symbol} & {tex_number(float(x[:, index].mean()))} & {tex_number(float(x[:, index].std()))} \\"
        for index, symbol in enumerate((r"$P$ (N)", r"$E$ (Pa)", r"$L$ (m)", r"$I$ (m$^4$)"))
    ]
    sample_rows.append(rf"$\delta$ (m) & {tex_number(float(y.mean()))} & {tex_number(float(y.std()))} \\")
    checkpoint = CHECKPOINT_EPOCH - 1
    layer_word = "hidden layer" if len(selected.hidden_layers) == 1 else "hidden layers"
    architecture = f"{len(selected.hidden_layers)} {layer_word} of {' + '.join(map(str, selected.hidden_layers))}"
    # Keep generated values in one place so answers cannot drift from the baseline.
    content = "\n".join(
        [
            rf"\newcommand{{\NoSkillMSE}}{{{tex_number(no_skill)}}}",
            rf"\newcommand{{\CheckpointTrainMSE}}{{{tex_number(baseline.train_mse[checkpoint])}}}",
            rf"\newcommand{{\CheckpointValMSE}}{{{tex_number(baseline.val_mse[checkpoint])}}}",
            rf"\newcommand{{\BaselineTrainMSE}}{{{tex_number(baseline.train_mse[-1])}}}",
            rf"\newcommand{{\BaselineValMSE}}{{{tex_number(baseline.val_mse[-1])}}}",
            rf"\newcommand{{\HigherRateValMSE}}{{{tex_number(probes['higher_rate'].val_mse[-1])}}}",
            rf"\newcommand{{\TinyRateValMSE}}{{{tex_number(probes['tiny_rate'].val_mse[-1])}}}",
            rf"\newcommand{{\RawInputsValMSE}}{{{tex_number(probes['raw_inputs'].val_mse[-1])}}}",
            rf"\newcommand{{\SgdTrainMSE}}{{{tex_number(probes['sgd'].train_mse[-1])}}}",
            rf"\newcommand{{\SgdValMSE}}{{{tex_number(probes['sgd'].val_mse[-1])}}}",
            rf"\newcommand{{\SelectedName}}{{{selected.name}}}",
            rf"\newcommand{{\SelectedValMSE}}{{{tex_number(selected.val_mse[-1])}}}",
            rf"\newcommand{{\SelectedTrainMSE}}{{{tex_number(selected.train_mse[-1])}}}",
            rf"\newcommand{{\SelectedArchitecture}}{{{architecture}}}",
            rf"\newcommand{{\SelectedActivation}}{{{selected.activation}}}",
            rf"\newcommand{{\SelectedRate}}{{{tex_number(selected.learning_rate, 0)}}}",
            rf"\newcommand{{\SelectedParams}}{{{selected.parameter_count}}}",
            rf"\newcommand{{\BaselineParams}}{{{baseline.parameter_count}}}",
            rf"\newcommand{{\SearchTrials}}{{{len(search)}}}",
            rf"\newcommand{{\ImprovementRatio}}{{{baseline.val_mse[-1] / selected.val_mse[-1]:.0f}}}",
            rf"\newcommand{{\MaxResponseError}}{{{tex_number(max(metrics.response_max_absolute_error.values()))}}}",
            rf"\newcommand{{\TestMSE}}{{{tex_number(metrics.test_mse)}}}",
            rf"\newcommand{{\TestRMSE}}{{{metrics.test_rmse:.7f}}}",
            rf"\newcommand{{\TestRtwo}}{{{metrics.test_r2:.4f}}}",
            rf"\newcommand{{\TailRMSE}}{{{metrics.tail_rmse:.7f}}}",
            rf"\newcommand{{\AnalyticalMean}}{{{metrics.mc_analytical.mean:.7f}}}",
            rf"\newcommand{{\SurrogateMean}}{{{metrics.mc_surrogate.mean:.7f}}}",
            rf"\newcommand{{\AnalyticalStd}}{{{metrics.mc_analytical.std:.7f}}}",
            rf"\newcommand{{\SurrogateStd}}{{{metrics.mc_surrogate.std:.7f}}}",
            rf"\newcommand{{\AnalyticalPnn}}{{{metrics.mc_analytical.p95:.7f}}}",
            rf"\newcommand{{\SurrogatePnn}}{{{metrics.mc_surrogate.p95:.7f}}}",
            r"\newcommand{\SampleRows}{%",
            *sample_rows,
            "}",
            r"\newcommand{\SearchRows}{%",
            *search_rows(search, baseline),
            "}",
            "",
        ]
    )
    (OUTPUT / "results.tex").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    rng = np.random.default_rng(SEED)
    x = sample_inputs(rng, DATA_SIZE)
    y = deflection(x)
    train_idx, val_idx, test_idx = split_indices(rng)
    assert len(set(train_idx) | set(val_idx) | set(test_idx)) == DATA_SIZE
    assert np.all(x[:, 1:] > 0) and np.all(np.isfinite(y))
    plot_inputs(x, y)

    x_mean = x[train_idx].mean(axis=0)
    x_std = x[train_idx].std(axis=0)
    x_model = scale_inputs(x, x_mean, x_std)
    standardized = SplitData(x_model[train_idx], y[train_idx], x_model[val_idx], y[val_idx])
    raw = SplitData(x[train_idx], y[train_idx], x[val_idx], y[val_idx])
    # The yardstick a trained model must beat: always predict the mean training deflection.
    no_skill = mse(y[val_idx], np.full(len(val_idx), y[train_idx].mean()))

    def train(name: str, rate: float, data: SplitData, solver: Solver = "adam") -> RunResult:
        run = run_experiment(name, BASELINE_ARCHITECTURE, rate, data, solver)
        print(f"{name}: solver={solver} lr={rate:g} train={run.train_mse[-1]:.5g} val={run.val_mse[-1]:.5g}")
        return run

    baseline = train("baseline", BASELINE_LEARNING_RATE, standardized)
    plot_baseline(baseline, no_skill)
    probes = {
        "tiny_rate": train("lr 1e-5", 1e-5, standardized),
        "higher_rate": train("lr 1e-1", 1e-1, standardized),
        "raw_inputs": train("raw inputs", BASELINE_LEARNING_RATE, raw),
        "sgd": train("plain SGD", BASELINE_LEARNING_RATE, standardized, "sgd"),
    }
    plot_probes(baseline, [
        (probes["tiny_rate"], "Learning rate 1e-5"),
        (probes["higher_rate"], "Learning rate 1e-1"),
        (probes["raw_inputs"], "Raw (unstandardized) inputs"),
        (probes["sgd"], "Plain SGD, lr=1e-4"),
    ])

    jobs = [(f"S{number}", trial, standardized) for number, trial in enumerate(SEARCH, start=1)]
    with ProcessPoolExecutor() as pool:
        search = list(pool.map(fit_trial, jobs))
    for run in search:
        print(f"{run.name}: {describe(run)} val={run.val_mse[-1]:.5g}")
    selected = min((baseline, *search), key=lambda run: run.val_mse[-1])
    plot_selected(selected)

    def predict_deflection(points: Array) -> Array:
        return selected.predict(scale_inputs(points, x_mean, x_std))

    test_actual = y[test_idx]
    test_predicted = predict_deflection(x[test_idx])
    test_mse = mse(test_actual, test_predicted)
    test_rmse = float(np.sqrt(test_mse))
    test_r2 = float(1 - np.sum((test_actual - test_predicted) ** 2) / np.sum((test_actual - test_actual.mean()) ** 2))
    tail = test_actual >= np.quantile(test_actual, 0.95)
    tail_rmse = float(np.sqrt(mse(test_actual[tail], test_predicted[tail])))
    plot_test(test_actual, test_predicted)

    mc_inputs = sample_inputs(rng, MC_SIZE)
    mc_actual = deflection(mc_inputs)
    mc_predicted = predict_deflection(mc_inputs)
    plot_mc(mc_actual, mc_predicted)
    response_errors = plot_responses(predict_deflection, np.array([P_MEAN, E_MEAN, 3.0, 8e-5]))

    def moments(values: Array) -> Moments:
        return Moments(float(values.mean()), float(values.std()), float(np.quantile(values, 0.95)))

    metrics = Metrics(test_mse, test_rmse, test_r2, tail_rmse, moments(mc_actual), moments(mc_predicted), response_errors)
    runs = [baseline, *probes.values(), *search]
    summary = {
        "seed": SEED,
        "dataset_size": DATA_SIZE,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "no_skill_validation_mse": no_skill,
        "sample_moments": {name: asdict(moments(x[:, index])) for index, name in enumerate(FEATURE_NAMES)} | {"deflection (m)": asdict(moments(y))},
        "baseline_checkpoint": {"epoch": CHECKPOINT_EPOCH, "training_mse": baseline.train_mse[CHECKPOINT_EPOCH - 1], "validation_mse": baseline.val_mse[CHECKPOINT_EPOCH - 1]},
        "runs": [
            {"name": run.name, "hidden_layers": list(run.hidden_layers), "learning_rate": run.learning_rate, "activation": run.activation, "solver": run.solver,
             "target_scaling": run.scales_target, "parameters": run.parameter_count,
             "completed_epochs": len(run.train_mse), "training_mse": run.train_mse[-1], "validation_mse": run.val_mse[-1]}
            for run in runs
        ],
        "selected_run": selected.name,
        "metrics": asdict(metrics),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_tex(baseline, no_skill, probes, search, selected, metrics, x, y)
    print(f"selected {selected.name}: test RMSE={test_rmse:.5f} m, R²={test_r2:.4f}")


if __name__ == "__main__":
    main()
