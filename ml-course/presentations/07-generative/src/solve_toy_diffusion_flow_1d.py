#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "scipy",
# ]
# ///

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR.parent.parent / "exercises" / "toy-diffusion-flow" / "assets" / "generated"

MIXTURE_WEIGHTS = np.array([0.58, 0.42], dtype=float)
MIXTURE_MEANS = np.array([-1.6, 1.8], dtype=float)
MIXTURE_STDS = np.array([0.35, 0.55], dtype=float)


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 20260404
    train_size: int = 2048
    val_size: int = 1024
    test_size: int = 2048
    train_pairs: int = 8192
    val_pairs: int = 4096
    sample_count: int = 3000
    epochs: int = 500
    hidden_layers: tuple[int, int, int, int, int, int, int] = (16, 16, 16, 16, 16, 16, 16)
    diffusion_steps: int = 40
    diffusion_beta_start: float = 1.0e-4
    diffusion_beta_end: float = 2.0e-2
    flow_steps: int = 60
    score_timesteps: tuple[int, int, int] = (5, 20, 40)


@dataclass(frozen=True)
class DiffusionSchedule:
    beta: np.ndarray
    alpha: np.ndarray
    alpha_bar: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.beta.shape[0] - 1)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(name: str) -> Path:
    return ensure_dir(OUTPUT_DIR) / name


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    variance = std**2
    return np.exp(-0.5 * ((x - mean) ** 2) / variance) / np.sqrt(2.0 * np.pi * variance)


def mixture_pdf(x: np.ndarray) -> np.ndarray:
    density = np.zeros_like(x, dtype=float)
    for weight, mean, std in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_STDS, strict=True):
        density += weight * normal_pdf(x, float(mean), float(std))
    return density


def sample_mixture(rng: np.random.Generator, size: int) -> np.ndarray:
    components = rng.choice(len(MIXTURE_WEIGHTS), size=size, p=MIXTURE_WEIGHTS)
    means = MIXTURE_MEANS[components]
    stds = MIXTURE_STDS[components]
    return rng.normal(loc=means, scale=stds)


def build_splits(config: BaselineConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "train": sample_mixture(rng, config.train_size),
        "val": sample_mixture(rng, config.val_size),
        "test": sample_mixture(rng, config.test_size),
    }


def build_linear_schedule(config: BaselineConfig) -> DiffusionSchedule:
    beta = np.concatenate(
        (
            np.array([0.0], dtype=float),
            np.linspace(config.diffusion_beta_start, config.diffusion_beta_end, config.diffusion_steps, dtype=float),
        )
    )
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    return DiffusionSchedule(beta=beta, alpha=alpha, alpha_bar=alpha_bar)


def build_diffusion_pairs(
    samples: np.ndarray,
    schedule: DiffusionSchedule,
    pair_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.integers(0, samples.size, size=pair_count)
    x0 = samples[indices]
    timestep = rng.integers(1, schedule.num_steps + 1, size=pair_count)
    epsilon = rng.normal(size=pair_count)
    signal = np.sqrt(schedule.alpha_bar[timestep])
    sigma = np.sqrt(1.0 - schedule.alpha_bar[timestep])
    xt = signal * x0 + sigma * epsilon
    features = np.column_stack([xt, timestep / schedule.num_steps])
    return features, epsilon


def build_flow_pairs(
    target_samples: np.ndarray,
    pair_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.integers(0, target_samples.size, size=pair_count)
    x1 = target_samples[indices]
    x0 = rng.normal(size=pair_count)
    time = rng.uniform(0.0, 1.0, size=pair_count)
    xt = (1.0 - time) * x0 + time * x1
    velocity = x1 - x0
    features = np.column_stack([xt, time])
    return features, velocity


def make_regressor(config: BaselineConfig, seed: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=config.hidden_layers,
        activation="tanh",
        solver="adam",
        learning_rate_init=1.0e-3,
        alpha=1.0e-4,
        batch_size=128,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=seed,
    )


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    difference = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(difference * difference))


def train_regressor(
    model: MLPRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
) -> tuple[MLPRegressor, list[float], list[float]]:
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        train_losses.append(mean_squared_error(y_train, model.predict(X_train)))
        val_losses.append(mean_squared_error(y_val, model.predict(X_val)))

    return model, train_losses, val_losses


def diffusion_posterior_mean(x_t: np.ndarray, x0_hat: np.ndarray, schedule: DiffusionSchedule, timestep: int) -> np.ndarray:
    alpha_t = schedule.alpha[timestep]
    beta_t = schedule.beta[timestep]
    alpha_bar_t = schedule.alpha_bar[timestep]
    alpha_bar_prev = schedule.alpha_bar[timestep - 1]
    coeff_x0 = np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coeff_xt = np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    return coeff_x0 * x0_hat + coeff_xt * x_t


def sample_diffusion(
    model: MLPRegressor,
    schedule: DiffusionSchedule,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x_t = rng.normal(size=sample_count)
    for timestep in range(schedule.num_steps, 0, -1):
        features = np.column_stack([x_t, np.full(sample_count, timestep / schedule.num_steps)])
        epsilon_hat = model.predict(features)
        sigma_t = np.sqrt(1.0 - schedule.alpha_bar[timestep])
        x0_hat = (x_t - sigma_t * epsilon_hat) / np.sqrt(schedule.alpha_bar[timestep])
        x0_hat = np.clip(x0_hat, -5.0, 5.0)
        x_t = diffusion_posterior_mean(x_t, x0_hat, schedule, timestep)
    return x_t


def sample_flow(
    model: MLPRegressor,
    sample_count: int,
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    def velocity(state: np.ndarray, time: float) -> np.ndarray:
        features = np.column_stack([state, np.full(state.size, time)])
        return model.predict(features)

    x_t = rng.normal(size=sample_count)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        time = step / num_steps
        k1 = velocity(x_t, time)
        k2 = velocity(x_t + 0.5 * dt * k1, min(time + 0.5 * dt, 1.0))
        k3 = velocity(x_t + 0.5 * dt * k2, min(time + 0.5 * dt, 1.0))
        k4 = velocity(x_t + dt * k3, min(time + dt, 1.0))
        x_t = x_t + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_t


def plot_loss_curves(ax: plt.Axes, train_losses: list[float], val_losses: list[float], title: str) -> None:
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train", color="#8c1c13", linewidth=1.8)
    ax.plot(epochs, val_losses, label="validation", color="#0b7189", linewidth=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()


def plot_sample_comparison(ax: plt.Axes, generated: np.ndarray, test: np.ndarray, title: str) -> None:
    grid = np.linspace(-4.2, 4.2, 500)
    bins = np.linspace(-4.2, 4.2, 45)
    ax.plot(grid, mixture_pdf(grid), color="black", linewidth=1.8, label="target density")
    ax.hist(test, bins=bins, density=True, alpha=0.45, color="#777777", label="test histogram")
    ax.hist(generated, bins=bins, density=True, alpha=0.45, color="#1f77b4", label="generated histogram")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()


def plot_diffusion_scores(ax: plt.Axes, model: MLPRegressor, schedule: DiffusionSchedule, config: BaselineConfig) -> None:
    grid = np.linspace(-4.1, 4.1, 600)
    colors = ["#8c1c13", "#0b7189", "#5b8e7d"]
    for timestep, color in zip(config.score_timesteps, colors, strict=True):
        features = np.column_stack([grid, np.full(grid.size, timestep / schedule.num_steps)])
        epsilon_hat = model.predict(features)
        sigma_t = np.sqrt(1.0 - schedule.alpha_bar[timestep])
        score_hat = -epsilon_hat / sigma_t
        ax.plot(grid, score_hat, color=color, linewidth=2.0, label=fr"$t={timestep}$")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel(r"estimated score")
    ax.set_title(r"Induced score estimate $-\varepsilon_\theta/\sigma_t$")
    ax.legend()


def plot_flow_uxt(ax: plt.Axes, model: MLPRegressor) -> None:
    state_grid = np.linspace(-4.2, 4.2, 220)
    time_grid = np.linspace(0.0, 1.0, 140)
    xx, tt = np.meshgrid(state_grid, time_grid)
    features = np.column_stack([xx.ravel(), tt.ravel()])
    velocity = model.predict(features).reshape(xx.shape)
    image = ax.contourf(state_grid, time_grid, velocity, levels=22, cmap="coolwarm")
    ax.set_xlabel("state x")
    ax.set_ylabel("time t")
    ax.set_title(r"Learned velocity $u_\theta(x,t)$")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="velocity")


def render_diffusion_figure(
    train_losses: list[float],
    val_losses: list[float],
    generated: np.ndarray,
    test: np.ndarray,
    model: MLPRegressor,
    schedule: DiffusionSchedule,
    config: BaselineConfig,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))
    plot_loss_curves(axes[0], train_losses, val_losses, "Diffusion loss")
    plot_sample_comparison(axes[1], generated, test, "Diffusion samples vs test")
    plot_diffusion_scores(axes[2], model, schedule, config)
    path = output_path("exercise1_diffusion_answer.png")
    save_figure(fig, path)
    return path


def render_flow_figure(
    train_losses: list[float],
    val_losses: list[float],
    generated: np.ndarray,
    test: np.ndarray,
    model: MLPRegressor,
    config: BaselineConfig,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.9, 3.8))
    plot_loss_curves(axes[0], train_losses, val_losses, "Flow-matching loss")
    plot_sample_comparison(axes[1], generated, test, "Flow samples vs test")
    plot_flow_uxt(axes[2], model)
    path = output_path("exercise1_flow_answer.png")
    save_figure(fig, path)
    return path


def write_results_tex(config: BaselineConfig, diffusion_w1: float, flow_w1: float) -> Path:
    better_text = (
        "In this baseline, diffusion achieved the lower test-set Wasserstein distance."
        if diffusion_w1 < flow_w1
        else "In this baseline, flow matching achieved the lower test-set Wasserstein distance."
    )
    content = "\n".join(
        [
            rf"\newcommand{{\ExerciseOneTrainSize}}{{{config.train_size}}}",
            rf"\newcommand{{\ExerciseOneValSize}}{{{config.val_size}}}",
            rf"\newcommand{{\ExerciseOneTestSize}}{{{config.test_size}}}",
            rf"\newcommand{{\ExerciseOneEpochs}}{{{config.epochs}}}",
            r"\newcommand{\ExerciseOneArchitecture}{a fully connected MLP with 7 hidden layers of width 16, input $(x,t)$, scalar output, and \texttt{tanh} activations}",
            rf"\newcommand{{\DiffusionWasserstein}}{{{diffusion_w1:.4f}}}",
            rf"\newcommand{{\FlowWasserstein}}{{{flow_w1:.4f}}}",
            rf"\newcommand{{\BetterMethodText}}{{{better_text}}}",
            "",
        ]
    )
    path = output_path("exercise1_results.tex")
    path.write_text(content, encoding="utf-8")
    return path


def write_summary_json(
    config: BaselineConfig,
    diffusion_w1: float,
    flow_w1: float,
    diffusion_losses: tuple[list[float], list[float]],
    flow_losses: tuple[list[float], list[float]],
) -> Path:
    payload = {
        "config": {
            **asdict(config),
            "hidden_layers": list(config.hidden_layers),
            "score_timesteps": list(config.score_timesteps),
        },
        "diffusion": {
            "wasserstein_test": diffusion_w1,
            "train_loss_final": diffusion_losses[0][-1],
            "validation_loss_final": diffusion_losses[1][-1],
        },
        "flow_matching": {
            "wasserstein_test": flow_w1,
            "train_loss_final": flow_losses[0][-1],
            "validation_loss_final": flow_losses[1][-1],
        },
    }
    path = output_path("exercise1_summary.json")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    config = BaselineConfig()
    rng = make_rng(config.seed)
    apply_plot_style()
    ensure_dir(OUTPUT_DIR)

    splits = build_splits(config, rng)
    schedule = build_linear_schedule(config)

    diffusion_train = build_diffusion_pairs(splits["train"], schedule, config.train_pairs, rng)
    diffusion_val = build_diffusion_pairs(splits["val"], schedule, config.val_pairs, rng)
    flow_train = build_flow_pairs(splits["train"], config.train_pairs, rng)
    flow_val = build_flow_pairs(splits["val"], config.val_pairs, rng)

    diffusion_model = make_regressor(config, seed=config.seed)
    flow_model = make_regressor(config, seed=config.seed + 1)

    diffusion_model, diffusion_train_losses, diffusion_val_losses = train_regressor(
        diffusion_model,
        diffusion_train[0],
        diffusion_train[1],
        diffusion_val[0],
        diffusion_val[1],
        epochs=config.epochs,
    )
    flow_model, flow_train_losses, flow_val_losses = train_regressor(
        flow_model,
        flow_train[0],
        flow_train[1],
        flow_val[0],
        flow_val[1],
        epochs=config.epochs,
    )

    diffusion_samples = sample_diffusion(diffusion_model, schedule, config.sample_count, rng)
    flow_samples = sample_flow(flow_model, config.sample_count, config.flow_steps, rng)

    diffusion_w1 = float(wasserstein_distance(diffusion_samples, splits["test"]))
    flow_w1 = float(wasserstein_distance(flow_samples, splits["test"]))

    render_diffusion_figure(
        diffusion_train_losses,
        diffusion_val_losses,
        diffusion_samples,
        splits["test"],
        diffusion_model,
        schedule,
        config,
    )
    render_flow_figure(
        flow_train_losses,
        flow_val_losses,
        flow_samples,
        splits["test"],
        flow_model,
        config,
    )

    write_results_tex(config, diffusion_w1, flow_w1)
    write_summary_json(
        config,
        diffusion_w1,
        flow_w1,
        (diffusion_train_losses, diffusion_val_losses),
        (flow_train_losses, flow_val_losses),
    )

    print(f"diffusion test Wasserstein: {diffusion_w1:.4f}")
    print(f"flow matching test Wasserstein: {flow_w1:.4f}")


if __name__ == "__main__":
    main()
