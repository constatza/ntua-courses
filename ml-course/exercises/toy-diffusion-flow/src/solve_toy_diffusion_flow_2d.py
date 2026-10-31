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
from scipy.stats import gaussian_kde, wasserstein_distance_nd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "assets" / "generated"

SPIRAL_T_MIN = 1.0
SPIRAL_T_MAX = 4.0 * np.pi
SPIRAL_SCALE = 0.18
SPIRAL_NOISE_STD = 0.08


@dataclass(frozen=True)
class BaselineConfig:
    seed: int = 20260405
    train_size: int = 3072
    val_size: int = 1536
    test_size: int = 320
    train_pairs: int = 12288
    val_pairs: int = 6144
    sample_count: int = 320
    epochs: int = 500
    hidden_layers: tuple[int, int, int, int, int, int] = (12, 12, 12, 12, 12, 12)
    diffusion_steps: int = 50
    diffusion_beta_start: float = 1.0e-4
    diffusion_beta_end: float = 2.0e-2
    flow_steps: int = 70
    score_timestep: int = 20
    flow_timestep: float = 0.5


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


def sample_swirl(rng: np.random.Generator, size: int) -> np.ndarray:
    phi = rng.uniform(SPIRAL_T_MIN, SPIRAL_T_MAX, size=size)
    radius = SPIRAL_SCALE * phi
    x = radius * np.cos(phi) + SPIRAL_NOISE_STD * rng.normal(size=size)
    y = radius * np.sin(phi) + SPIRAL_NOISE_STD * rng.normal(size=size)
    return np.column_stack([x, y])


def build_splits(config: BaselineConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "train": sample_swirl(rng, config.train_size),
        "val": sample_swirl(rng, config.val_size),
        "test": sample_swirl(rng, config.test_size),
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
    indices = rng.integers(0, samples.shape[0], size=pair_count)
    x0 = samples[indices]
    timestep = rng.integers(1, schedule.num_steps + 1, size=pair_count)
    epsilon = rng.normal(size=(pair_count, 2))
    signal = np.sqrt(schedule.alpha_bar[timestep])[:, None]
    sigma = np.sqrt(1.0 - schedule.alpha_bar[timestep])[:, None]
    xt = signal * x0 + sigma * epsilon
    features = np.column_stack([xt, timestep / schedule.num_steps])
    return features, epsilon


def build_flow_pairs(
    target_samples: np.ndarray,
    pair_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.integers(0, target_samples.shape[0], size=pair_count)
    x1 = target_samples[indices]
    x0 = rng.normal(size=(pair_count, 2))
    time = rng.uniform(0.0, 1.0, size=pair_count)
    xt = (1.0 - time)[:, None] * x0 + time[:, None] * x1
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
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(diff * diff))


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
    x_t = rng.normal(size=(sample_count, 2))
    for timestep in range(schedule.num_steps, 0, -1):
        features = np.column_stack([x_t, np.full(sample_count, timestep / schedule.num_steps)])
        epsilon_hat = model.predict(features)
        sigma_t = np.sqrt(1.0 - schedule.alpha_bar[timestep])
        x0_hat = (x_t - sigma_t * epsilon_hat) / np.sqrt(schedule.alpha_bar[timestep])
        x0_hat = np.clip(x0_hat, -3.5, 3.5)
        x_t = diffusion_posterior_mean(x_t, x0_hat, schedule, timestep)
    return x_t


def sample_flow(
    model: MLPRegressor,
    sample_count: int,
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    def velocity(state: np.ndarray, time: float) -> np.ndarray:
        features = np.column_stack([state, np.full(state.shape[0], time)])
        return model.predict(features)

    x_t = rng.normal(size=(sample_count, 2))
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
    ax.plot(epochs, train_losses, color="#111111", linewidth=1.8, label="train")
    ax.plot(epochs, val_losses, color="#1f77b4", linewidth=1.8, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(title)
    ax.legend()


def kde_density(points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    estimator = gaussian_kde(points.T)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=0)
    return estimator(grid_points).reshape(grid_x.shape)


def plot_sample_comparison(ax: plt.Axes, generated: np.ndarray, test: np.ndarray, title: str) -> None:
    axis = np.linspace(-2.9, 2.9, 160)
    grid_x, grid_y = np.meshgrid(axis, axis)
    test_density = kde_density(test, grid_x, grid_y)
    generated_density = kde_density(generated, grid_x, grid_y)
    ax.scatter(test[:, 0], test[:, 1], s=8, alpha=0.22, color="#555555", label="test points")
    ax.scatter(generated[:, 0], generated[:, 1], s=8, alpha=0.22, color="#1f77b4", label="generated points")
    ax.contour(grid_x, grid_y, test_density, levels=5, colors="#555555", linewidths=0.9)
    ax.contour(grid_x, grid_y, generated_density, levels=5, colors="#1f77b4", linewidths=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper right")


def plot_diffusion_score_field(ax: plt.Axes, model: MLPRegressor, schedule: DiffusionSchedule, config: BaselineConfig, overlay: np.ndarray) -> None:
    axis = np.linspace(-2.6, 2.6, 15)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    features = np.column_stack([points, np.full(points.shape[0], config.score_timestep / schedule.num_steps)])
    epsilon_hat = model.predict(features)
    sigma_t = np.sqrt(1.0 - schedule.alpha_bar[config.score_timestep])
    score_hat = -epsilon_hat / sigma_t
    ax.scatter(overlay[:, 0], overlay[:, 1], s=8, alpha=0.18, color="#555555")
    ax.quiver(
        grid_x,
        grid_y,
        score_hat[:, 0].reshape(grid_x.shape),
        score_hat[:, 1].reshape(grid_y.shape),
        color="#1f77b4",
        scale=26.0,
        width=0.004,
        pivot="mid",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(fr"Induced score field at $t={config.score_timestep}$")


def plot_flow_velocity_field(ax: plt.Axes, model: MLPRegressor, config: BaselineConfig, overlay: np.ndarray) -> None:
    axis = np.linspace(-2.6, 2.6, 15)
    grid_x, grid_y = np.meshgrid(axis, axis)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    features = np.column_stack([points, np.full(points.shape[0], config.flow_timestep)])
    velocity = model.predict(features)
    ax.scatter(overlay[:, 0], overlay[:, 1], s=8, alpha=0.18, color="#555555")
    ax.quiver(
        grid_x,
        grid_y,
        velocity[:, 0].reshape(grid_x.shape),
        velocity[:, 1].reshape(grid_y.shape),
        color="#1f77b4",
        scale=18.0,
        width=0.004,
        pivot="mid",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(fr"Velocity field at $t={config.flow_timestep:.1f}$")


def render_diffusion_figure(
    train_losses: list[float],
    val_losses: list[float],
    generated: np.ndarray,
    test: np.ndarray,
    model: MLPRegressor,
    schedule: DiffusionSchedule,
    config: BaselineConfig,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    plot_loss_curves(axes[0], train_losses, val_losses, "Diffusion loss")
    plot_sample_comparison(axes[1], generated, test, "Diffusion samples vs test")
    plot_diffusion_score_field(axes[2], model, schedule, config, overlay=test[:220])
    path = output_path("exercise2_diffusion_answer.png")
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
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
    plot_loss_curves(axes[0], train_losses, val_losses, "Flow-matching loss")
    plot_sample_comparison(axes[1], generated, test, "Flow samples vs test")
    plot_flow_velocity_field(axes[2], model, config, overlay=test[:220])
    path = output_path("exercise2_flow_answer.png")
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
            rf"\newcommand{{\ExerciseTwoTrainSize}}{{{config.train_size}}}",
            rf"\newcommand{{\ExerciseTwoValSize}}{{{config.val_size}}}",
            rf"\newcommand{{\ExerciseTwoTestSize}}{{{config.test_size}}}",
            rf"\newcommand{{\ExerciseTwoEpochs}}{{{config.epochs}}}",
            r"\newcommand{\ExerciseTwoArchitecture}{a fully connected MLP with 6 hidden layers of width 12, input $(x_1,x_2,t)$, two outputs, and \texttt{tanh} activations}",
            rf"\newcommand{{\DiffusionWassersteinTwoD}}{{{diffusion_w1:.4f}}}",
            rf"\newcommand{{\FlowWassersteinTwoD}}{{{flow_w1:.4f}}}",
            rf"\newcommand{{\BetterMethodTextTwoD}}{{{better_text}}}",
            "",
        ]
    )
    path = output_path("exercise2_results.tex")
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
    path = output_path("exercise2_summary.json")
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

    diffusion_w1 = float(wasserstein_distance_nd(diffusion_samples, splits["test"]))
    flow_w1 = float(wasserstein_distance_nd(flow_samples, splits["test"]))

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
