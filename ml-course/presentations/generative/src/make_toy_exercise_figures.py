#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax[cpu]",
#   "matplotlib",
#   "scipy",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import erf, erfinv
from scipy.stats import gaussian_kde


jax.config.update("jax_enable_x64", True)

ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT_DIR.parent.parent / "exercises" / "toy-diffusion-flow" / "assets" / "generated"

RNG_SEED = 20260403

MIXTURE_WEIGHTS = np.array([0.58, 0.42], dtype=float)
MIXTURE_MEANS = np.array([-1.6, 1.8], dtype=float)
MIXTURE_STDS = np.array([0.35, 0.55], dtype=float)

SPIRAL_T_MIN = 1.0
SPIRAL_T_MAX = 4.0 * np.pi
SPIRAL_SCALE = 0.18
SPIRAL_NOISE_STD = 0.08


def generated_path(name: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_DIR / name


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def normal_pdf_1d(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = jnp.asarray(x, dtype=jnp.float64)
    variance = std**2
    return jnp.exp(-0.5 * ((x - mean) ** 2) / variance) / jnp.sqrt(2.0 * jnp.pi * variance)


def normal_cdf_1d(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = jnp.asarray(x, dtype=jnp.float64)
    return 0.5 * (1.0 + erf((x - mean) / (std * jnp.sqrt(2.0))))


def mixture_pdf_1d(x: np.ndarray) -> np.ndarray:
    density = np.zeros_like(x, dtype=float)
    for weight, mean, std in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_STDS, strict=True):
        density += weight * normal_pdf_1d(x, float(mean), float(std))
    return density


def mixture_cdf_1d(x: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(x, dtype=float)
    for weight, mean, std in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_STDS, strict=True):
        cumulative += weight * normal_cdf_1d(x, float(mean), float(std))
    return cumulative


def sample_mixture_1d(rng: np.random.Generator, n: int) -> np.ndarray:
    components = rng.choice(len(MIXTURE_WEIGHTS), size=n, p=MIXTURE_WEIGHTS)
    means = MIXTURE_MEANS[components]
    stds = MIXTURE_STDS[components]
    return rng.normal(loc=means, scale=stds)


def smoothed_score_1d(x: np.ndarray, noise_std: float) -> np.ndarray:
    component_densities = []
    component_scores = []
    for weight, mean, std in zip(MIXTURE_WEIGHTS, MIXTURE_MEANS, MIXTURE_STDS, strict=True):
        variance = std**2 + noise_std**2
        density = weight * np.exp(-0.5 * ((x - mean) ** 2) / variance) / np.sqrt(2.0 * np.pi * variance)
        score = (mean - x) / variance
        component_densities.append(density)
        component_scores.append(score)

    stacked_densities = np.stack(component_densities, axis=0)
    responsibilities = stacked_densities / np.sum(stacked_densities, axis=0, keepdims=True)
    stacked_scores = np.stack(component_scores, axis=0)
    return np.sum(responsibilities * stacked_scores, axis=0)


def normal_ppf(probabilities: np.ndarray) -> np.ndarray:
    probabilities = jnp.asarray(probabilities, dtype=jnp.float64)
    return jnp.sqrt(2.0) * erfinv(2.0 * probabilities - 1.0)


def build_1d_transport(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_samples = normal_ppf(probabilities)
    support = np.linspace(-4.8, 4.8, 6000)
    target_cdf = mixture_cdf_1d(support)
    target_samples = np.interp(probabilities, target_cdf, support)
    velocities = target_samples - source_samples
    return source_samples, target_samples, velocities


def render_1d_distribution(rng: np.random.Generator) -> None:
    x_grid = np.linspace(-4.2, 4.2, 500)
    train_samples = sample_mixture_1d(rng, 2500)
    test_samples = sample_mixture_1d(rng, 2500)
    pdf = mixture_pdf_1d(x_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    axes[0].hist(train_samples, bins=60, density=True, alpha=0.55, color="#5b8e7d", label="example train samples")
    axes[0].plot(x_grid, pdf, color="#8c1c13", linewidth=2.0, label="target density")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].set_title("1D bimodal target")
    axes[0].legend()

    axes[1].hist(
        test_samples,
        bins=60,
        density=True,
        alpha=0.5,
        color="#0b7189",
        label="example test samples",
    )
    axes[1].plot(x_grid, pdf, color="#8c1c13", linewidth=2.0, label="target density")
    for weight, mean, std, color in zip(
        MIXTURE_WEIGHTS,
        MIXTURE_MEANS,
        MIXTURE_STDS,
        ["#f0a202", "#7d4e57"],
        strict=True,
    ):
        axes[1].plot(
            x_grid,
            weight * normal_pdf_1d(x_grid, float(mean), float(std)),
            color=color,
            linewidth=1.4,
            linestyle="--",
        )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density")
    axes[1].set_title("Train/test PDF-style view")
    axes[1].legend()

    save_figure(fig, generated_path("toy_1d_distribution.png"))


def render_1d_fields() -> None:
    x_grid = np.linspace(-4.1, 4.1, 600)
    noise_levels = [0.15, 0.45, 0.9]
    score_colors = ["#8c1c13", "#0b7189", "#5b8e7d"]

    probabilities = np.linspace(1.0e-4, 1.0 - 1.0e-4, 2400)
    source_samples, target_samples, velocities = build_1d_transport(probabilities)
    time_grid = np.linspace(0.02, 0.98, 120)
    state_grid = np.linspace(-4.2, 4.2, 360)

    field_rows = []
    for t in time_grid:
        xt = (1.0 - t) * source_samples + t * target_samples
        row = np.interp(state_grid, xt, velocities, left=np.nan, right=np.nan)
        field_rows.append(row)
    field = np.stack(field_rows, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    for noise_std, color in zip(noise_levels, score_colors, strict=True):
        axes[0].plot(
            x_grid,
            smoothed_score_1d(x_grid, noise_std=noise_std),
            linewidth=2.0,
            color=color,
            label=fr"$\sigma={noise_std:.2f}$",
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"score $\nabla_x \log p_t(x)$")
    axes[0].set_title("Diffusion-style score view")
    axes[0].legend()

    contour = axes[1].contourf(time_grid, state_grid, field.T, levels=18, cmap="coolwarm")
    chosen_probs = np.array([0.06, 0.22, 0.40, 0.58, 0.76, 0.94])
    chosen_source, chosen_target, _ = build_1d_transport(chosen_probs)
    trajectory_times = np.linspace(0.0, 1.0, 120)
    for source_value, target_value in zip(chosen_source, chosen_target, strict=True):
        trajectory = (1.0 - trajectory_times) * source_value + trajectory_times * target_value
        axes[1].plot(trajectory_times, trajectory, color="black", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("state x")
    axes[1].set_title(r"Flow-style $u(x,t)$ view")
    fig.colorbar(contour, ax=axes[1], fraction=0.046, pad=0.04, label="velocity")

    save_figure(fig, generated_path("toy_1d_fields.png"))


def sample_swirl_2d(rng: np.random.Generator, n: int) -> np.ndarray:
    phi = rng.uniform(SPIRAL_T_MIN, SPIRAL_T_MAX, size=n)
    radius = SPIRAL_SCALE * phi
    x = radius * np.cos(phi) + SPIRAL_NOISE_STD * rng.normal(size=n)
    y = radius * np.sin(phi) + SPIRAL_NOISE_STD * rng.normal(size=n)
    return np.column_stack([x, y])


def swirl_centerline(num_points: int = 900) -> tuple[np.ndarray, np.ndarray]:
    phi = np.linspace(SPIRAL_T_MIN, SPIRAL_T_MAX, num_points)
    radius = SPIRAL_SCALE * phi
    points = np.column_stack([radius * np.cos(phi), radius * np.sin(phi)])
    tangent = np.column_stack(
        [
            SPIRAL_SCALE * np.cos(phi) - radius * np.sin(phi),
            SPIRAL_SCALE * np.sin(phi) + radius * np.cos(phi),
        ]
    )
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-8)
    return points, tangent


def nearest_swirl_velocity(points: np.ndarray) -> np.ndarray:
    curve_points, curve_tangent = swirl_centerline()
    offsets = points[:, None, :] - curve_points[None, :, :]
    distances = np.sum(offsets * offsets, axis=2)
    indices = np.argmin(distances, axis=1)
    nearest = curve_points[indices]
    tangent = curve_tangent[indices]
    return 0.85 * (nearest - points) + 0.25 * tangent


def render_2d_distribution(rng: np.random.Generator) -> None:
    samples = sample_swirl_2d(rng, 3500)
    axis = np.linspace(-2.9, 2.9, 220)
    xx, yy = np.meshgrid(axis, axis)
    estimator = gaussian_kde(samples.T)
    density = estimator(np.stack([xx.ravel(), yy.ravel()], axis=0)).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    contour = ax.contourf(xx, yy, density, levels=20, cmap="YlGnBu", alpha=0.66)
    shown = samples[::14]
    ax.scatter(shown[:, 0], shown[:, 1], s=12, alpha=0.72, color="white", edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("2D swirl KDE with example points")
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="density")

    save_figure(fig, generated_path("toy_2d_distribution.png"))


def render_2d_fields(rng: np.random.Generator) -> None:
    dense_axis = np.linspace(-2.9, 2.9, 220)
    dense_x, dense_y = np.meshgrid(dense_axis, dense_axis)
    samples = sample_swirl_2d(rng, 4500)
    estimator = gaussian_kde(samples.T)
    grid_points = np.stack([dense_x.ravel(), dense_y.ravel()], axis=0)
    density = estimator(grid_points).reshape(dense_x.shape)
    spacing = dense_axis[1] - dense_axis[0]
    log_density = np.log(np.maximum(density, 1.0e-10))
    grad_y, grad_x = np.gradient(log_density, spacing, spacing)

    coarse_indices = np.linspace(10, dense_axis.size - 11, 18, dtype=int)
    coarse_x = dense_x[np.ix_(coarse_indices, coarse_indices)]
    coarse_y = dense_y[np.ix_(coarse_indices, coarse_indices)]
    score_x = grad_x[np.ix_(coarse_indices, coarse_indices)]
    score_y = grad_y[np.ix_(coarse_indices, coarse_indices)]
    coarse_points = np.column_stack([coarse_x.ravel(), coarse_y.ravel()])
    flow_velocity = nearest_swirl_velocity(coarse_points)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    filled = axes[0].contourf(dense_x, dense_y, density, levels=18, cmap="YlGnBu")
    axes[0].quiver(
        coarse_x,
        coarse_y,
        score_x,
        score_y,
        color="#8c1c13",
        scale=24.0,
        width=0.004,
        pivot="mid",
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_title("Diffusion-style score field")
    fig.colorbar(filled, ax=axes[0], fraction=0.046, pad=0.04, label="density")

    axes[1].scatter(samples[::20, 0], samples[::20, 1], s=7, alpha=0.14, color="#555555")
    axes[1].quiver(
        coarse_x,
        coarse_y,
        flow_velocity[:, 0].reshape(coarse_x.shape),
        flow_velocity[:, 1].reshape(coarse_y.shape),
        color="#0b7189",
        scale=18.0,
        width=0.004,
        pivot="mid",
    )
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$x_2$")
    axes[1].set_aspect("equal")
    axes[1].set_title("Flow-style velocity field")

    save_figure(fig, generated_path("toy_2d_fields.png"))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    apply_plot_style()
    render_1d_distribution(rng)
    render_1d_fields()
    render_2d_distribution(rng)
    render_2d_fields(rng)


if __name__ == "__main__":
    main()
