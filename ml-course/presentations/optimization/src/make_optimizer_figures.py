from __future__ import annotations

from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

SEED = 20261029
FIGURE_DPI = 220
NAVY = "#10284b"
BLUE = "#3f7cac"
ORANGE = "#d97706"
GREEN = "#27864a"
PURPLE = "#7656a3"
RED = "#b5473c"
GRAY = "#667085"

ROTATION_ANGLE = np.deg2rad(28.0)
ROTATION = np.array(
    [
        [np.cos(ROTATION_ANGLE), -np.sin(ROTATION_ANGLE)],
        [np.sin(ROTATION_ANGLE), np.cos(ROTATION_ANGLE)],
    ]
)
HESSIAN = ROTATION @ np.diag((1.0, 18.0)) @ ROTATION.T
INITIAL_POINT = np.array((-3.0, 2.6))
OPTIMUM = np.zeros(2)

EXPECTED_FILENAMES = (
    "objective-geometry.png",
    "learning-rates.png",
    "conditioning.png",
    "gd-vs-sgd.png",
    "momentum.png",
    "rmsprop.png",
    "optimizer-comparison.png",
)


class OptimizerName(str, Enum):
    GD = "GD"
    SGD = "SGD"
    MOMENTUM = "Momentum"
    RMSPROP = "RMSProp"
    ADAM = "Adam"


def quadratic_value(parameters: NDArray[np.float64]) -> float:
    return float(0.5 * parameters @ HESSIAN @ parameters)


def quadratic_gradient(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    return HESSIAN @ parameters


def _gradient_descent(learning_rate: float, steps: int) -> NDArray[np.float64]:
    parameters = INITIAL_POINT.copy()
    trajectory = [parameters.copy()]
    for _ in range(steps):
        parameters = parameters - learning_rate * quadratic_gradient(parameters)
        trajectory.append(parameters.copy())
    return np.asarray(trajectory)


def run_optimizer(optimizer: OptimizerName, steps: int, seed: int) -> NDArray[np.float64]:
    parameters = INITIAL_POINT.copy()
    trajectory = [parameters.copy()]
    rng = np.random.default_rng(seed)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)

    for iteration in range(1, steps + 1):
        gradient = quadratic_gradient(parameters)

        match optimizer:
            case OptimizerName.GD:
                parameters = parameters - 0.075 * gradient
            case OptimizerName.SGD:
                stochastic_gradient = gradient + rng.normal(0.0, 1.4, size=parameters.shape)
                parameters = parameters - 0.035 * stochastic_gradient
            case OptimizerName.MOMENTUM:
                first_moment = 0.82 * first_moment + gradient
                parameters = parameters - 0.012 * first_moment
            case OptimizerName.RMSPROP:
                second_moment = 0.9 * second_moment + 0.1 * gradient**2
                parameters = parameters - 0.09 * gradient / (np.sqrt(second_moment) + 1e-8)
            case OptimizerName.ADAM:
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient**2
                corrected_first = first_moment / (1.0 - 0.9**iteration)
                corrected_second = second_moment / (1.0 - 0.999**iteration)
                parameters = parameters - 0.11 * corrected_first / (np.sqrt(corrected_second) + 1e-8)

        trajectory.append(parameters.copy())

    return np.asarray(trajectory)


def _objective_grid(
    matrix: NDArray[np.float64] = HESSIAN,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    horizontal, vertical = np.meshgrid(np.linspace(-4.2, 4.2, 360), np.linspace(-3.6, 3.6, 320))
    points = np.stack((horizontal, vertical), axis=-1)
    values = 0.5 * np.einsum("...i,ij,...j->...", points, matrix, points)
    return horizontal, vertical, values


def _prepare_axis(axis: plt.Axes, matrix: NDArray[np.float64] = HESSIAN) -> None:
    horizontal, vertical, values = _objective_grid(matrix)
    levels = np.geomspace(0.35, 110.0, 13)
    axis.contour(horizontal, vertical, values, levels=levels, colors=BLUE, linewidths=1.15, alpha=0.72)
    axis.scatter(*OPTIMUM, s=90, color=GREEN, edgecolor="white", linewidth=1.0, zorder=8)
    axis.text(0.13, 0.12, "minimum", color=GREEN, fontsize=12, weight="bold")
    axis.set_xlim(-4.2, 4.2)
    axis.set_ylim(-3.6, 3.6)
    axis.set_xlabel(r"parameter $\theta_1$", fontsize=14)
    axis.set_ylabel(r"parameter $\theta_2$", fontsize=14)
    axis.tick_params(labelsize=10)
    axis.set_aspect("equal", adjustable="box")
    axis.spines[["top", "right"]].set_visible(False)


def _plot_trajectory(
    axis: plt.Axes,
    trajectory: NDArray[np.float64],
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color=color,
        linewidth=2.2,
        marker=marker,
        markersize=3.2,
        markevery=max(1, len(trajectory) // 14),
        label=label,
        zorder=6,
    )
    axis.scatter(*trajectory[0], s=72, color=color, edgecolor="white", linewidth=0.8, zorder=7)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _objective_geometry(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    _prepare_axis(axis)
    point = np.array((1.5, 1.55))
    gradient = quadratic_gradient(point)
    direction = gradient / np.linalg.norm(gradient)
    axis.scatter(*point, s=85, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=8)
    axis.arrow(
        point[0],
        point[1],
        0.9 * direction[0],
        0.9 * direction[1],
        width=0.025,
        head_width=0.17,
        color=RED,
        length_includes_head=True,
        zorder=9,
    )
    axis.arrow(
        point[0],
        point[1],
        -0.9 * direction[0],
        -0.9 * direction[1],
        width=0.025,
        head_width=0.17,
        color=GREEN,
        length_includes_head=True,
        zorder=9,
    )
    axis.text(point[0] + 0.45, point[1] + 0.75, r"$\nabla_{\theta}\mathcal{L}$", color=RED, fontsize=15)
    axis.text(point[0] - 1.35, point[1] - 0.7, r"$-\nabla_{\theta}\mathcal{L}$", color=GREEN, fontsize=15)
    _save(figure, path)


def _learning_rates(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    settings = (
        (0.008, "too small", GRAY),
        (0.075, "useful", GREEN),
        (0.13, "too large", RED),
    )
    for learning_rate, label, color in settings:
        trajectory = _gradient_descent(learning_rate, 36)
        objective_values = [quadratic_value(parameters) for parameters in trajectory]
        axis.semilogy(
            objective_values,
            color=color,
            linewidth=3.0,
            label=rf"$\eta={learning_rate}$: {label}",
        )
    axis.set_xlabel("update", fontsize=14)
    axis.set_ylabel(r"objective $\mathcal{L}(\boldsymbol{\theta}_t)$", fontsize=14)
    axis.tick_params(labelsize=10)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=12, loc="upper left")
    _save(figure, path)


def _conditioning(path: Path) -> None:
    well_conditioned = np.eye(2) * 4.0
    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(10.6, 5.1))
    _prepare_axis(left_axis, well_conditioned)
    left_axis.set_title("well-conditioned", color=NAVY, fontsize=16, weight="bold")
    circular_trajectory = [INITIAL_POINT.copy()]
    point = INITIAL_POINT.copy()
    for _ in range(18):
        point = point - 0.16 * well_conditioned @ point
        circular_trajectory.append(point.copy())
    _plot_trajectory(left_axis, np.asarray(circular_trajectory), "GD", ORANGE)

    _prepare_axis(right_axis)
    right_axis.set_title("ill-conditioned", color=NAVY, fontsize=16, weight="bold")
    _plot_trajectory(right_axis, run_optimizer(OptimizerName.GD, 28, SEED), "GD", ORANGE)
    for axis in (left_axis, right_axis):
        axis.legend(frameon=False, fontsize=11, loc="upper left")
    figure.subplots_adjust(wspace=0.28)
    _save(figure, path)


def _comparison(
    path: Path,
    methods: tuple[tuple[OptimizerName, str], ...],
    steps: int = 55,
) -> None:
    colors = {
        OptimizerName.GD: ORANGE,
        OptimizerName.SGD: GRAY,
        OptimizerName.MOMENTUM: GREEN,
        OptimizerName.RMSPROP: PURPLE,
        OptimizerName.ADAM: RED,
    }
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    _prepare_axis(axis)
    for method, label in methods:
        _plot_trajectory(axis, run_optimizer(method, steps, SEED), label, colors[method])
    axis.legend(frameon=False, fontsize=11, loc="upper left")
    _save(figure, path)


def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_directory / filename for filename in EXPECTED_FILENAMES)

    _objective_geometry(outputs[0])
    _learning_rates(outputs[1])
    _conditioning(outputs[2])
    _comparison(outputs[3], ((OptimizerName.GD, "full-batch GD"), (OptimizerName.SGD, "SGD")))
    _comparison(outputs[4], ((OptimizerName.GD, "GD"), (OptimizerName.MOMENTUM, "momentum")))
    _comparison(outputs[5], ((OptimizerName.GD, "GD"), (OptimizerName.RMSPROP, "RMSProp")))
    _comparison(
        outputs[6],
        (
            (OptimizerName.GD, "GD"),
            (OptimizerName.SGD, "SGD"),
            (OptimizerName.MOMENTUM, "momentum"),
            (OptimizerName.RMSPROP, "RMSProp"),
            (OptimizerName.ADAM, "Adam"),
        ),
    )

    trajectories = tuple(run_optimizer(method, 80, SEED) for method in OptimizerName)
    assert all(np.allclose(trajectory[0], INITIAL_POINT) for trajectory in trajectories)
    assert all(quadratic_value(trajectory[-1]) < quadratic_value(trajectory[0]) for trajectory in trajectories)
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    return outputs


def main() -> None:
    """Run cheap optimizer experiments and write all approved figures."""
    output_directory = Path(__file__).resolve().parents[1] / "assets" / "generated"
    generate_figures(output_directory)


if __name__ == "__main__":
    main()
