from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

SEED = 20261029
FIGURE_DPI = 220
NAVY = "#10284b"
BLUE = "#3f7cac"
ORANGE = "#d97706"
GREEN = "#27864a"
GRAY = "#667085"
LIGHT_BLUE = "#dce9f5"
LIGHT_ORANGE = "#f9e4c8"

EXPECTED_FILENAMES = (
    "linear-regression.png",
    "logistic-regression.png",
    "normalization.png",
)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _linear_regression_figure(rng: np.random.Generator, path: Path) -> None:
    x = np.linspace(0.5, 9.5, 24)
    y = 1.65 * x + 2.4 + rng.normal(0.0, 1.35, size=x.size)
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept

    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    axis.scatter(x, y, s=58, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3, label="observations")
    axis.plot(x, prediction, color=ORANGE, linewidth=3.2, label=r"$\hat y = w x + b$")
    for index in (2, 7, 13, 19, 22):
        axis.plot(
            [x[index], x[index]],
            [y[index], prediction[index]],
            color=GRAY,
            linewidth=1.4,
            linestyle="--",
            alpha=0.75,
        )
    axis.text(7.4, 6.2, "residual", color=GRAY, fontsize=14)
    axis.set_xlabel(r"feature $x$", fontsize=16)
    axis.set_ylabel(r"target $y$", fontsize=16)
    axis.legend(frameon=False, fontsize=14, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=12)
    axis.grid(alpha=0.15)
    _save(figure, path)


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-values))


def _logistic_regression_figure(rng: np.random.Generator, path: Path) -> None:
    negative = rng.multivariate_normal((-1.4, -0.7), ((0.65, 0.18), (0.18, 0.55)), size=42)
    positive = rng.multivariate_normal((1.2, 0.9), ((0.7, 0.12), (0.12, 0.6)), size=42)
    grid_x, grid_y = np.meshgrid(np.linspace(-4, 4, 260), np.linspace(-3.5, 3.5, 240))
    logits = 1.25 * grid_x + 1.05 * grid_y - 0.1
    probabilities = _sigmoid(logits)
    color_map = LinearSegmentedColormap.from_list("class_probability", (LIGHT_BLUE, "#ffffff", LIGHT_ORANGE))

    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    background = axis.contourf(grid_x, grid_y, probabilities, levels=np.linspace(0.0, 1.0, 21), cmap=color_map)
    axis.contour(grid_x, grid_y, probabilities, levels=(0.5,), colors=(NAVY,), linewidths=2.6)
    axis.scatter(negative[:, 0], negative[:, 1], s=54, color=BLUE, edgecolor="white", linewidth=0.7, label=r"$y=0$")
    axis.scatter(positive[:, 0], positive[:, 1], s=54, color=ORANGE, edgecolor="white", linewidth=0.7, label=r"$y=1$")
    axis.text(-0.2, 2.8, r"decision boundary: $\hat p=0.5$", color=NAVY, fontsize=14, rotation=-49)
    color_bar = figure.colorbar(background, ax=axis, pad=0.02)
    color_bar.set_label(r"predicted probability $\hat p(y=1\mid\mathbf{x})$", fontsize=13)
    color_bar.ax.tick_params(labelsize=11)
    axis.set_xlabel(r"feature $x_1$", fontsize=16)
    axis.set_ylabel(r"feature $x_2$", fontsize=16)
    axis.legend(frameon=False, fontsize=14, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=12)
    _save(figure, path)


def _normalization_figure(rng: np.random.Generator, path: Path) -> None:
    latent = rng.multivariate_normal((0.0, 0.0), ((1.0, 0.72), (0.72, 1.0)), size=90)
    raw = np.column_stack((45.0 + 18.0 * latent[:, 0], 1.8 + 0.35 * latent[:, 1]))
    normalized = (raw - raw.mean(axis=0)) / raw.std(axis=0)

    figure, (raw_axis, normalized_axis) = plt.subplots(1, 2, figsize=(10.6, 5.1))
    raw_axis.scatter(raw[:, 0], raw[:, 1], s=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    raw_axis.set_title("Different scales", color=NAVY, fontsize=17, weight="bold")
    raw_axis.set_xlabel(r"feature $x_1$", fontsize=14)
    raw_axis.set_ylabel(r"feature $x_2$", fontsize=14)
    raw_axis.tick_params(labelsize=11)

    normalized_axis.scatter(
        normalized[:, 0],
        normalized[:, 1],
        s=40,
        color=GREEN,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    normalized_axis.axhline(0.0, color=GRAY, linewidth=1.0, alpha=0.5)
    normalized_axis.axvline(0.0, color=GRAY, linewidth=1.0, alpha=0.5)
    normalized_axis.set_title("Zero mean, unit scale", color=NAVY, fontsize=17, weight="bold")
    normalized_axis.set_xlabel(r"normalized $\tilde x_1$", fontsize=14)
    normalized_axis.set_ylabel(r"normalized $\tilde x_2$", fontsize=14)
    normalized_axis.tick_params(labelsize=11)

    for axis in (raw_axis, normalized_axis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.12)

    figure.text(0.5, 0.5, r"$\longrightarrow$", color=ORANGE, fontsize=30, ha="center", va="center")
    figure.subplots_adjust(wspace=0.42)
    _save(figure, path)


def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    outputs = tuple(output_directory / filename for filename in EXPECTED_FILENAMES)

    _linear_regression_figure(rng, outputs[0])
    _logistic_regression_figure(rng, outputs[1])
    _normalization_figure(rng, outputs[2])

    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    return outputs


def main() -> None:
    """Write the supervised-learning figures under assets/generated/."""
    output_directory = Path(__file__).resolve().parents[1] / "assets" / "generated"
    generate_figures(output_directory)


if __name__ == "__main__":
    main()
