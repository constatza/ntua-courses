# /// script
# dependencies = ["matplotlib", "numpy"]
# ///
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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

EXPECTED_FILENAMES = (
    "fit-regimes.png",
    "learning-curves.png",
    "bias-variance-targets.png",
    "bias-variance.png",
    "l2-one-weight.png",
    "l2-contours.png",
    "weight-priors.png",
    "l1-l2-geometry.png",
    "regularization-path.png",
)

FIT_DEGREES = (1, 3, 9)
FIT_TRAIN_SIZE = 12
FIT_NOISE = 0.25
FIT_YLIM = 2.0

# Schematic target: how far the cloud centre sits from the best weights w*, and how
# widely the fits scatter. Values are in arbitrary weight units and only need to be
# visibly low versus high.
LOW_BIAS, HIGH_BIAS = 0.0, 1.0
LOW_SPREAD, HIGH_SPREAD = 0.17, 0.55
TARGET_FITS_PER_PANEL = 40
TARGET_RING_RADII = (0.4, 0.8, 1.2, 1.6)
TARGET_LIMIT = 2.0
BIAS_DIRECTION = np.array([0.8, -0.6])

L2_UNREGULARIZED = np.array([2.0, 1.0])
L2_STRENGTH = 2.0
L2_XLIM = (-1.2, 3.2)
L2_YLIM = (-1.7, 2.7)
# Contour heights of the summed loss above its own minimum, so the picture reads "how much
# worse than the best compromise" rather than showing arbitrary levels.
L2_TOTAL_LEVEL_OFFSETS = (0.25, 0.75, 1.5, 3.0)
ONE_WEIGHT_UNREGULARIZED = 2.0

# Constraint view: loss centre outside both shapes, shapes of radius 1.
GEOMETRY_CENTRE = np.array([1.6, 0.2])
GEOMETRY_CURVATURE_Y = 2.2
BALL_RADIUS = 1.0


@dataclass(frozen=True, slots=True)
class TargetPanel:
    """One bias/variance regime: offset of the cloud centre and its scatter."""

    bias: float
    spread: float

    def sample_weights(self, rng: np.random.Generator, count: int) -> NDArray[np.float64]:
        """Fitted weights (w1, w2), one row per training set, around w* = 0."""
        centre = self.bias * BIAS_DIRECTION
        return centre + rng.normal(0.0, self.spread, size=(count, 2))


def l2_solution(unregularized: NDArray[np.float64], strength: float) -> NDArray[np.float64]:
    return unregularized / (1.0 + strength)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _fit_regimes(path: Path, rng: np.random.Generator) -> None:
    x_train = np.sort(rng.uniform(-1.0, 1.0, FIT_TRAIN_SIZE))
    y_train = np.sin(np.pi * x_train) + rng.normal(0.0, FIT_NOISE, x_train.size)
    grid = np.linspace(-1.0, 1.0, 400)
    titles = ("underfit: too rigid", "good fit", "overfit: chases noise")
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    for axis, degree, title in zip(axes, FIT_DEGREES, titles, strict=True):
        coefficients = np.polyfit(x_train, y_train, degree)
        axis.plot(grid, np.sin(np.pi * grid), color=GRAY, linestyle="--", linewidth=1.8, label="true function")
        axis.plot(grid, np.polyval(coefficients, grid), color=ORANGE, linewidth=2.8, label=f"degree-{degree} fit")
        axis.scatter(x_train, y_train, s=46, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4)
        axis.set_ylim(-FIT_YLIM, FIT_YLIM)
        axis.set_title(title, color=NAVY, fontsize=15, weight="bold")
        axis.set_xlabel(r"input $x$", fontsize=13)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=10, loc="upper right")
    axes[0].set_ylabel(r"output $y$", fontsize=13)
    figure.subplots_adjust(wspace=0.08)
    _save(figure, path)


def _bias_variance_targets(path: Path, rng: np.random.Generator) -> None:
    regimes = (
        (TargetPanel(LOW_BIAS, LOW_SPREAD), "low bias, low variance"),
        (TargetPanel(LOW_BIAS, HIGH_SPREAD), "low bias, high variance"),
        (TargetPanel(HIGH_BIAS, LOW_SPREAD), "high bias, low variance"),
        (TargetPanel(HIGH_BIAS, HIGH_SPREAD), "high bias, high variance"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 8.0), sharex=True, sharey=True)
    for axis, (panel, title) in zip(axes.flat, regimes, strict=True):
        for radius in TARGET_RING_RADII:
            axis.add_patch(plt.Circle((0.0, 0.0), radius, fill=False, color=GRAY, alpha=0.5, linewidth=1.2))
        weights = panel.sample_weights(rng, TARGET_FITS_PER_PANEL)
        axis.scatter(*weights.T, s=34, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
        axis.scatter([0.0], [0.0], marker="*", s=230, color=ORANGE, edgecolor="white", zorder=4, label=r"best weights $\mathbf{w}^*$")
        axis.set_title(title, color=NAVY, fontsize=14, weight="bold")
        axis.set(xlim=(-TARGET_LIMIT, TARGET_LIMIT), ylim=(-TARGET_LIMIT, TARGET_LIMIT))
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in axes[1]:
        axis.set_xlabel(r"fitted weight $w_1$", fontsize=13)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"fitted weight $w_2$", fontsize=13)
    axes[0, 0].legend(frameon=False, fontsize=10, loc="lower left")
    figure.text(0.5, 0.005, "schematic: each dot is the fit from one training set", ha="center", color=GRAY, fontsize=11, style="italic")
    figure.subplots_adjust(wspace=0.06, hspace=0.18)
    _save(figure, path)


def _learning_curves(path: Path) -> None:
    capacity = np.linspace(0.0, 1.0, 180)
    training = 0.58 * np.exp(-3.2 * capacity) + 0.035
    validation = training + 0.025 + 0.72 * (capacity - 0.55) ** 2
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    axis.plot(capacity, training, color=BLUE, linewidth=3, label="training error")
    axis.plot(capacity, validation, color=ORANGE, linewidth=3, label="validation error")
    optimum = int(np.argmin(validation))
    axis.axvline(capacity[optimum], color=GREEN, linestyle="--", linewidth=2)
    axis.text(capacity[optimum] + 0.02, validation[optimum] + 0.04, "selected capacity", color=GREEN)
    axis.text(0.08, 0.68, "underfit", color=GRAY, fontsize=14)
    axis.text(0.79, 0.68, "overfit", color=GRAY, fontsize=14)
    axis.set(xlabel="model capacity", ylabel="prediction error", xlim=(0, 1), ylim=(0, 0.85))
    axis.set_xticks([])
    axis.legend(frameon=False, fontsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
    _save(figure, path)


def _bias_variance(path: Path, rng: np.random.Generator) -> None:
    grid = np.linspace(-1.0, 1.0, 240)
    truth = np.sin(np.pi * grid)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    for axis, degree, title in zip(axes, (1, 9), ("high bias", "high variance"), strict=True):
        predictions = []
        for _ in range(12):
            x = np.sort(rng.uniform(-1.0, 1.0, 18))
            y = np.sin(np.pi * x) + rng.normal(0.0, 0.22, size=x.size)
            coefficients = np.polyfit(x, y, degree)
            prediction = np.polyval(coefficients, grid)
            predictions.append(prediction)
            axis.plot(grid, prediction, color=BLUE, alpha=0.22, linewidth=1.2)
        axis.plot(grid, truth, color=NAVY, linewidth=3, label="true function")
        axis.plot(grid, np.mean(predictions, axis=0), color=ORANGE, linewidth=3, label="mean fit")
        axis.set_title(title, color=NAVY, fontsize=16, weight="bold")
        axis.set_xlabel("input $x$")
        axis.set_ylim(-2.1, 2.1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.12)
    axes[0].set_ylabel("prediction")
    axes[1].legend(frameon=False, fontsize=11, loc="upper right")
    _save(figure, path)


@dataclass(frozen=True, slots=True)
class L2Landscape:
    """Data loss, penalty and their sum on one grid, for 0.5*||w-w0||^2 + 0.5*lam*||w||^2."""

    horizontal: NDArray[np.float64]
    vertical: NDArray[np.float64]
    data_loss: NDArray[np.float64]
    penalty: NDArray[np.float64]
    total: NDArray[np.float64]

    def minimizer(self) -> NDArray[np.float64]:
        """Grid point where the total loss is smallest, found numerically."""
        row, column = np.unravel_index(np.argmin(self.total), self.total.shape)
        return np.array([self.horizontal[row, column], self.vertical[row, column]])


def l2_landscape(unregularized: NDArray[np.float64], strength: float) -> L2Landscape:
    horizontal, vertical = np.meshgrid(np.linspace(*L2_XLIM, 360), np.linspace(*L2_YLIM, 340))
    data_loss = 0.5 * ((horizontal - unregularized[0]) ** 2 + (vertical - unregularized[1]) ** 2)
    penalty = 0.5 * strength * (horizontal**2 + vertical**2)
    return L2Landscape(horizontal, vertical, data_loss, penalty, data_loss + penalty)


def _l2_contours(path: Path) -> None:
    unregularized, strength = L2_UNREGULARIZED, L2_STRENGTH
    landscape = l2_landscape(unregularized, strength)
    regularized = l2_solution(unregularized, strength)
    assert np.allclose(landscape.minimizer(), regularized, atol=0.02)
    data_at_point = 0.5 * float(np.sum((regularized - unregularized) ** 2))
    penalty_at_point = 0.5 * strength * float(np.sum(regularized**2))
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.contour(landscape.horizontal, landscape.vertical, landscape.data_loss,
                 levels=(0.15, 0.5, 1.2, 2.4), colors=BLUE, linewidths=1.6)
    axis.contour(landscape.horizontal, landscape.vertical, landscape.penalty,
                 levels=(0.25, 1.0, 2.5, 5.0), colors=PURPLE, linestyles="dashed")
    total_levels = data_at_point + penalty_at_point + np.array(L2_TOTAL_LEVEL_OFFSETS)
    axis.contour(landscape.horizontal, landscape.vertical, landscape.total,
                 levels=total_levels, colors=GRAY, linewidths=1.2)
    axis.scatter(*unregularized, marker="*", s=260, color=BLUE, edgecolor="white", zorder=5)
    axis.scatter(*regularized, color=GREEN, s=95, edgecolor="white", zorder=5)
    axis.arrow(*unregularized, *(regularized - unregularized), color=ORANGE, width=0.015,
               head_width=0.13, length_includes_head=True, zorder=4)
    axis.set(xlabel=r"weight $w_1$", ylabel=r"weight $w_2$", xlim=L2_XLIM, ylim=L2_YLIM)
    axis.set_aspect("equal")
    handles = [
        Line2D([], [], color=BLUE, linewidth=1.6, label="data loss"),
        Line2D([], [], color=PURPLE, linestyle="dashed", label=rf"penalty $\lambda\|\mathbf{{w}}\|^2/2$, $\lambda={strength:g}$"),
        Line2D([], [], color=GRAY, label="sum of the two"),
        Line2D([], [], marker="*", color="none", markerfacecolor=BLUE, markersize=15, label="best fit without penalty"),
        Line2D([], [], marker="o", color="none", markerfacecolor=GREEN, markersize=9,
               label=f"minimum of the sum: {data_at_point:.2f}+{penalty_at_point:.2f}={data_at_point + penalty_at_point:.2f}"),
    ]
    axis.legend(handles=handles, frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    axis.spines[["top", "right"]].set_visible(False)
    _save(figure, path)


@dataclass(frozen=True, slots=True)
class OneWeightCurves:
    """The same compromise with a single weight, so the sum can be drawn as a curve."""

    weight: NDArray[np.float64]
    data_loss: NDArray[np.float64]
    penalty: NDArray[np.float64]
    total: NDArray[np.float64]


def one_weight_curves(unregularized: float, strength: float) -> OneWeightCurves:
    weight = np.linspace(-0.6, 2.6, 400)
    data_loss = 0.5 * (weight - unregularized) ** 2
    penalty = 0.5 * strength * weight**2
    return OneWeightCurves(weight, data_loss, penalty, data_loss + penalty)


def _l2_one_weight(path: Path) -> None:
    curves = one_weight_curves(ONE_WEIGHT_UNREGULARIZED, L2_STRENGTH)
    best = float(curves.weight[np.argmin(curves.total)])
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.plot(curves.weight, curves.data_loss, color=BLUE, linewidth=3, label="data misfit")
    axis.plot(curves.weight, curves.penalty, color=PURPLE, linewidth=3, linestyle="--", label=r"penalty $\lambda w^2/2$")
    axis.plot(curves.weight, curves.total, color=GRAY, linewidth=3, label="sum")
    axis.scatter([ONE_WEIGHT_UNREGULARIZED], [0.0], marker="*", s=240, color=BLUE, zorder=5)
    axis.scatter([best], [float(np.min(curves.total))], s=90, color=GREEN, zorder=5)
    axis.annotate("best without penalty", (ONE_WEIGHT_UNREGULARIZED, 0.0), (1.55, 0.9), color=BLUE, arrowprops={"arrowstyle": "->", "color": BLUE, "shrinkB": 10})
    axis.annotate("best compromise", (best, float(np.min(curves.total))), (0.1, 3.0), color=GREEN, arrowprops={"arrowstyle": "->", "color": GREEN, "shrinkB": 8})
    axis.set(xlabel=r"weight $w$", ylabel="loss", ylim=(0, 4.5))
    axis.legend(frameon=False, fontsize=12, loc="upper center")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
    _save(figure, path)


def _weight_priors(path: Path) -> None:
    weights = np.linspace(-3.5, 3.5, 400)
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    for scale, color in ((0.5, PURPLE), (1.2, BLUE)):
        density = np.exp(-0.5 * (weights / scale) ** 2) / (np.sqrt(2 * np.pi) * scale)
        axis.plot(weights, density, color=color, linewidth=3, label=rf"$\tau={scale}$")
    axis.axvline(0, color=GRAY, linewidth=1.4)
    axis.set(xlabel=r"weight $w_j$", ylabel="prior density", ylim=(0, 0.85))
    axis.legend(frameon=False, fontsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
    _save(figure, path)


def geometry_loss(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x - GEOMETRY_CENTRE[0]) ** 2 + GEOMETRY_CURVATURE_Y * (y - GEOMETRY_CENTRE[1]) ** 2


def l2_ball_boundary(points: int = 40001) -> NDArray[np.float64]:
    angle = np.linspace(0.0, 2.0 * np.pi, points)
    return BALL_RADIUS * np.column_stack((np.cos(angle), np.sin(angle)))


def l1_ball_boundary(points_per_edge: int = 10001) -> NDArray[np.float64]:
    """Diamond |x|+|y|=r sampled densely on each edge, corners included exactly."""
    t = np.linspace(0.0, BALL_RADIUS, points_per_edge)
    edges = (
        np.column_stack((t, BALL_RADIUS - t)),
        np.column_stack((-t, BALL_RADIUS - t)),
        np.column_stack((t, t - BALL_RADIUS)),
        np.column_stack((-t, t - BALL_RADIUS)),
    )
    return np.vstack(edges)


def constrained_minimizer(boundary: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lowest-loss point on the shape's boundary; the centre lies outside the shape, so the optimum is on it."""
    return boundary[np.argmin(geometry_loss(boundary[:, 0], boundary[:, 1]))]


def _l1_l2_geometry(path: Path) -> None:
    angle = np.linspace(0, 2 * np.pi, 400)
    x, y = np.meshgrid(np.linspace(-1.8, 2.3, 300), np.linspace(-1.5, 1.5, 300))
    loss = geometry_loss(x, y)
    shapes = (
        (r"L2: $w_1^2+w_2^2\leq 1$", l2_ball_boundary()),
        (r"L1: $|w_1|+|w_2|\leq 1$", l1_ball_boundary()),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    for axis, (title, boundary) in zip(axes, shapes, strict=True):
        best = constrained_minimizer(boundary)
        touching = float(geometry_loss(*best))
        axis.contour(x, y, loss, levels=(0.3 * touching, touching, 2.0 * touching, 3.5 * touching), colors=BLUE, linewidths=(1.0, 2.4, 1.0, 1.0))
        axis.axhline(0, color=GRAY, linewidth=0.7)
        axis.axvline(0, color=GRAY, linewidth=0.7)
        axis.scatter(*GEOMETRY_CENTRE, marker="*", s=260, color=BLUE, edgecolor="white", zorder=6, label="best fit without penalty")
        axis.scatter(*best, color=GREEN, s=110, edgecolor="white", zorder=6, label="best allowed weights")
        axis.set_title(title, color=NAVY, fontsize=15, weight="bold")
        axis.set_xlabel(r"weight $w_1$")
        axis.set_aspect("equal")
        axis.set_xlim(-1.4, 2.3)
        axis.set_ylim(-1.4, 1.4)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"weight $w_2$")
    axes[0].plot(np.cos(angle), np.sin(angle), color=PURPLE, linewidth=3)
    axes[1].plot((0, 1, 0, -1, 0), (1, 0, -1, 0, 1), color=PURPLE, linewidth=3)
    axes[1].legend(frameon=False, fontsize=10, loc="lower left")
    _save(figure, path)


def _regularization_path(path: Path) -> None:
    strengths = np.logspace(-3, 2, 220)
    coefficients = np.array([2.2, -1.4, 0.65])[:, None] / (1.0 + strengths[None, :])
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    for values, color, label in zip(coefficients, (BLUE, ORANGE, GREEN), (r"$w_1$", r"$w_2$", r"$w_3$"), strict=True):
        axis.semilogx(strengths, values, color=color, linewidth=3, label=label)
    axis.axhline(0, color=GRAY, linewidth=1)
    axis.set(xlabel=r"regularization strength $\lambda$", ylabel="ridge coefficient")
    axis.legend(frameon=False, fontsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
    _save(figure, path)


def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_directory / name for name in EXPECTED_FILENAMES)
    rng = np.random.default_rng(SEED)
    _fit_regimes(outputs[0], rng)
    _learning_curves(outputs[1])
    _bias_variance_targets(outputs[2], rng)
    _bias_variance(outputs[3], rng)
    _l2_one_weight(outputs[4])
    _l2_contours(outputs[5])
    _weight_priors(outputs[6])
    _l1_l2_geometry(outputs[7])
    _regularization_path(outputs[8])
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    return outputs


def main() -> None:
    output_directory = Path(__file__).resolve().parents[1] / "assets" / "generated"
    generate_figures(output_directory)


if __name__ == "__main__":
    main()
