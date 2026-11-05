from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray
from supervised_math import (
    GaussianScores,
    augment,
    fit_logistic,
    normal_cdf,
    sigmoid,
    trapezoid,
)

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
    "task-types.png",
    "polynomial-fit.png",
    "linear-regression.png",
    "logistic-regression.png",
    "threshold.png",
    "roc.png",
    "pr.png",
    "normalization.png",
    "linear-vs-logistic.png",
)

POLYNOMIAL_DEGREES = (1, 3, 9)
ERROR_CURVE_DEGREES = tuple(range(1, 10))
POLYNOMIAL_YLIM = 2.0
POLYNOMIAL_TRAIN_SIZE = 12
POLYNOMIAL_NOISE = 0.25
THRESHOLD = 1.0
SCORE_SEPARATION = 2.0
PREVALENCE = 0.1


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


def two_class_sample(rng: np.random.Generator, per_class: int = 42) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    negative = rng.multivariate_normal((-1.4, -0.7), ((0.65, 0.18), (0.18, 0.55)), size=per_class)
    positive = rng.multivariate_normal((1.2, 0.9), ((0.7, 0.12), (0.12, 0.6)), size=per_class)
    features = np.vstack((negative, positive))
    labels = np.concatenate((np.zeros(per_class), np.ones(per_class)))
    return features, labels


def _logistic_regression_figure(rng: np.random.Generator, path: Path) -> None:
    features, labels = two_class_sample(rng)
    beta = fit_logistic(augment(features), labels)  # fitted, not hand-set
    grid_x, grid_y = np.meshgrid(np.linspace(-4, 4, 260), np.linspace(-3.5, 3.5, 240))
    probabilities = sigmoid(beta[0] * grid_x + beta[1] * grid_y + beta[2])
    color_map = LinearSegmentedColormap.from_list("class_probability", (LIGHT_BLUE, "#ffffff", LIGHT_ORANGE))

    figure, axis = plt.subplots(figsize=(9.6, 5.4))
    background = axis.contourf(grid_x, grid_y, probabilities, levels=np.linspace(0.0, 1.0, 21), cmap=color_map)
    axis.contour(grid_x, grid_y, probabilities, levels=(0.5,), colors=(NAVY,), linewidths=2.6)
    negative, positive = features[labels == 0], features[labels == 1]
    axis.scatter(negative[:, 0], negative[:, 1], s=54, color=BLUE, edgecolor="white", linewidth=0.7, label=r"$y=0$")
    axis.scatter(positive[:, 0], positive[:, 1], s=54, color=ORANGE, edgecolor="white", linewidth=0.7, label=r"$y=1$")
    axis.plot([], [], color=NAVY, linewidth=2.6, label=r"decision boundary $\hat p=0.5$")
    color_bar = figure.colorbar(background, ax=axis, pad=0.02)
    color_bar.set_label(r"predicted probability $\hat p(y=1\mid\mathbf{x})$", fontsize=13)
    color_bar.ax.tick_params(labelsize=11)
    axis.set_xlabel(r"feature $x_1$", fontsize=16)
    axis.set_ylabel(r"feature $x_2$", fontsize=16)
    axis.legend(frameon=False, fontsize=13, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=12)
    _save(figure, path)


def true_curve(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(np.pi * x)


@dataclass(frozen=True, slots=True)
class PolynomialSample:
    x_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.float64]

    def rmse(self, degree: int) -> tuple[float, float]:
        """(train RMSE, test RMSE) of the degree-``degree`` least-squares polynomial."""
        coefficients = np.polyfit(self.x_train, self.y_train, degree)
        train = np.sqrt(np.mean((np.polyval(coefficients, self.x_train) - self.y_train) ** 2))
        test = np.sqrt(np.mean((np.polyval(coefficients, self.x_test) - self.y_test) ** 2))
        return float(train), float(test)

    def curve_extreme(self, degree: int) -> tuple[float, float]:
        """(x, value) where the fitted curve is farthest from zero on [-1, 1]."""
        coefficients = np.polyfit(self.x_train, self.y_train, degree)
        values = np.polyval(coefficients, self.x_test)
        index = int(np.argmax(np.abs(values)))
        return float(self.x_test[index]), float(values[index])


# The figure and the tests both draw this sample from default_rng(SEED), so the numbers quoted on
# the slides are the numbers the tests check.
def polynomial_sample(rng: np.random.Generator) -> PolynomialSample:
    x_train = np.sort(rng.uniform(-1.0, 1.0, POLYNOMIAL_TRAIN_SIZE))
    x_test = np.linspace(-1.0, 1.0, 400)
    return PolynomialSample(
        x_train,
        true_curve(x_train) + rng.normal(0.0, POLYNOMIAL_NOISE, x_train.size),
        x_test,
        true_curve(x_test) + rng.normal(0.0, POLYNOMIAL_NOISE, x_test.size),
    )


def _polynomial_figure(path: Path) -> None:
    sample = polynomial_sample(np.random.default_rng(SEED))
    grid = np.linspace(-1.0, 1.0, 400)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))
    fit_axes = (axes[0, 0], axes[0, 1], axes[1, 0])
    for axis, degree in zip(fit_axes, POLYNOMIAL_DEGREES, strict=True):
        coefficients = np.polyfit(sample.x_train, sample.y_train, degree)
        train, test = sample.rmse(degree)
        axis.plot(grid, true_curve(grid), color=GRAY, linestyle="--", linewidth=1.8, label="true function")
        axis.plot(grid, np.polyval(coefficients, grid), color=ORANGE, linewidth=2.8, label=f"degree-{degree} fit")
        axis.scatter(
            sample.x_train, sample.y_train, s=46, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4,
            label="training points",
        )
        axis.set_ylim(-POLYNOMIAL_YLIM, POLYNOMIAL_YLIM)
        axis.set_title(f"degree {degree}: train RMSE {train:.2f}, test RMSE {test:.2f}", color=NAVY, fontsize=13, weight="bold")
        axis.set_ylabel(r"target $y$", fontsize=12)
        axis.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=9, loc="lower right")
        axis.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        extreme_x, extreme_value = sample.curve_extreme(degree)
        if abs(extreme_value) > POLYNOMIAL_YLIM:
            axis.text(
                -0.45, 1.5, f"off scale: reaches {extreme_value:.0f}\nat x = {extreme_x:.1f}",
                color=ORANGE, fontsize=11, weight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.0},
            )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel(r"feature $x$", fontsize=12)

    errors = np.array([sample.rmse(degree) for degree in ERROR_CURVE_DEGREES])
    error_axis = axes[1, 1]
    error_axis.semilogy(ERROR_CURVE_DEGREES, errors[:, 0], color=BLUE, marker="o", linewidth=2.4, label="training points")
    error_axis.semilogy(ERROR_CURVE_DEGREES, errors[:, 1], color=ORANGE, marker="o", linewidth=2.4, label="fresh points")
    error_axis.axhline(POLYNOMIAL_NOISE, color=GRAY, linestyle="--", linewidth=1.5, label=f"noise level {POLYNOMIAL_NOISE}")
    error_axis.set_xlabel("polynomial degree", fontsize=12)
    error_axis.set_ylabel("RMSE (log scale)", fontsize=12)
    error_axis.set_title("error versus degree", color=NAVY, fontsize=13, weight="bold")
    error_axis.legend(frameon=False, fontsize=10, loc="upper left")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
    figure.subplots_adjust(hspace=0.42, wspace=0.22)
    _save(figure, path)


def _task_types_figure(rng: np.random.Generator, path: Path) -> None:
    figure, (regression, classification) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    x = np.linspace(0.5, 9.5, 28)
    y = 1.4 * x + 2.0 + rng.normal(0.0, 1.5, size=x.size)
    slope, intercept = np.polyfit(x, y, deg=1)
    regression.scatter(x, y, s=46, color=BLUE, edgecolor="white", linewidth=0.7)
    regression.plot(x, slope * x + intercept, color=ORANGE, linewidth=3.0)
    regression.set_title(r"regression: $y\in\mathbb{R}$", color=NAVY, fontsize=15, weight="bold")
    regression.set_xlabel(r"feature $x$", fontsize=13)
    regression.set_ylabel(r"target $y$", fontsize=13)

    features, labels = two_class_sample(rng, per_class=30)
    beta = fit_logistic(augment(features), labels)
    for value, color in ((0.0, BLUE), (1.0, ORANGE)):
        group = features[labels == value]
        classification.scatter(group[:, 0], group[:, 1], s=46, color=color, edgecolor="white", linewidth=0.7)
    line_x = np.linspace(-3.5, 3.5, 50)
    classification.plot(line_x, -(beta[0] * line_x + beta[2]) / beta[1], color=NAVY, linewidth=2.8)
    classification.set_ylim(-3.2, 3.5)
    classification.set_title(r"classification: $y\in\{0,1\}$", color=NAVY, fontsize=15, weight="bold")
    classification.set_xlabel(r"feature $x_1$", fontsize=13)
    classification.set_ylabel(r"feature $x_2$", fontsize=13)
    for axis in (regression, classification):
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=11)
    figure.subplots_adjust(wspace=0.28)
    _save(figure, path)


def _threshold_figure(path: Path) -> None:
    scores = np.linspace(-3.5, 5.5, 600)
    negative = np.exp(-0.5 * scores**2) / np.sqrt(2 * np.pi)
    positive = np.exp(-0.5 * (scores - SCORE_SEPARATION) ** 2) / np.sqrt(2 * np.pi)
    false_positive = scores >= THRESHOLD
    false_negative = scores < THRESHOLD
    figure, axis = plt.subplots(figsize=(9.6, 4.6))
    axis.plot(scores, negative, color=BLUE, linewidth=2.6, label=r"$y=0$")
    axis.plot(scores, positive, color=ORANGE, linewidth=2.6, label=r"$y=1$")
    axis.fill_between(scores, negative, where=false_positive, color=BLUE, alpha=0.3)
    axis.fill_between(scores, positive, where=false_negative, color=ORANGE, alpha=0.3)
    axis.axvline(THRESHOLD, color=NAVY, linewidth=2.2)
    fpr = float(1.0 - normal_cdf(THRESHOLD))
    fnr = float(normal_cdf(THRESHOLD - SCORE_SEPARATION))
    axis.text(THRESHOLD, 0.415, r"threshold $\tau$", color=NAVY, fontsize=12, ha="center")
    axis.text(3.0, 0.27, rf"false positives:" "\n" rf"{fpr:.0%} of $y=0$", color=BLUE, fontsize=12)
    axis.text(-3.4, 0.32, rf"false negatives:" "\n" rf"{fnr:.0%} of $y=1$", color=ORANGE, fontsize=12)
    axis.set_xlabel(r"model score $z=\mathbf{w}^\top\mathbf{x}+b$", fontsize=14)
    axis.set_ylabel("density", fontsize=14)
    axis.set_ylim(top=0.45)
    axis.legend(frameon=False, fontsize=13, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    _save(figure, path)


def _roc_figure(path: Path) -> None:
    model = GaussianScores(SCORE_SEPARATION, PREVALENCE)
    thresholds = np.linspace(8.0, -6.0, 1200)
    tpr = model.true_positive_rate(thresholds)
    fpr = model.false_positive_rate(thresholds)
    marker = np.array([THRESHOLD])
    figure, axis = plt.subplots(figsize=(6.0, 4.8))
    axis.plot(fpr, tpr, color=ORANGE, linewidth=2.8, label=f"model, AUC = {trapezoid(tpr, fpr):.2f}")
    axis.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.5, label="random scores, AUC = 0.50")
    axis.scatter(model.false_positive_rate(marker), model.true_positive_rate(marker), s=80, color=NAVY, zorder=5, label=r"threshold $\tau=1$")
    axis.set_xlabel("false-positive rate", fontsize=13)
    axis.set_ylabel("true-positive rate (recall)", fontsize=13)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, fontsize=11, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    axis.grid(alpha=0.15)
    _save(figure, path)


def _pr_figure(path: Path) -> None:
    model = GaussianScores(SCORE_SEPARATION, PREVALENCE)
    thresholds = np.linspace(8.0, -6.0, 1200)
    marker = np.array([THRESHOLD])
    figure, axis = plt.subplots(figsize=(6.0, 4.8))
    axis.plot(model.true_positive_rate(thresholds), model.precision(thresholds), color=ORANGE, linewidth=2.8, label="model")
    axis.axhline(PREVALENCE, color=GRAY, linestyle="--", linewidth=1.5, label=f"random scores ({PREVALENCE:.0%} positives)")
    axis.scatter(model.true_positive_rate(marker), model.precision(marker), s=80, color=NAVY, zorder=5, label=r"threshold $\tau=1$")
    axis.set_xlabel("recall", fontsize=13)
    axis.set_ylabel("precision", fontsize=13)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, fontsize=11, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    axis.grid(alpha=0.15)
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


LABEL_FIT_SAMPLES = 40
LABEL_FIT_SLOPE = 1.6


def label_fit_sample(rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One feature, 0/1 labels drawn from a logistic law, for the 'why not a line' figure."""
    x = np.sort(rng.uniform(-3.0, 3.0, LABEL_FIT_SAMPLES))
    labels = (rng.uniform(size=x.size) < sigmoid(LABEL_FIT_SLOPE * x)).astype(np.float64)
    return x, labels


def _linear_vs_logistic_figure(rng: np.random.Generator, path: Path) -> None:
    x, labels = label_fit_sample(rng)
    grid = np.linspace(-3.4, 3.4, 300)
    slope, intercept = np.polyfit(x, labels, deg=1)
    beta = fit_logistic(augment(x[:, None]), labels)
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    axis.axhspan(0.0, 1.0, color=LIGHT_BLUE, alpha=0.45, zorder=0)
    axis.scatter(x, labels, s=54, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4, label="labels (0 or 1)")
    axis.plot(grid, slope * grid + intercept, color=GRAY, linewidth=2.6, linestyle="--", label="straight-line fit")
    axis.plot(grid, sigmoid(beta[0] * grid + beta[1]), color=ORANGE, linewidth=3.0, label="logistic fit")
    axis.set_ylim(-0.6, 1.6)
    axis.set_xlabel(r"feature $x$", fontsize=14)
    axis.set_ylabel("label / predicted value", fontsize=14)
    axis.legend(frameon=False, fontsize=12, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    _save(figure, path)


def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    outputs = tuple(output_directory / filename for filename in EXPECTED_FILENAMES)

    _task_types_figure(rng, outputs[0])
    _polynomial_figure(outputs[1])
    _linear_regression_figure(rng, outputs[2])
    _logistic_regression_figure(rng, outputs[3])
    _threshold_figure(outputs[4])
    _roc_figure(outputs[5])
    _pr_figure(outputs[6])
    _normalization_figure(rng, outputs[7])
    _linear_vs_logistic_figure(rng, outputs[8])

    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    return outputs


def main() -> None:
    """Write the supervised-learning figures under assets/generated/."""
    output_directory = Path(__file__).resolve().parents[1] / "assets" / "generated"
    generate_figures(output_directory)


if __name__ == "__main__":
    main()
