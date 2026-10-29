from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance, ImageFilter

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
    "learning-curves.png",
    "bias-variance.png",
    "l2-contours.png",
    "weight-priors.png",
    "l1-l2-geometry.png",
    "regularization-path.png",
    "augmentation-grid.png",
)


def l2_solution(unregularized: NDArray[np.float64], strength: float) -> NDArray[np.float64]:
    return unregularized / (1.0 + strength)


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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


def _l2_contours(path: Path) -> None:
    unregularized = np.array([2.0, 1.0])
    strength = 2.0
    regularized = l2_solution(unregularized, strength)
    horizontal, vertical = np.meshgrid(np.linspace(-1.2, 3.2, 360), np.linspace(-1.7, 2.7, 340))
    data_loss = 0.5 * ((horizontal - unregularized[0]) ** 2 + (vertical - unregularized[1]) ** 2)
    penalty = 0.5 * strength * (horizontal**2 + vertical**2)
    total = data_loss + penalty
    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    axis.contour(horizontal, vertical, data_loss, levels=(0.15, 0.5, 1.2, 2.4), colors=BLUE, linewidths=1.6)
    axis.contour(horizontal, vertical, penalty, levels=(0.25, 1.0, 2.5, 5.0), colors=PURPLE, linestyles="dashed")
    axis.contour(horizontal, vertical, total, levels=(1.7, 2.2, 3.2, 5.0), colors=GRAY, linewidths=1.0)
    axis.scatter(*unregularized, color=BLUE, s=95, edgecolor="white", zorder=5, label="data-loss minimum")
    axis.scatter(*regularized, color=GREEN, s=95, edgecolor="white", zorder=5, label="regularized minimum")
    axis.arrow(*unregularized, *(regularized - unregularized), color=ORANGE, width=0.015,
               head_width=0.13, length_includes_head=True, zorder=4)
    axis.set(xlabel=r"weight $w_1$", ylabel=r"weight $w_2$", xlim=(-1.2, 3.2), ylim=(-1.7, 2.7))
    axis.set_aspect("equal")
    axis.legend(frameon=False, fontsize=11, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(-1.0, 2.45, "blue: data loss   purple dashed: L2 penalty   gray: sum", color=NAVY, fontsize=11)
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


def _l1_l2_geometry(path: Path) -> None:
    angle = np.linspace(0, 2 * np.pi, 400)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    for axis, title in zip(
        axes,
        (r"L2: $\|\mathbf{w}\|_2\leq c$", r"L1: $\|\mathbf{w}\|_1\leq c$"),
        strict=True,
    ):
        x, y = np.meshgrid(np.linspace(-2, 2, 220), np.linspace(-2, 2, 220))
        loss = (x - 1.45) ** 2 + 2.2 * (y - 0.9) ** 2
        axis.contour(x, y, loss, levels=(0.2, 0.6, 1.2, 2.2), colors=BLUE)
        axis.axhline(0, color=GRAY, linewidth=0.7)
        axis.axvline(0, color=GRAY, linewidth=0.7)
        axis.set_title(title, color=NAVY, fontsize=15, weight="bold")
        axis.set_aspect("equal")
        axis.set_xlim(-1.8, 1.8)
        axis.set_ylim(-1.8, 1.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].plot(np.cos(angle), np.sin(angle), color=PURPLE, linewidth=3)
    axes[1].plot((0, 1, 0, -1, 0), (1, 0, -1, 0, 1), color=PURPLE, linewidth=3)
    axes[1].scatter(1, 0, color=GREEN, s=75, zorder=5)
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


def _augmentation_grid(path: Path, rng: np.random.Generator) -> None:
    source = Path(__file__).resolve().parents[1] / "assets" / "source" / "cat.png"
    image = Image.open(source).convert("RGB").resize((420, 420))
    crop = image.crop((45, 20, 365, 340)).resize(image.size)
    noisy = np.asarray(image, dtype=np.float32) + rng.normal(0, 13, (420, 420, 3))
    variants = (
        ("original", image),
        (r"rotate $15^\circ$", image.rotate(15, resample=Image.Resampling.BICUBIC, fillcolor=(225, 222, 219))),
        ("horizontal flip", image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        ("crop + resize", crop),
        ("color shift", ImageEnhance.Color(image).enhance(0.45)),
        ("sensor noise", Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))),
        ("mild blur", image.filter(ImageFilter.GaussianBlur(radius=3.0))),
    )
    figure, axes = plt.subplots(2, 4, figsize=(11.2, 6.0))
    for axis, (title, variant) in zip(axes.flat, variants, strict=False):
        axis.imshow(variant)
        axis.set_title(title, color=NAVY, fontsize=12, weight="bold")
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.subplots_adjust(wspace=0.04, hspace=0.16)
    _save(figure, path)


def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_directory / name for name in EXPECTED_FILENAMES)
    rng = np.random.default_rng(SEED)
    _learning_curves(outputs[0])
    _bias_variance(outputs[1], rng)
    _l2_contours(outputs[2])
    _weight_priors(outputs[3])
    _l1_l2_geometry(outputs[4])
    _regularization_path(outputs[5])
    _augmentation_grid(outputs[6], rng)
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    return outputs


def main() -> None:
    output_directory = Path(__file__).resolve().parents[1] / "assets" / "generated"
    generate_figures(output_directory)


if __name__ == "__main__":
    main()
