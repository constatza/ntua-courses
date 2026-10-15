#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "generated"
BLUE = "#5B8DB8"
TEAL = "#63A6A0"
GOLD = "#D7A85E"
CORAL = "#C77C6B"
PURPLE = "#8B82A8"
INK = "#263746"


def two_layer_loss(
    x: np.ndarray,
    target: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> float:
    hidden = np.tanh(w1 @ x + b1)
    error = w2 @ hidden + b2 - target
    return float(0.5 * error @ error)


def two_layer_gradients(
    x: np.ndarray,
    target: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Differentiate a tanh hidden layer and linear MSE output."""
    z1 = w1 @ x + b1
    hidden = np.tanh(z1)
    prediction = w2 @ hidden + b2
    delta2 = prediction - target
    delta1 = (w2.T @ delta2) * (1.0 - hidden**2)
    return {
        "w1": np.outer(delta1, x),
        "b1": delta1,
        "w2": np.outer(delta2, hidden),
        "b2": delta2,
    }


def save_learning_curves() -> None:
    epochs = np.arange(1, 121)
    train = 0.72 * np.exp(-epochs / 34.0) + 0.045
    validation = 0.62 * np.exp(-epochs / 25.0) + 0.09 + 0.000018 * np.maximum(epochs - 58, 0) ** 2
    best = int(np.argmin(validation))
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    ax.plot(epochs, train, color=BLUE, lw=2.8, label="training loss")
    ax.plot(epochs, validation, color=CORAL, lw=2.8, label="validation loss")
    ax.axvline(epochs[best], color=TEAL, ls="--", lw=2.0, alpha=0.85, label="restore best checkpoint")
    ax.fill_between(epochs[best:], validation[best:], validation[-1], color=CORAL, alpha=0.09)
    ax.set(xlabel="epoch", ylabel="loss", ylim=(0.0, 0.78))
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=180, transparent=True)
    plt.close(fig)


def save_capacity_tradeoff() -> None:
    rng = np.random.default_rng(12)
    x = np.linspace(-1.0, 1.0, 28)
    truth = 0.3 + 0.8 * x - 0.9 * x**2 + 0.55 * x**3
    y = truth + rng.normal(0.0, 0.12, x.size)
    dense = np.linspace(-1.05, 1.05, 500)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.25), sharey=True, constrained_layout=True)
    for ax, degree, title, color in zip(
        axes,
        (1, 3, 14),
        ("underfit", "useful capacity", "overfit"),
        (GOLD, TEAL, CORAL),
        strict=True,
    ):
        coefficients = np.polyfit(x, y, degree)
        ax.scatter(x, y, color=BLUE, alpha=0.48, s=22, edgecolor="white", linewidth=0.4)
        ax.plot(dense, np.polyval(coefficients, dense), color=color, lw=2.7)
        ax.set_title(title)
        ax.set_xlabel("input")
        ax.grid(True, alpha=0.15)
    axes[0].set_ylabel("target")
    for ax in axes:
        ax.set_ylim(-2.0, 1.1)
    fig.savefig(OUTPUT_DIR / "capacity_tradeoff.png", dpi=180, transparent=True)
    plt.close(fig)


def save_optimizer_paths() -> None:
    x = np.linspace(-2.2, 2.2, 160)
    y = np.linspace(-1.6, 1.6, 160)
    xx, yy = np.meshgrid(x, y)
    loss = 0.12 * xx**2 + 1.7 * yy**2
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.contour(xx, yy, loss, levels=12, cmap="viridis", alpha=0.58)
    gd = np.array([[-2.0, 1.4], [-1.76, -0.98], [-1.55, 0.69], [-1.36, -0.48], [-1.2, 0.34], [-1.05, -0.24], [-0.92, 0.17]])
    momentum = np.array([[-2.0, 1.4], [-1.76, 0.72], [-1.38, 0.05], [-0.96, -0.28], [-0.57, -0.26], [-0.25, -0.08], [-0.06, 0.02]])
    ax.plot(gd[:, 0], gd[:, 1], "o-", color=CORAL, lw=2.0, ms=4.5, label="plain gradient descent")
    ax.plot(momentum[:, 0], momentum[:, 1], "o-", color=PURPLE, lw=2.0, ms=4.5, label="with momentum")
    ax.scatter([0], [0], marker="*", s=130, color=TEAL, label="minimum", zorder=5)
    ax.set(xlabel=r"parameter $\theta_1$", ylabel=r"parameter $\theta_2$")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUTPUT_DIR / "optimizer_paths.png", dpi=180, transparent=True)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_learning_curves()
    save_capacity_tradeoff()
    save_optimizer_paths()


if __name__ == "__main__":
    main()
