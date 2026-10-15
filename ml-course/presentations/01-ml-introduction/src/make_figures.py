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
INK = "#263746"


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the element-wise logistic function without overflow."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9AA7B2",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "font.size": 11,
            "grid.alpha": 0.18,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def save_regression_bridge() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(-2.5, 2.5, 36)
    target = 0.55 * x + 0.7 * np.sin(1.8 * x)
    y = target + rng.normal(0.0, 0.18, x.size)
    linear = np.polyval(np.polyfit(x, y, deg=1), x)

    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=38, color=BLUE, alpha=0.62, edgecolor="white", linewidth=0.6, label="data")
    ax.plot(x, linear, color=CORAL, lw=2.6, alpha=0.9, label="one affine neuron")
    ax.plot(x, target, color=TEAL, lw=3.0, alpha=0.92, label="nonlinear target")
    ax.set(xlabel=r"input $x$", ylabel=r"target $y$", title="Why add nonlinear hidden units?")
    ax.grid(True)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    fig.savefig(OUTPUT_DIR / "regression_bridge.png", dpi=180, transparent=True)
    plt.close(fig)


def save_activation_functions() -> None:
    x = np.linspace(-4.0, 4.0, 500)
    fig, ax = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    ax.plot(x, sigmoid(x), color=BLUE, lw=2.8, label="sigmoid")
    ax.plot(x, np.tanh(x), color=TEAL, lw=2.8, label="tanh")
    ax.plot(x, np.maximum(x, 0.0), color=GOLD, lw=2.8, label="ReLU")
    ax.axhline(0.0, color=INK, lw=0.8, alpha=0.35)
    ax.axvline(0.0, color=INK, lw=0.8, alpha=0.35)
    ax.set(xlabel=r"pre-activation $z$", ylabel=r"activation $\phi(z)$", ylim=(-1.2, 4.1))
    ax.grid(True)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    fig.savefig(OUTPUT_DIR / "activation_functions.png", dpi=180, transparent=True)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    save_regression_bridge()
    save_activation_functions()


if __name__ == "__main__":
    main()
