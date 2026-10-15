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


def cross_correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply valid, unit-stride 2D cross-correlation."""
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both be two-dimensional")
    out_height = image.shape[0] - kernel.shape[0] + 1
    out_width = image.shape[1] - kernel.shape[1] + 1
    if out_height <= 0 or out_width <= 0:
        raise ValueError("kernel cannot be larger than the image")
    output = np.empty((out_height, out_width), dtype=float)
    for row in range(out_height):
        for column in range(out_width):
            window = image[row : row + kernel.shape[0], column : column + kernel.shape[1]]
            output[row, column] = np.sum(window * kernel)
    return output


def convolution_output_size(size: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    """Return one spatial output size for dilation one."""
    if min(size, kernel, stride) <= 0 or padding < 0:
        raise ValueError("size, kernel and stride must be positive; padding must be non-negative")
    return (size + 2 * padding - kernel) // stride + 1


def scaled_dot_product_attention(
    queries: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return attention output and row-normalized weights."""
    if queries.ndim != 2 or keys.ndim != 2 or values.ndim != 2:
        raise ValueError("queries, keys and values must be matrices")
    if queries.shape[1] != keys.shape[1] or keys.shape[0] != values.shape[0]:
        raise ValueError("incompatible attention shapes")
    scores = queries @ keys.T / np.sqrt(keys.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ values, weights


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray) -> None:
    threshold = (float(matrix.min()) + float(matrix.max())) / 2.0
    for row, column in np.ndindex(matrix.shape):
        color = "white" if matrix[row, column] > threshold else INK
        ax.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color=color)


def save_convolution_walkthrough() -> None:
    image = np.array([[1, 2, 0], [0, 1, 3], [2, 1, 0]], dtype=float)
    kernel = np.array([[1, 0], [0, -1]], dtype=float)
    output = cross_correlate2d(image, kernel)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)
    for ax, matrix, title in zip(axes, (image, kernel, output), ("input $X$", "kernel $K$", r"$X \star K$"), strict=True):
        ax.imshow(matrix, cmap="viridis", alpha=0.82)
        annotate_matrix(ax, matrix)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUTPUT_DIR / "convolution_walkthrough.png", dpi=180, transparent=True)
    plt.close(fig)


def save_receptive_fields() -> None:
    sizes = np.arange(1, 8)
    receptive_field = 1 + 2 * sizes
    fig, ax = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    ax.fill_between(sizes, 1, receptive_field, color=TEAL, alpha=0.22)
    ax.plot(sizes, receptive_field, marker="o", color=TEAL, lw=2.8)
    ax.set(xlabel=r"stacked $3\times3$ convolutional layers", ylabel="receptive-field width", xticks=sizes)
    ax.grid(True, alpha=0.18)
    fig.savefig(OUTPUT_DIR / "receptive_field_growth.png", dpi=180, transparent=True)
    plt.close(fig)


def save_attention_heatmap() -> None:
    embeddings = np.array([[1.0, 0.1], [0.2, 1.0], [0.8, 0.4], [-0.3, 0.9]])
    _, weights = scaled_dot_product_attention(embeddings, embeddings, embeddings)
    labels = ["load", "rises", "then", "falls"]
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    image = ax.imshow(weights, cmap="viridis", vmin=0.0, vmax=weights.max(), alpha=0.84)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set(xlabel="key token", ylabel="query token", title="A toy self-attention matrix")
    for row, column in np.ndindex(weights.shape):
        ax.text(column, row, f"{weights[row, column]:.2f}", ha="center", va="center", color="white" if weights[row, column] > 0.3 else INK)
    fig.colorbar(image, ax=ax, shrink=0.78, label="attention weight")
    fig.savefig(OUTPUT_DIR / "attention_heatmap.png", dpi=180, transparent=True)
    plt.close(fig)


def save_memory_decay() -> None:
    steps = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    for factor, color, label in ((0.6, CORAL, "$0.6^k$"), (0.85, GOLD, "$0.85^k$"), (0.97, BLUE, "$0.97^k$")):
        ax.plot(steps, factor**steps, lw=2.6, color=color, alpha=0.9, label=label)
    ax.set(xlabel="time steps $k$", ylabel="gradient multiplier", ylim=(-0.03, 1.03))
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "recurrent_gradient_decay.png", dpi=180, transparent=True)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_convolution_walkthrough()
    save_receptive_fields()
    save_attention_heatmap()
    save_memory_decay()


if __name__ == "__main__":
    main()
