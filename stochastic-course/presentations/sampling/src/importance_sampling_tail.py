#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "imageio",
# ]
# ///

"""Importance sampling demo estimating the mean of a bimodal mixture target."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ISResult:
    samples: np.ndarray
    weights: np.ndarray
    running_mean: np.ndarray
    ess_traj: np.ndarray


TARGET_MEANS = np.array([1.5, -1.0])
TARGET_STDS = np.array([0.6, 0.9])
TARGET_WEIGHTS = np.array([1.0, 0.5])  # unnormalised weights

PROPOSAL_MEANS = np.array([1.2, -1.2])
PROPOSAL_STDS = np.array([0.8, 0.8])
PROPOSAL_WEIGHTS = np.array([0.5, 0.5])  # proper mixture weights

SQRT_2PI = np.sqrt(2.0 * np.pi)


def target_unnormalised(x: np.ndarray) -> np.ndarray:
    term1 = TARGET_WEIGHTS[0] * np.exp(-0.5 * ((x - TARGET_MEANS[0]) / TARGET_STDS[0]) ** 2)
    term2 = TARGET_WEIGHTS[1] * np.exp(-0.5 * ((x - TARGET_MEANS[1]) / TARGET_STDS[1]) ** 2)
    return term1 + term2


def target_normalization_constant() -> float:
    return np.sum(TARGET_WEIGHTS * TARGET_STDS) * SQRT_2PI


def target_pdf(x: np.ndarray) -> np.ndarray:
    return target_unnormalised(x) / target_normalization_constant()


def proposal_pdf(x: np.ndarray) -> np.ndarray:
    components = []
    for weight, mean, std in zip(PROPOSAL_WEIGHTS, PROPOSAL_MEANS, PROPOSAL_STDS):
        components.append(weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * SQRT_2PI))
    return np.sum(components, axis=0)


def simulate_importance_sampling(
    n_samples: int,
    rng: np.random.Generator,
) -> ISResult:
    proposals = rng.choice(len(PROPOSAL_WEIGHTS), size=n_samples, p=PROPOSAL_WEIGHTS)
    samples = rng.normal(loc=PROPOSAL_MEANS[proposals], scale=PROPOSAL_STDS[proposals])

    target_vals = target_unnormalised(samples)
    proposal_vals = proposal_pdf(samples)
    unnormalised_weights = target_vals / proposal_vals

    f_values = samples  # estimate E_pi[X]
    cumulative_weight = np.cumsum(unnormalised_weights)
    running_mean = np.cumsum(unnormalised_weights * f_values) / cumulative_weight
    normalized_weights = unnormalised_weights / cumulative_weight[-1]
    ess_traj = cumulative_weight**2 / np.cumsum(unnormalised_weights**2)
    return ISResult(samples=samples, weights=normalized_weights, running_mean=running_mean, ess_traj=ess_traj)


def build_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_path = figures_dir / "is_gaussian_tail_final.png"
    gif_path = figures_dir / "is_gaussian_tail_evolution.gif"
    return final_path, gif_path


def render_final_plot(result: ISResult, output: Path) -> None:
    grid = np.linspace(-4, 4, 600)
    target_vals = target_pdf(grid)
    proposal_vals = proposal_pdf(grid)
    target_mean = np.trapezoid(grid * target_vals, grid)

    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=0.32)
    ax_density = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.2)
    ax_running = fig.add_subplot(right_spec[0, 0])
    ax_weights = fig.add_subplot(right_spec[1, 0])

    counts, bin_edges = np.histogram(result.samples, bins=40, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    ax_density.bar(
        bin_centers,
        counts,
        width=widths,
        color="#ffbb78",
        edgecolor="#d95f02",
        alpha=0.6,
        label="Proposal histogram",
    )
    ax_density.plot(grid, proposal_vals, color="#ff7f0e", linestyle="--", linewidth=1.5, label="Proposal pdf")
    ax_density.plot(grid, target_vals, label="Target mixture", color="#1f77b4")

    draw_count = min(1500, result.samples.size)
    sample_indices = np.linspace(0, result.samples.size - 1, draw_count, dtype=int)
    weights_subset = result.weights[sample_indices]
    if weights_subset.max() > 0:
        weight_scale = np.sqrt(weights_subset / weights_subset.max())
    else:
        weight_scale = weights_subset
    colors = cm.viridis(weight_scale)
    ax_density.scatter(
        result.samples[sample_indices],
        target_pdf(result.samples[sample_indices]),
        s=25,
        c=colors,
        alpha=0.7,
        edgecolor="k",
        linewidths=0.2,
        label="Proposal samples",
    )
    ax_density.set_title("Proposal histogram vs target/proposal densities")
    ax_density.set_ylim(bottom=0.0)
    ax_density.legend(loc="upper right")

    iterations = np.arange(1, len(result.samples) + 1)
    ax_running.plot(iterations, result.running_mean, color="#2ca02c", label="SNIS estimate of $E[X]$")
    ax_running.axhline(target_mean, color="black", linestyle=":", linewidth=1.5, label="True mean")
    ax_running.set_xlabel("Iteration")
    ax_running.set_ylabel("Estimate of $E[X]$")
    ax_running.set_title("Estimator convergence")
    ax_running.legend(loc="upper right")

    final_estimate = result.running_mean[-1]
    ax_running.text(
        0.02,
        0.1,
        f"Final estimate = {final_estimate:.3f}",
        transform=ax_running.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_weights.plot(iterations, result.ess_traj / iterations, color="#d62728")
    ax_weights.set_xlabel("Iteration")
    ax_weights.set_ylabel("ESS / N")
    ax_weights.set_ylim(0, 1.05)
    ax_weights.set_title("Effective sample size ratio")

    final_ess_ratio = result.ess_traj[-1] / len(result.samples)
    ax_weights.text(
        0.02,
        0.1,
        f"ESS / N = {final_ess_ratio:.2f}",
        transform=ax_weights.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.suptitle("Importance sampling for an unnormalised bimodal target")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _frame_indices(n: int, frame_count: int) -> Iterable[int]:
    if n <= frame_count:
        return list(range(1, n + 1))

    early = np.geomspace(1, min(500, n), num=min(6, max(frame_count // 4, 2)))
    early = np.unique(np.round(early).astype(int))
    tail_start = int(early[-1]) + 1 if early.size else 1
    remaining_count = max(frame_count - early.size, 1)
    remaining = np.linspace(tail_start, n, remaining_count, dtype=int)
    indices = np.unique(np.concatenate([early, remaining]))
    if indices[-1] != n:
        indices = np.append(indices, n)
    return indices.tolist()


def render_gif(result: ISResult, output: Path, frame_count: int = 40) -> None:
    iterations = np.arange(1, len(result.samples) + 1)
    frames = []
    grid = np.linspace(-4, 4, 400)
    target_norm = target_pdf(grid)
    proposal_norm = proposal_pdf(grid)
    target_mean = np.trapezoid(grid * target_norm, grid)

    for stop in _frame_indices(len(result.samples), frame_count):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_hist, ax_running = axes

        ax_hist.hist(result.samples[:stop], bins=40, density=True, color="#ff7f0e", alpha=0.35)
        ax_hist.plot(grid, target_norm, color="#1f77b4", label="Target")
        ax_hist.plot(grid, proposal_norm, color="#ff7f0e", linestyle="--", label="Proposal")
        scatter_start = max(0, stop - 600)
        scatter_slice = slice(scatter_start, stop)
        subset = result.samples[scatter_slice]
        subset_weights = result.weights[scatter_slice]
        if subset.size:
            weight_norm = subset_weights / subset_weights.max() if subset_weights.max() > 0 else subset_weights
            ax_hist.scatter(
                subset,
                target_pdf(subset),
                c=weight_norm,
                cmap="viridis",
                s=22,
                alpha=0.75,
                edgecolor="k",
                linewidths=0.2,
            )
        ax_hist.set_title(f"Samples up to iteration {stop}")
        ax_hist.legend(loc="upper right")

        ax_running.plot(iterations[:stop], result.running_mean[:stop], color="#2ca02c")
        ax_running.axhline(target_mean, color="black", linestyle=":", linewidth=1.2)
        ax_running.set_xlabel("Iteration")
        ax_running.set_ylabel("Estimate of $E[X]$")
        ax_running.set_ylim(target_mean - 1, target_mean + 1)
        ax_running.set_title("Running mean estimate")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=200)
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    imageio.mimsave(output, frames, duration=0.2)


def main() -> None:
    rng = np.random.default_rng(seed=321)
    n_samples = 20000
    result = simulate_importance_sampling(n_samples=n_samples, rng=rng)

    final_path, gif_path = build_paths()
    render_final_plot(result, final_path)
    render_gif(result, gif_path)


if __name__ == "__main__":
    main()
