#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

"""Importance sampling demo for a triangular target using a uniform proposal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class TriangularISResult:
    samples: np.ndarray
    weights: np.ndarray
    running_moment2: np.ndarray
    running_tail_prob: np.ndarray
    ess_traj: np.ndarray


A = 0.0
B = 1.0
MODE = 0.8
TAIL_THRESHOLD = 0.9
VARIANCE = (A**2 + B**2 + MODE**2 - A * B - A * MODE - B * MODE) / 18.0
TRUE_MEAN = (A + B + MODE) / 3.0
TRUE_MOMENT2 = VARIANCE + TRUE_MEAN**2
TRUE_TAIL_PROB = (B - TAIL_THRESHOLD) ** 2 / ((B - A) * (B - MODE))


def triangular_pdf(x: np.ndarray) -> np.ndarray:
    pdf = np.zeros_like(x, dtype=float)
    left_mask = (A <= x) & (x <= MODE)
    right_mask = (MODE < x) & (x <= B)
    if MODE > A:
        pdf[left_mask] = 2.0 * (x[left_mask] - A) / ((B - A) * (MODE - A))
    if B > MODE:
        pdf[right_mask] = 2.0 * (B - x[right_mask]) / ((B - A) * (B - MODE))
    return pdf


def uniform_pdf(x: np.ndarray) -> np.ndarray:
    inside = (A <= x) & (x <= B)
    return inside.astype(float) / (B - A)


def simulate_importance_sampling(n_samples: int, rng: np.random.Generator) -> TriangularISResult:
    samples = rng.uniform(low=A, high=B, size=n_samples)
    target_vals = triangular_pdf(samples)
    proposal_vals = uniform_pdf(samples)
    weights = target_vals / proposal_vals

    cumulative_weight = np.cumsum(weights)
    running_moment2 = np.cumsum(weights * samples**2) / cumulative_weight
    tail_indicator = (samples > TAIL_THRESHOLD).astype(float)
    running_tail_prob = np.cumsum(weights * tail_indicator) / cumulative_weight
    ess_traj = cumulative_weight**2 / np.cumsum(weights**2)
    normalised_weights = weights / cumulative_weight[-1]
    return TriangularISResult(
        samples=samples,
        weights=normalised_weights,
        running_moment2=running_moment2,
        running_tail_prob=running_tail_prob,
        ess_traj=ess_traj,
    )


def build_paths() -> Path:
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / "is_triangular_demo.png"


def render_plot(result: TriangularISResult, output: Path) -> None:
    grid = np.linspace(0.0, 1.0, 400)
    target_vals = triangular_pdf(grid)
    proposal_vals = uniform_pdf(grid)

    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=0.32)
    ax_pdf = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.2)
    ax_moment = fig.add_subplot(right_spec[0, 0])
    ax_tail = fig.add_subplot(right_spec[1, 0])

    hist_counts, bin_edges = np.histogram(
        result.samples,
        bins=40,
        range=(0.0, 1.0),
        weights=result.weights,
        density=True,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    ax_pdf.bar(
        bin_centers,
        hist_counts,
        width=widths,
        color="#ffdd99",
        edgecolor="#d95f02",
        alpha=0.7,
        label="Weighted histogram",
    )
    ax_pdf.plot(grid, proposal_vals, linestyle="--", color="#ff7f0e", label="Proposal $\\mathcal{U}(0,1)$")
    ax_pdf.plot(grid, target_vals, color="#1f77b4", linewidth=1.5, label="Target triangular pdf")
    ax_pdf.set_ylim(bottom=0.0)
    ax_pdf.set_title("Importance-weighted samples vs pdfs")
    ax_pdf.legend(loc="upper left")

    iterations = np.arange(1, result.samples.size + 1)
    ax_moment.plot(iterations, result.running_moment2, color="#2ca02c", label="IS estimate")
    ax_moment.axhline(TRUE_MOMENT2, color="black", linestyle=":", linewidth=1.4, label="True $E[X^2]$")
    ax_moment.set_xlabel("Iteration")
    ax_moment.set_ylabel("Estimate of $E[X^2]$")
    ax_moment.set_title("Convergence for $E[X^2]$")
    ax_moment.legend(loc="lower right")

    ax_tail.plot(iterations, result.running_tail_prob, color="#d62728", label="IS estimate")
    ax_tail.axhline(TRUE_TAIL_PROB, color="black", linestyle=":", linewidth=1.4, label=f"True $P(X>{TAIL_THRESHOLD})$")
    ax_tail.set_xlabel("Iteration")
    ax_tail.set_ylabel(f"Estimate of $P(X>{TAIL_THRESHOLD})$")
    ax_tail.set_title("Heavy tail probability convergence")
    ax_tail.legend(loc="lower right")

    final_ess_ratio = result.ess_traj[-1] / result.samples.size
    ax_tail.text(
        0.02,
        0.1,
        f"ESS / N = {final_ess_ratio:.2f}",
        transform=ax_tail.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle("Importance sampling to emulate a triangular target")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def print_numerical_summary(result: TriangularISResult) -> None:
    final_moment_est = result.running_moment2[-1]
    final_tail_est = result.running_tail_prob[-1]
    ess = result.ess_traj[-1]

    uniform_estimate = np.mean(result.samples**2)
    uniform_tail = np.mean(result.samples > TAIL_THRESHOLD)

    print("Target: triangular distribution on [0, 1] with mode at 0.8\n")
    print(f"True mean               : {TRUE_MEAN:.4f}")
    print(f"True E[X^2]             : {TRUE_MOMENT2:.4f}")
    print(f"True P(X > {TAIL_THRESHOLD}) : {TRUE_TAIL_PROB:.4f}\n")
    print("Importance sampling with proposal U(0,1):")
    print(f"  Weighted E[X^2] estimate   = {final_moment_est:.4f}")
    print(f"  Weighted tail probability  = {final_tail_est:.4f}")
    print(f"  Effective sample size      = {ess:.1f} (out of {result.samples.size})\n")
    print("Plain uniform Monte Carlo (no weighting, for comparison):")
    print(f"  Unweighted E[X^2] estimate = {uniform_estimate:.4f}")
    print(f"  Unweighted tail probability= {uniform_tail:.4f}")


def main() -> None:
    rng = np.random.default_rng(seed=2024)
    n_samples = 6000
    result = simulate_importance_sampling(n_samples=n_samples, rng=rng)
    figure_path = build_paths()
    render_plot(result, figure_path)
    print_numerical_summary(result)


if __name__ == "__main__":
    main()
