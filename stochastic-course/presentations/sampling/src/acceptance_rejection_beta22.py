#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

"""Acceptance--rejection sampling for a Beta(2, 2) target on [0, 1].

This script provides two ready-to-use slide assets:

1. Numeric summary for the "Problem & Solution" slide:
   - Defines the target density, proposal, bounding constant, and theoretical acceptance rate.
   - Reports simulated acceptance statistics plus moment estimates for the accepted samples.

2. A figure for the "Sampling Illustration" slide that shows:
   - Target pdf alongside the uniform proposal and its scaled envelope.
   - Simulated (x, y) pairs with accepted and rejected points visualised relative to the envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ARResult:
    proposals: np.ndarray
    heights: np.ndarray
    accepted: np.ndarray
    running_acceptance: np.ndarray


M_BOUND = 1.5  # Envelope constant (3 / 2)
THEORETICAL_ACCEPTANCE = 1.0 / M_BOUND  # 2 / 3
TRUE_MEAN = 0.5
TRUE_VARIANCE = 1.0 / 20.0
TRUE_SECOND_MOMENT = TRUE_VARIANCE + TRUE_MEAN**2


def target_pdf(x: np.ndarray) -> np.ndarray:
    """Density of Beta(2, 2): f(x) = 6 x (1 - x) for x in [0, 1]."""
    inside = (0.0 <= x) & (x <= 1.0)
    pdf = np.zeros_like(x, dtype=float)
    pdf[inside] = 6.0 * x[inside] * (1.0 - x[inside])
    return pdf


def proposal_pdf(x: np.ndarray) -> np.ndarray:
    """Uniform(0, 1) proposal."""
    return np.where((0.0 <= x) & (x <= 1.0), 1.0, 0.0)


def simulate_acceptance_rejection(
    n_samples: int,
    rng: np.random.Generator,
) -> ARResult:
    proposals = rng.random(n_samples)
    uniforms = rng.random(n_samples)

    target_vals = target_pdf(proposals)
    accept_prob = target_vals / M_BOUND
    accepted = uniforms <= accept_prob

    running_acceptance = np.cumsum(accepted) / np.arange(1, n_samples + 1)
    heights = uniforms * M_BOUND  # Useful for the scatter plot (y-axis)

    return ARResult(
        proposals=proposals,
        heights=heights,
        accepted=accepted,
        running_acceptance=running_acceptance,
    )


def build_output_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / "ars_beta22_summary.png"


def render_plot(result: ARResult, output: Path) -> None:
    grid = np.linspace(0.0, 1.0, 400)
    target_vals = target_pdf(grid)
    proposal_vals = proposal_pdf(grid)
    envelope_vals = M_BOUND * proposal_vals
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.15, 1.0), wspace=0.32)
    ax_pdfs = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.2)
    ax_mean = fig.add_subplot(right_spec[0, 0])
    ax_accept = fig.add_subplot(right_spec[1, 0])

    ax_pdfs.plot(grid, target_vals, color="#1f77b4", linewidth=2.0, label="Target $f(x)=6x(1-x)$")
    ax_pdfs.plot(grid, proposal_vals, color="#800020", linestyle="--", linewidth=1.5, label="Proposal $g(x)$")
    ax_pdfs.plot(grid, envelope_vals, color="#2ca02c", linestyle="-.", linewidth=1.5, label=r"Envelope $M g(x)$")
    ax_pdfs.fill_between(grid, target_vals, color="#1f77b4", alpha=0.15)
    ax_pdfs.fill_between(grid, envelope_vals, color="#2ca02c", alpha=0.08)
    accepted_samples = result.proposals[result.accepted]
    ax_pdfs.hist(
        accepted_samples,
        bins=30,
        range=(0.0, 1.0),
        density=True,
        color="#1b9e77",
        alpha=0.35,
        label="Accepted histogram",
    )
    ax_pdfs.set_xlabel("x")
    ax_pdfs.set_ylabel("Density")
    ax_pdfs.set_ylim(bottom=0.0, top=M_BOUND * 1.05)
    ax_pdfs.set_title("Target vs proposal and envelope")
    ax_pdfs.legend(loc="upper center", ncol=2)

    accepted = result.accepted
    ax_pdfs.scatter(
        result.proposals[~accepted],
        result.heights[~accepted],
        s=18,
        alpha=0.35,
        color="#d95f02",
        label="Rejected proposals",
    )
    ax_pdfs.scatter(
        result.proposals[accepted],
        result.heights[accepted],
        s=18,
        alpha=0.6,
        color="#1b9e77",
        label="Accepted proposals",
    )
    ax_pdfs.plot(grid, target_pdf(grid), color="black", linestyle=":", linewidth=1.2, label="Acceptance boundary")
    ax_pdfs.set_ylim(0.0, M_BOUND * 1.1)

    accepted_samples = result.proposals[accepted]
    accepted_indices = np.flatnonzero(accepted)
    if accepted_samples.size:
        running_mean = np.cumsum(accepted_samples) / np.arange(1, accepted_samples.size + 1)
        ax_mean.plot(np.arange(1, accepted_samples.size + 1), running_mean, color="#1f77b4", label="Running mean")
    ax_mean.axhline(TRUE_MEAN, color="black", linestyle=":", linewidth=1.2, label="True mean")
    ax_mean.set_ylabel("Running E[X]")
    ax_mean.set_title("Running mean of accepted draws")
    ax_mean.set_xlabel("Accepted sample index")
    ax_mean.legend(loc="lower right")

    draw_axis = np.arange(1, result.running_acceptance.size + 1)
    ax_accept.plot(draw_axis, result.running_acceptance, color="#2ca02c", label="Running acceptance")
    ax_accept.axhline(THEORETICAL_ACCEPTANCE, color="black", linestyle=":", linewidth=1.2, label="Theoretical 2/3")
    ax_accept.set_xlabel("Draw index")
    ax_accept.set_ylabel("Acceptance rate")
    ax_accept.set_ylim(0.0, 1.0)
    ax_accept.set_title("Running acceptance probability")
    ax_accept.legend(loc="lower right")

    fig.suptitle("Acceptance--Rejection Sampling for Beta(2, 2) target", y=0.98)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def print_summary(result: ARResult) -> None:
    accepted_samples = result.proposals[result.accepted]
    n_draws = result.proposals.size
    n_accepted = accepted_samples.size
    acceptance_rate = n_accepted / n_draws

    mc_mean = accepted_samples.mean()
    mc_var = accepted_samples.var(ddof=0)
    mc_second_moment = np.mean(accepted_samples**2)

    print("Acceptance--Rejection sampling for Beta(2, 2) target on [0, 1]\n")
    print("Target density      : f(x) = 6 x (1 - x)")
    print("Proposal density    : g(x) = Uniform(0, 1)")
    print(f"Envelope constant   : M = {M_BOUND:.1f}")
    print(f"Theoretical accept. : {THEORETICAL_ACCEPTANCE:.4f} (i.e. 2 / 3)")
    print(f"Simulated draws     : {n_draws}")
    print(f"Accepted samples    : {n_accepted}")
    print(f"Simulated accept.   : {acceptance_rate:.4f}\n")

    print("Moment comparison (accepted samples):")
    print(f"  True mean               = {TRUE_MEAN:.4f}")
    print(f"  Estimated mean          = {mc_mean:.4f}")
    print(f"  True E[X^2]             = {TRUE_SECOND_MOMENT:.4f}")
    print(f"  Estimated E[X^2]        = {mc_second_moment:.4f}")
    print(f"  True variance           = {TRUE_VARIANCE:.4f}")
    print(f"  Estimated variance      = {mc_var:.4f}")


def main() -> None:
    rng = np.random.default_rng(seed=2024)
    n_samples = 8000
    result = simulate_acceptance_rejection(n_samples=n_samples, rng=rng)
    figure_path = build_output_path()
    render_plot(result, figure_path)
    print_summary(result)


if __name__ == "__main__":
    main()
