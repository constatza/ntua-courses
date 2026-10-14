#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

"""Acceptance--rejection sampling for the bimodal Gaussian mixture target."""

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
    m_bound: float


TARGET_MEANS = np.array([-2.0, 2.0])
TARGET_STDS = np.array([0.25, 0.25])
TARGET_WEIGHTS = np.array([0.5, 0.5])

PROPOSAL_STDS = np.array([1.0, 1.0])
PROPOSAL_WEIGHTS = np.array([0.5, 0.5])


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    var = std**2
    return np.exp(-0.5 * ((x - mean) ** 2) / var) / (np.sqrt(2 * np.pi * var))


def target_pdf(x: np.ndarray) -> np.ndarray:
    components = [
        weight * normal_pdf(x, mean=mu, std=sigma) for weight, mu, sigma in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS)
    ]
    return np.sum(components, axis=0)


def proposal_pdf(x: np.ndarray) -> np.ndarray:
    components = [
        weight * normal_pdf(x, mean=mu, std=sigma) for weight, mu, sigma in zip(PROPOSAL_WEIGHTS, TARGET_MEANS, PROPOSAL_STDS)
    ]
    return np.sum(components, axis=0)


def envelope_constant() -> float:
    grid = np.linspace(-8.0, 8.0, 4000)
    ratio = target_pdf(grid) / proposal_pdf(grid)
    return float(1.2 * np.max(ratio))


def simulate_acceptance_rejection(n_samples: int, rng: np.random.Generator) -> ARResult:
    m_bound = envelope_constant()
    components = rng.choice(len(PROPOSAL_WEIGHTS), size=n_samples, p=PROPOSAL_WEIGHTS)
    proposals = rng.normal(loc=TARGET_MEANS[components], scale=PROPOSAL_STDS[components])
    uniforms = rng.random(n_samples)

    target_vals = target_pdf(proposals)
    proposal_vals = proposal_pdf(proposals)
    accept_prob = target_vals / (m_bound * proposal_vals)
    accepted = uniforms <= accept_prob

    heights = uniforms * m_bound * proposal_vals
    running_acceptance = np.cumsum(accepted) / np.arange(1, n_samples + 1)
    return ARResult(
        proposals=proposals,
        heights=heights,
        accepted=accepted,
        running_acceptance=running_acceptance,
        m_bound=m_bound,
    )


def build_output_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / "ars_bimodal_comparison.png"


def render_plot(result: ARResult, output: Path) -> None:
    grid = np.linspace(-6.0, 6.0, 800)
    target_vals = target_pdf(grid)
    proposal_vals = proposal_pdf(grid)
    envelope_vals = result.m_bound * proposal_vals

    fig = plt.figure(figsize=(13.5, 6.6))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.15, 1.0), wspace=0.32)
    ax_density = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.22)
    ax_mean = fig.add_subplot(right_spec[0, 0])
    ax_accept = fig.add_subplot(right_spec[1, 0])

    ax_density.plot(grid, target_vals, color="#1f77b4", linewidth=2.0, label=r"Target $\pi(x)$")
    ax_density.plot(grid, proposal_vals, color="#800020", linestyle="--", linewidth=1.5, label=r"Proposal $g(x)$")
    ax_density.plot(grid, envelope_vals, color="#2ca02c", linestyle="-.", linewidth=1.5, label=r"Envelope $M g(x)$")
    accepted = result.accepted
    accepted_samples = result.proposals[accepted]
    ax_density.hist(
        accepted_samples,
        bins=40,
        range=(-6.0, 6.0),
        density=True,
        color="#1b9e77",
        alpha=0.3,
        label="Accepted histogram",
    )
    ax_density.scatter(
        result.proposals[~accepted],
        result.heights[~accepted],
        s=14,
        alpha=0.35,
        color="#d95f02",
        label="Rejected proposals",
    )
    ax_density.scatter(
        accepted_samples,
        result.heights[accepted],
        s=16,
        alpha=0.55,
        color="#1b9e77",
        label="Accepted proposals",
    )
    ax_density.plot(grid, target_vals, color="black", linestyle=":", linewidth=1.4, label=r"$u \leq \pi(x)/(M g(x))$")
    ax_density.set_ylabel("Density")
    ax_density.set_title("Envelope and sampling outcomes")
    ax_density.set_ylim(0.0, envelope_vals.max() * 1.1)
    ax_density.legend(loc="upper center", ncol=3)

    x_axis = np.arange(1, accepted_samples.size + 1) if accepted_samples.size else np.array([])
    if accepted_samples.size:
        running_mean = np.cumsum(accepted_samples) / np.arange(1, accepted_samples.size + 1)
        ax_mean.plot(x_axis, running_mean, color="#d62728", label="Running mean")
    ax_mean.axhline(0.0, color="black", linestyle=":", linewidth=1.3, label="True mean")
    ax_mean.set_ylabel("Running mean")
    ax_mean.set_title("Running mean of accepted draws")
    ax_mean.set_xlabel("Accepted sample index")
    ax_mean.legend(loc="lower right")

    draw_indices = np.arange(1, result.running_acceptance.size + 1)
    ax_accept.plot(draw_indices, result.running_acceptance, color="#2ca02c")
    ax_accept.set_xlabel("Draw index")
    ax_accept.set_ylabel("Acceptance rate")
    ax_accept.set_ylim(0.0, 1.0)
    ax_accept.set_title("Running acceptance probability")
    ax_accept.text(
        0.02,
        0.15,
        f"Final acceptance = {result.running_acceptance[-1]:.3f}",
        transform=ax_accept.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle("Acceptance--rejection attempt for the bimodal mixture target")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def print_summary(result: ARResult) -> None:
    print("Acceptance--rejection for bimodal Gaussian mixture\n")
    print("Target density: π(x) = 0.5 N(-2, 0.25^2) + 0.5 N(2, 0.25^2)")
    print("Proposal density: g(x) = 0.5 N(-2, 1^2) + 0.5 N(2, 1^2)")
    print("Acceptance rule: accept x' if u ≤ π(x')/(M g(x')) with M ≈ {:.3f}".format(result.m_bound))
    print(f"Simulated acceptance rate: {result.running_acceptance[-1]:.4f}")
    accepted_samples = result.proposals[result.accepted]
    running_mean = np.cumsum(accepted_samples) / np.arange(1, accepted_samples.size + 1)
    print(f"Running mean after {accepted_samples.size} accepts: {running_mean[-1]:.4f} (true mean = 0)")


def main() -> None:
    rng = np.random.default_rng(seed=2024)
    n_samples = 12000
    result = simulate_acceptance_rejection(n_samples=n_samples, rng=rng)
    output = build_output_path()
    render_plot(result, output)
    print_summary(result)


if __name__ == "__main__":
    main()
