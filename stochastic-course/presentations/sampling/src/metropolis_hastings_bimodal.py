#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "imageio",
# ]
# ///

"""Independence Metropolis--Hastings sampling for a bimodal target."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MHResult:
    chain: np.ndarray
    accepted: np.ndarray
    running_mean: np.ndarray


TARGET_MEANS = np.array([-2.0, 2.0])
TARGET_STDS = np.array([0.25, 0.25])
TARGET_WEIGHTS = np.array([0.5, 0.5])

PROPOSAL_STDS = np.array([1.0, 1.0])
PROPOSAL_WEIGHTS = np.array([0.5, 0.5])


def normal_log_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    var = std**2
    return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


def target_log_pdf(x: np.ndarray) -> np.ndarray:
    components = np.stack(
        [
            np.log(weight) + normal_log_pdf(x, mean=mu, std=sigma)
            for weight, mu, sigma in zip(TARGET_WEIGHTS, TARGET_MEANS, TARGET_STDS)
        ]
    )
    max_comp = np.max(components, axis=0)
    return max_comp + np.log(np.sum(np.exp(components - max_comp), axis=0))


def proposal_log_pdf(x: np.ndarray) -> np.ndarray:
    components = np.stack(
        [
            np.log(weight) + normal_log_pdf(x, mean=mu, std=sigma)
            for weight, mu, sigma in zip(PROPOSAL_WEIGHTS, TARGET_MEANS, PROPOSAL_STDS)
        ]
    )
    max_comp = np.max(components, axis=0)
    return max_comp + np.log(np.sum(np.exp(components - max_comp), axis=0))


def sample_from_proposal(rng: np.random.Generator, size: int = 1) -> np.ndarray:
    components = rng.choice(len(PROPOSAL_WEIGHTS), size=size, p=PROPOSAL_WEIGHTS)
    means = TARGET_MEANS[components]
    scales = PROPOSAL_STDS[components]
    return rng.normal(loc=means, scale=scales)


def run_metropolis_hastings(
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
    initial: float | None = None,
) -> MHResult:
    total_steps = burn_in + n_samples
    chain = np.empty(total_steps)
    accepted = np.zeros(total_steps, dtype=bool)

    if initial is None:
        current = float(sample_from_proposal(rng, size=1)[0])
    else:
        current = initial
    current_target_logp = float(target_log_pdf(np.array([current]))[0])
    current_proposal_logp = float(proposal_log_pdf(np.array([current]))[0])

    for i in range(total_steps):
        proposal = float(sample_from_proposal(rng, size=1)[0])
        proposal_target_logp = float(target_log_pdf(np.array([proposal]))[0])
        proposal_proposal_logp = float(proposal_log_pdf(np.array([proposal]))[0])

        log_alpha = (proposal_target_logp + current_proposal_logp) - (current_target_logp + proposal_proposal_logp)
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_target_logp = proposal_target_logp
            current_proposal_logp = proposal_proposal_logp
            accepted[i] = True
        chain[i] = current

    post_burn_chain = chain[burn_in:]
    running_mean = np.cumsum(post_burn_chain) / np.arange(1, len(post_burn_chain) + 1)
    return MHResult(chain=chain, accepted=accepted, running_mean=running_mean)


def build_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_path = figures_dir / "mcmc_bimodal_final.png"
    gif_path = figures_dir / "mcmc_bimodal_evolution.gif"
    return final_path, gif_path


def render_final_plot(result: MHResult, burn_in: int, output: Path) -> None:
    chain = result.chain
    post_burn_chain = chain[burn_in:]
    iterations = np.arange(1, len(post_burn_chain) + 1)
    acceptance_rate = result.accepted[burn_in:].mean()

    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=0.3)
    ax_hist = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_trace = fig.add_subplot(right_spec[0, 0])
    ax_running = fig.add_subplot(right_spec[1, 0])

    grid = np.linspace(-4, 4, 400)
    target_pdf = np.exp(target_log_pdf(grid))
    ax_hist.hist(post_burn_chain, bins=40, density=True, color="#ff7f0e", alpha=0.6, label="Samples")
    ax_hist.plot(grid, target_pdf, color="#2ca02c", linewidth=2, label="Target density")
    ax_hist.set_title("Posterior approximation after burn-in")
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlabel("State")
    ax_hist.legend(loc="upper right")

    ax_trace.plot(chain, color="#1f77b4")
    ax_trace.axvline(burn_in, color="black", linestyle="--", linewidth=1.2, label="Burn-in end")
    ax_trace.set_title("Metropolis--Hastings trace")
    ax_trace.set_ylabel("State")
    ax_trace.legend(loc="upper right")

    ax_running.plot(iterations, result.running_mean, color="#d62728", label="Running mean")
    ax_running.hlines(0, 1, len(post_burn_chain), colors="black", linestyles=":", linewidth=1.2, label="True mean")
    ax_running.set_xlabel("Iteration")
    ax_running.set_ylabel("Running mean")
    ax_running.set_title("Running mean after burn-in")
    ax_running.legend(loc="upper right")
    ax_running_secondary = ax_running.twinx()
    ax_running_secondary.plot(iterations, np.cumsum(result.accepted[burn_in:]) / iterations, color="#2ca02c", linestyle="--", label="Acceptance rate")
    ax_running_secondary.set_ylabel("Acceptance rate")
    ax_running_secondary.set_ylim(0.0, 1.0)
    ax_running_secondary.grid(False)
    lines_r, labels_r = ax_running.get_legend_handles_labels()
    lines_s, labels_s = ax_running_secondary.get_legend_handles_labels()
    ax_running.legend(lines_r + lines_s, labels_r + labels_s, loc="upper right")
    ax_running_secondary.text(
        0.02,
        0.75,
        f"Overall acceptance ≈ {acceptance_rate:.2f}",
        transform=ax_running_secondary.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.suptitle("Metropolis--Hastings sampling of a bimodal target")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _frame_indices(n: int, frame_count: int) -> Iterable[int]:
    step = max(1, n // frame_count)
    return range(step, n + 1, step)


def render_gif(result: MHResult, burn_in: int, output: Path, frame_count: int = 50) -> None:
    chain = result.chain
    total_steps = len(chain)
    frames = []
    grid = np.linspace(-4, 4, 400)
    target_pdf = np.exp(target_log_pdf(grid))

    for stop in _frame_indices(total_steps, frame_count):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        ax_trace, ax_hist = axes

        ax_trace.plot(chain[:stop], color="#1f77b4")
        ax_trace.axvline(burn_in, color="black", linestyle="--", linewidth=1.0)
        ax_trace.set_xlim(0, total_steps)
        ax_trace.set_ylabel("State")
        ax_trace.set_title(f"Trace until iteration {stop}")

        hist_data = chain[max(burn_in, stop // 10) : stop]
        if hist_data.size > 0:
            ax_hist.hist(hist_data, bins=30, density=True, color="#ff7f0e", alpha=0.6)
        ax_hist.plot(grid, target_pdf, color="#2ca02c", linewidth=2)
        ax_hist.set_xlim(-4, 4)
        ax_hist.set_ylim(0, target_pdf.max() * 1.1)
        ax_hist.set_title("Histogram of recent samples")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=200)
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    imageio.mimsave(output, frames, duration=0.2)


def print_summary(result: MHResult, burn_in: int) -> None:
    post_burn_chain = result.chain[burn_in:]
    acceptance_rate = result.accepted[burn_in:].mean()
    print("Independence Metropolis--Hastings for bimodal Gaussian mixture\n")
    print("Target density: π(x) = 0.5 N(-2, 0.25^2) + 0.5 N(2, 0.25^2)")
    print("Proposal density: g(x) = 0.5 N(-2, 1^2) + 0.5 N(2, 1^2)")
    print("Acceptance rule: α(x,x') = min{1, [π(x')/g(x')]/[π(x)/g(x)]}")
    print(f"Post burn-in mean estimate: {post_burn_chain.mean():.4f}")
    print(f"Final running mean         : {result.running_mean[-1]:.4f}")
    print(f"Acceptance rate            : {acceptance_rate:.4f}")


def main() -> None:
    rng = np.random.default_rng(seed=2024)
    n_samples = 6000
    burn_in = 1000
    result = run_metropolis_hastings(
        n_samples=n_samples,
        burn_in=burn_in,
        rng=rng,
    )

    final_path, gif_path = build_paths()
    render_final_plot(result, burn_in, final_path)
    render_gif(result, burn_in, gif_path)
    print_summary(result, burn_in)


if __name__ == "__main__":
    main()
