#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

"""Independence Metropolis--Hastings sampling for a triangular target on [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MHResult:
    chain: np.ndarray
    accepted: np.ndarray
    running_mean: np.ndarray


A = 0.0
B = 1.0
MODE = 0.8
TRUE_MEAN = (A + B + MODE) / 3.0


def triangular_pdf(x: np.ndarray) -> np.ndarray:
    pdf = np.zeros_like(x, dtype=float)
    left = (A <= x) & (x <= MODE)
    right = (MODE < x) & (x <= B)
    if MODE > A:
        pdf[left] = 2.0 * (x[left] - A) / ((B - A) * (MODE - A))
    if MODE < B:
        pdf[right] = 2.0 * (B - x[right]) / ((B - A) * (B - MODE))
    return pdf


def run_independence_mh(
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
    initial: float = 0.5,
) -> MHResult:
    total_steps = burn_in + n_samples
    chain = np.empty(total_steps)
    accepted = np.zeros(total_steps, dtype=bool)

    current = np.clip(initial, A, B)
    current_pdf = float(triangular_pdf(np.array([current]))[0])

    for i in range(total_steps):
        proposal = rng.uniform(A, B)
        proposal_pdf = float(triangular_pdf(np.array([proposal]))[0])

        if current_pdf <= 0.0:
            accept_prob = 1.0
        else:
            accept_prob = min(1.0, proposal_pdf / current_pdf)

        if rng.uniform() < accept_prob:
            current = proposal
            current_pdf = proposal_pdf
            accepted[i] = True
        chain[i] = current

    post_burn_chain = chain[burn_in:]
    running_mean = np.cumsum(post_burn_chain) / np.arange(1, post_burn_chain.size + 1)
    return MHResult(chain=chain, accepted=accepted, running_mean=running_mean)


def build_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / "mcmc_triangular_final.png"


def render_diagnostics(result: MHResult, burn_in: int, output: Path) -> None:
    chain = result.chain
    post_burn_chain = chain[burn_in:]
    iterations = np.arange(1, post_burn_chain.size + 1)
    acceptance_rate = result.accepted[burn_in:].mean()

    fig = plt.figure(figsize=(15, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.1, 1.0), wspace=0.3)
    ax_hist = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_trace = fig.add_subplot(right_spec[0, 0])
    ax_running = fig.add_subplot(right_spec[1, 0])

    grid = np.linspace(0.0, 1.0, 400)
    target_vals = triangular_pdf(grid)
    ax_hist.hist(post_burn_chain, bins=40, density=True, color="#ff7f0e", alpha=0.6, label="Samples")
    ax_hist.plot(grid, target_vals, color="#2ca02c", linewidth=2, label="Target $f(x)$")
    ax_hist.set_ylabel("Density")
    ax_hist.set_xlabel("State")
    ax_hist.set_title("Posterior approximation after burn-in")
    ax_hist.legend(loc="upper left")

    ax_trace.plot(chain, color="#1f77b4")
    ax_trace.axvline(burn_in, color="black", linestyle="--", linewidth=1.2, label="Burn-in end")
    ax_trace.set_ylabel("$X_t$")
    ax_trace.set_title("Independence MH trace on triangular target")
    ax_trace.legend(loc="upper right")

    ax_running.plot(iterations, result.running_mean, color="#d62728", label="$\\widehat{\\mathbb{E}}[X]$")
    ax_running.hlines(TRUE_MEAN, 1, post_burn_chain.size, colors="black", linestyles=":", linewidth=1.2, label="$\\mathbb{E}[X]$")
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
    fig.suptitle("Independence MH targeting a triangular density on [0, 1]")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def print_summary(result: MHResult, burn_in: int) -> None:
    chain = result.chain
    post_burn_chain = chain[burn_in:]
    print("Independence Metropolis--Hastings for triangular target on [0, 1]\n")
    print("Target density: f(x) = \\begin{cases} \\frac{2(x-A)}{(B-A)(MODE-A)} & x\\le MODE \\\\ \\frac{2(B-x)}{(B-A)(B-MODE)} & x>MODE \\end{cases}")
    print("Proposal density: g(x) = \\mathrm{Uniform}(0,1)")
    print("Acceptance rule: \\alpha(x,x') = \\min\\{1, f(x') / f(x)\\}\n")
    print(f"Mean estimate after burn-in: {post_burn_chain.mean():.4f}")
    print(f"Running mean final value   : {result.running_mean[-1]:.4f}")
    print(f"Acceptance rate            : {result.accepted[burn_in:].mean():.4f}")


def main() -> None:
    rng = np.random.default_rng(seed=2024)
    n_samples = 4000
    burn_in = 500
    result = run_independence_mh(n_samples=n_samples, burn_in=burn_in, rng=rng)
    output = build_path()
    render_diagnostics(result, burn_in, output)
    print_summary(result, burn_in)


if __name__ == "__main__":
    main()
