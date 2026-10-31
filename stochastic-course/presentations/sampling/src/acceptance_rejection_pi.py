#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "imageio",
# ]
# ///

"""Acceptance--rejection sampling demo for estimating pi.

The script simulates proposals on the square [-1, 1]^2, accepts points inside
the unit disk, and tracks a running estimate of pi. Outputs include a summary
figure and an animated GIF showing the sampling evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SimulationResult:
    points: np.ndarray
    accepted: np.ndarray
    running_pi: np.ndarray


def simulate_acceptance_rejection(n_samples: int, rng: np.random.Generator) -> SimulationResult:
    """Run acceptance--rejection sampling on the unit square."""
    points = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    radius_sq = np.sum(points**2, axis=1)
    accepted = radius_sq <= 1.0
    running_pi = 4.0 * np.cumsum(accepted) / np.arange(1, n_samples + 1)
    return SimulationResult(points=points, accepted=accepted, running_pi=running_pi)


def build_paths() -> tuple[Path, Path, Path]:
    """Return figure directory, static figure path, and GIF path."""
    base_dir = Path(__file__).resolve().parent
    figures_dir = (base_dir / "../assets/generated").resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_plot = figures_dir / "ars_pi_final.png"
    gif_path = figures_dir / "ars_pi_evolution.gif"
    return figures_dir, final_plot, gif_path


def render_final_plot(result: SimulationResult, output: Path) -> None:
    """Render the summary figure showing proposals and running estimate."""
    points = result.points
    accepted = result.accepted
    running_pi = result.running_pi
    iterations = np.arange(1, len(points) + 1)

    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=(1.1, 1.0), wspace=0.32)
    ax_hist = fig.add_subplot(gs[0, 0])
    right_spec = gs[0, 1].subgridspec(2, 1, hspace=0.22)
    ax_running = fig.add_subplot(right_spec[0, 0])
    ax_accept = fig.add_subplot(right_spec[1, 0])

    accepted_points = points[accepted]
    radii = np.linalg.norm(accepted_points, axis=1)
    ax_hist.hist(
        radii,
        bins=30,
        range=(0.0, 1.0),
        density=True,
        color="#1b9e77",
        alpha=0.35,
        label="Histogram of accepted radii",
    )
    r_grid = np.linspace(0.0, 1.0, 200)
    radial_pdf = 2.0 * r_grid  # uniform disk => f_R(r)=2r
    ax_hist.plot(r_grid, radial_pdf, color="#800020", linestyle="--", linewidth=1.5, label="$f_R(r)=2r$")
    ax_hist.set_xlabel("Radius of accepted point $r$")
    all_radii = np.linalg.norm(points, axis=1)
    accepted_radii = all_radii[accepted]
    rejected_radii = all_radii[~accepted]
    ax_hist.scatter(
        accepted_radii,
        np.minimum(2.0 * np.clip(accepted_radii, None, 1.0), 2.0),
        s=18,
        color="#1b9e77",
        alpha=0.6,
        label="Accepted proposals",
    )
    ax_hist.scatter(
        rejected_radii,
        np.zeros_like(rejected_radii),
        s=16,
        color="#d95f02",
        alpha=0.35,
        label="Rejected proposals",
    )
    ax_hist.set_ylabel("Density / scaled height")
    ax_hist.set_title("Radial pdf and proposal outcomes")
    ax_hist.set_ylim(0.0, 2.2)
    ax_hist.legend(loc="upper left")

    ax_running.plot(iterations, running_pi, color="#1f77b4", label="Running estimate")
    ax_running.axhline(np.pi, color="black", linestyle=":", linewidth=1.5, label="$\\pi$")
    ax_running.set_xlabel("Iteration")
    ax_running.set_ylabel("Estimate of $\\pi$")
    ax_running.set_title("Running estimate of $\\pi$")
    ax_running.legend(loc="upper right")

    final_estimate = running_pi[-1]
    acceptance_rate = np.mean(accepted)
    ax_running.text(
        0.02,
        0.05,
        f"Final estimate = {final_estimate:.5f}\nAcceptance rate = {acceptance_rate:.3f}",
        transform=ax_running.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_accept.plot(iterations, np.cumsum(accepted) / iterations, color="#2ca02c")
    ax_accept.axhline(acceptance_rate, color="black", linestyle=":", linewidth=1.2)
    ax_accept.set_xlabel("Iteration")
    ax_accept.set_ylabel("Acceptance rate")
    ax_accept.set_ylim(0.0, 1.0)
    ax_accept.set_title("Running acceptance probability")
    ax_accept.text(
        0.02,
        0.15,
        f"Final acceptance = {acceptance_rate:.3f}",
        transform=ax_accept.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle("Acceptance--Rejection Sampling for $\\pi$")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _frame_indices(n: int, frame_count: int) -> Iterable[int]:
    step = max(1, n // frame_count)
    return range(step, n + 1, step)


def render_gif(result: SimulationResult, output: Path, frame_count: int = 40) -> None:
    """Render an animated GIF showing the sampling evolution."""
    points = result.points
    accepted = result.accepted
    running_pi = result.running_pi
    iterations = np.arange(1, len(points) + 1)

    frames = []
    for stop in _frame_indices(len(points), frame_count):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_scatter, ax_running = axes

        ax_scatter.scatter(
            points[:stop][~accepted[:stop], 0],
            points[:stop][~accepted[:stop], 1],
            s=10,
            color="#d95f02",
            alpha=0.5,
        )
        ax_scatter.scatter(
            points[:stop][accepted[:stop], 0],
            points[:stop][accepted[:stop], 1],
            s=10,
            color="#1b9e77",
            alpha=0.7,
        )
        circle = plt.Circle((0, 0), 1, color="black", fill=False, linestyle="--", linewidth=1.2)
        ax_scatter.add_patch(circle)
        ax_scatter.set_xlim(-1.05, 1.05)
        ax_scatter.set_ylim(-1.05, 1.05)
        ax_scatter.set_aspect("equal")
        ax_scatter.set_title(f"Proposals up to iteration {stop}")

        ax_running.plot(iterations[:stop], running_pi[:stop], color="#1f77b4")
        ax_running.axhline(np.pi, color="black", linestyle=":", linewidth=1.2)
        ax_running.set_xlabel("Iteration")
        ax_running.set_ylabel("Estimate of $\\pi$")
        ax_running.set_ylim(2.5, 4.0)
        ax_running.set_title("Running estimate")

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=200)
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    imageio.mimsave(output, frames, duration=0.2)


def main() -> None:
    rng = np.random.default_rng(seed=42)
    n_samples = 5000
    result = simulate_acceptance_rejection(n_samples=n_samples, rng=rng)

    _, final_plot, gif_path = build_paths()
    render_final_plot(result, final_plot)
    render_gif(result, gif_path)


if __name__ == "__main__":
    main()
