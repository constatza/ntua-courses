#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax[cpu]",
#   "matplotlib",
# ]
# ///

from __future__ import annotations

"""Train the tiny 1D PINN and write the figures and the table used by the deck."""

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinn_core import Problem, TrainingRun, train

GENERATED_DIR = Path(__file__).resolve().parents[1] / "assets" / "generated"
NUM_STEPS = 5000
LEARNING_RATE = 1.0e-2
SEEDS = (0, 1, 2)
SNAPSHOT_STEPS = [0, 100, 500, NUM_STEPS]
FREQUENCIES = (1, 2, 3, 4)
SAVE_DPI = 200
EXACT_COLOR = "black"
CURVE_COLORS = ["#B5473C", "#D97706", "#3F7CAC", "#27864A"]


def render_training(run: TrainingRun, problem: Problem, output_path: Path) -> None:
    """Left: network output at four training steps; right: errors during training."""
    x = run.test_points
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 3.8))
    left.plot(x, problem.exact(x), color=EXACT_COLOR, linewidth=2.4, label=r"exact $\sin(\pi x)$")
    for step, color in zip(SNAPSHOT_STEPS, CURVE_COLORS, strict=True):
        left.plot(x, run.predictions[step], color=color, linewidth=1.6, label=f"step {step}")
    left.set_xlabel(r"$x$")
    left.set_ylabel(r"$u_\theta(x)$")
    left.legend(fontsize=8, loc="lower center")
    right.semilogy(run.steps, run.relative_error, color="#3F7CAC", label="error vs exact solution")
    right.semilogy(run.steps, run.residual_loss, color="#D97706", label="residual loss (what is trained)")
    right.set_xlabel("training step")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI)
    plt.close(fig)


def render_frequencies(runs: dict[int, TrainingRun], output_path: Path) -> None:
    """Final network output (seed 0) against the exact solution for each frequency."""
    fig, axes = plt.subplots(1, len(FREQUENCIES), figsize=(13.0, 2.9), sharey=True)
    for ax, frequency in zip(axes, FREQUENCIES, strict=True):
        run = runs[frequency]
        problem = Problem(frequency)
        x = run.test_points
        ax.plot(x, problem.exact(x), color=EXACT_COLOR, linewidth=2.0)
        ax.plot(x, run.predictions[NUM_STEPS], color="#B5473C", linewidth=1.4)
        ax.set_title(fr"$k={frequency}$: error {run.relative_error[-1]:.1e}")
        ax.set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$u(x)$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI)
    plt.close(fig)


def write_error_table(errors: dict[int, list[float]], output_path: Path) -> None:
    """LaTeX tabular: final relative error over the seeds, per frequency."""
    rows = [
        f"{frequency} & {min(values):.0e} & {max(values):.0e} \\\\"
        for frequency, values in errors.items()
    ]
    lines = [
        r"\begin{tabular}{ccc}",
        r"\hline",
        r"$k$ & smallest error & largest error \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    runs: dict[int, TrainingRun] = {}
    errors: dict[int, list[float]] = {}
    for frequency in FREQUENCIES:
        errors[frequency] = []
        for seed in SEEDS:
            run = train(Problem(frequency), seed, NUM_STEPS, LEARNING_RATE, SNAPSHOT_STEPS)
            errors[frequency].append(run.relative_error[-1])
            if seed == SEEDS[0]:
                runs[frequency] = run
    render_training(runs[1], Problem(1), GENERATED_DIR / "pinn_training.png")
    render_frequencies(runs, GENERATED_DIR / "pinn_frequencies.png")
    write_error_table(errors, GENERATED_DIR / "pinn_error_table.tex")
    assert all(jnp.isfinite(jnp.array(v)).all() for v in errors.values())


if __name__ == "__main__":
    main()
