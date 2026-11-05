#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax[cpu]",
#   "matplotlib",
# ]
# ///

from __future__ import annotations

"""Generate the 1D flow-matching figures (independent coupling, exact averaged field)."""

from pathlib import Path

import jax.numpy as jnp
from jax import random
from matplotlib.axes import Axes

from figure_io import apply_plot_style, assert_all_finite, generated_path, save_figure, plt, to_plot_array
from flow_core import (
    DEFAULT_TARGET,
    integrate_field,
    marginal_velocity,
    normal_pdf,
    sample_independent_pairs,
    straight_path,
    target_pdf,
)

FLOW_SEED = 20260402
NUM_PAIRS = 24
NUM_TRAJECTORIES = 13
TRAJECTORY_START_MIN = -2.4
TRAJECTORY_START_MAX = 2.4
ODE_STEPS = 200
NUM_PARTICLES = 20000
HISTOGRAM_SNAPSHOT_TIMES = [0.0, 0.35, 0.7, 1.0]
HISTOGRAM_BINS = 60

FIELD_TIME_GRID_SIZE = 160
FIELD_STATE_GRID_SIZE = 320
STATE_LIMIT = 4.2
TIME_MIN = -0.17
TIME_MAX = 1.17
BAND_WIDTH = 0.14
SOURCE_COLOR = "#667085"
TARGET_COLOR = "#27864A"
PAIR_COLOR = "#667085"
HIGHLIGHT_COLORS = ("#D97706", "#3F7CAC")
PATH_COLOR = "black"
SUPPORT_SIZE = 800
FIGSIZE = (9.0, 5.2)


def draw_density_bands(ax: Axes) -> None:
    """Draw the noise density `p_0` on the left and the data density `p_1` on the right."""
    support = jnp.linspace(-STATE_LIMIT, STATE_LIMIT, SUPPORT_SIZE, dtype=jnp.float64)
    source = normal_pdf(support, 0.0, 1.0)
    target = target_pdf(support)
    left = -BAND_WIDTH * source / jnp.max(source)
    right = 1.0 + BAND_WIDTH * target / jnp.max(target)
    ax.fill_betweenx(to_plot_array(support), 0.0, to_plot_array(left), color=SOURCE_COLOR, alpha=0.85)
    ax.fill_betweenx(to_plot_array(support), 1.0, to_plot_array(right), color=TARGET_COLOR, alpha=0.85)
    ax.text(-BAND_WIDTH / 2, STATE_LIMIT - 0.15, r"$p_0$ (noise)", ha="center", va="top", fontsize=9)
    ax.text(1.0 + BAND_WIDTH / 2, STATE_LIMIT - 0.15, r"$p_1$ (data)", ha="center", va="top", fontsize=9)


def style_axes(ax: Axes) -> None:
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="black", linewidth=0.8)
    ax.set_xlim(TIME_MIN, TIME_MAX)
    ax.set_ylim(-STATE_LIMIT, STATE_LIMIT)
    ax.set_xlabel(r"time $t$ (0 = noise, 1 = data)")
    ax.set_ylabel(r"position $x$")


def first_crossing_pair(x0: jnp.ndarray, x1: jnp.ndarray) -> tuple[int, int]:
    """Return indices of two pairs whose straight lines cross inside (0, 1)."""
    count = x0.shape[0]
    for first in range(count):
        for second in range(first + 1, count):
            if (x0[first] - x0[second]) * (x1[first] - x1[second]) < 0.0:
                return first, second
    raise AssertionError("no crossing pair found; increase the number of pairs")


def render_independent_pairs(output_path: Path) -> None:
    """Straight pair paths under independent coupling: they cross."""
    x0, x1 = sample_independent_pairs(random.PRNGKey(FLOW_SEED), NUM_PAIRS)
    times = jnp.linspace(0.0, 1.0, 2, dtype=jnp.float64)
    paths = straight_path(x0, x1, times)
    first, second = first_crossing_pair(x0, x1)
    crossing_time = float((x0[second] - x0[first]) / ((x0[second] - x0[first]) - (x1[second] - x1[first])))
    crossing_x = float((1.0 - crossing_time) * x0[first] + crossing_time * x1[first])
    assert 0.0 < crossing_time < 1.0
    assert_all_finite(paths)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_density_bands(ax)
    for index in range(NUM_PAIRS):
        ax.plot(to_plot_array(times), to_plot_array(paths[:, index]), color=PAIR_COLOR, linewidth=1.0, alpha=0.7)
    for index, color in zip((first, second), HIGHLIGHT_COLORS, strict=True):
        ax.plot(to_plot_array(times), to_plot_array(paths[:, index]), color=color, linewidth=2.6)
    ax.scatter([crossing_time], [crossing_x], s=70, color="black", zorder=5)
    ax.annotate(
        "two pairs cross:\ndifferent velocities\nat the same point",
        xy=(crossing_time, crossing_x),
        xytext=(0.3, 3.3),
        arrowprops={"arrowstyle": "->"},
        fontsize=10,
    )
    style_axes(ax)
    ax.set_title(r"Independent coupling: each pair moves on a straight line $x_t=(1-t)x_0+t x_1$")
    save_figure(fig, output_path)


def render_marginal_field(output_path: Path) -> None:
    """Exact averaged field with the paths it generates: these do not cross."""
    times = jnp.linspace(0.0, 1.0, FIELD_TIME_GRID_SIZE, dtype=jnp.float64)
    states = jnp.linspace(-STATE_LIMIT, STATE_LIMIT, FIELD_STATE_GRID_SIZE, dtype=jnp.float64)
    field = jnp.stack([marginal_velocity(states, float(t)) for t in times], axis=1)
    starts = jnp.linspace(TRAJECTORY_START_MIN, TRAJECTORY_START_MAX, NUM_TRAJECTORIES, dtype=jnp.float64)
    trajectories = integrate_field(starts, ODE_STEPS)
    assert_all_finite(field, trajectories)
    assert bool(jnp.all(jnp.diff(trajectories, axis=1) > 0.0)), "ODE paths must keep their order (no crossing)"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    filled = ax.contourf(
        to_plot_array(times), to_plot_array(states), to_plot_array(field), levels=18, cmap="coolwarm"
    )
    draw_density_bands(ax)
    path_times = jnp.linspace(0.0, 1.0, ODE_STEPS + 1, dtype=jnp.float64)
    for index in range(NUM_TRAJECTORIES):
        ax.plot(to_plot_array(path_times), to_plot_array(trajectories[:, index]), color=PATH_COLOR, linewidth=1.2)
    style_axes(ax)
    ax.set_title(r"Averaged field $u_t(x)$ (color) and the paths it generates (black)")
    cbar = fig.colorbar(filled, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"velocity $u_t(x)$ (red: moving up, blue: moving down)")
    save_figure(fig, output_path)


def render_density_evolution(output_path: Path) -> None:
    """Histogram of particles carried by the averaged field, against the exact density."""
    x0 = random.normal(random.PRNGKey(FLOW_SEED + 1), (NUM_PARTICLES,), dtype=jnp.float64)
    trajectory = integrate_field(x0, ODE_STEPS)
    support = jnp.linspace(-STATE_LIMIT, STATE_LIMIT, SUPPORT_SIZE, dtype=jnp.float64)

    fig, axes = plt.subplots(1, len(HISTOGRAM_SNAPSHOT_TIMES), figsize=(13.0, 2.8), sharey=True)
    for ax, snapshot in zip(axes, HISTOGRAM_SNAPSHOT_TIMES, strict=True):
        step = round(snapshot * ODE_STEPS)
        variance = (1.0 - snapshot) ** 2 + snapshot**2 * DEFAULT_TARGET.stds**2
        exact = jnp.zeros_like(support)
        for weight, mean, var in zip(DEFAULT_TARGET.weights, snapshot * DEFAULT_TARGET.means, variance, strict=True):
            exact += weight * normal_pdf(support, mean, var)
        ax.hist(
            to_plot_array(trajectory[step]), bins=HISTOGRAM_BINS, range=(-STATE_LIMIT, STATE_LIMIT),
            density=True, color="#3F7CAC", alpha=0.65,
        )
        ax.plot(to_plot_array(support), to_plot_array(exact), color="black", linewidth=1.2)
        ax.set_title(fr"$t={snapshot:g}$")
        ax.set_xlabel(r"position $x$")
    axes[0].set_ylabel("density")
    save_figure(fig, output_path)


def main() -> None:
    """Generate all flow-matching figures used by the deck."""
    apply_plot_style()
    render_independent_pairs(generated_path("flow_matching_independent_pairs.png"))
    render_marginal_field(generated_path("flow_matching_marginal_field.png"))
    render_density_evolution(generated_path("flow_matching_density_evolution.png"))


if __name__ == "__main__":
    main()
