#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax[cpu]",
#   "matplotlib",
#   "scikit-image",
# ]
# ///

from __future__ import annotations

"""Generate the diffusion-related figures for the presentation."""

from pathlib import Path

import jax.numpy as jnp

from common import (
    apply_plot_style,
    assert_all_finite,
    generated_path,
    image_extent,
    load_reference_cat_image,
    make_rng,
    save_figure,
    select_indices,
    state_to_display_grayscale,
    plt,
    to_plot_array,
)
from diffusion_core import (
    DiffusionSchedule,
    GaussianMixture2D,
    build_linear_schedule,
    direct_noisy_sample,
    ddpm_reverse_sample,
    make_default_gmm,
    noisy_gmm_parameters,
    oracle_reverse_chain,
    sample_gmm,
    simulate_forward_chain,
    smoothed_gmm_density,
    smoothed_gmm_score,
)
from jax import random


DIFFUSION_SEED = 20260402
NUM_STEPS = 1000
BETA_START = 1.0e-4
BETA_END = 2.0e-2
CHAIN_STEP_VALUES = [0, 60, 180, 360, 560, 780, 1000]
CHAIN_STEPS = select_indices(NUM_STEPS, CHAIN_STEP_VALUES)

SCORE_NOISE_STD = 0.85
SCORE_DENSE_MIN = -4.5
SCORE_DENSE_MAX = 4.5
SCORE_DENSE_GRID_SIZE = 220
SCORE_COARSE_MIN = -4.0
SCORE_COARSE_MAX = 4.0
SCORE_COARSE_GRID_SIZE = 17
SCORE_FIGSIZE = (8.6, 6.8)
SCORE_CONTOUR_LEVELS = 18
SCORE_LINE_LEVELS = 10
SCORE_VECTOR_SCALE = 24.0
SCORE_VECTOR_WIDTH = 0.004
SCORE_MARKER_SIZE = 34
SCORE_VECTOR_COLOR = "#8c1c13"
SCORE_CONTOUR_CMAP = "YlGnBu"
SCORE_CONTOUR_COLOR = "#24415a"
SCORE_CONTOUR_LINEWIDTH = 0.8
SCORE_CONTOUR_ALPHA = 0.65
SCORE_QUIVER_KEY_X = 0.83
SCORE_QUIVER_KEY_Y = 1.04
SCORE_QUIVER_KEY_LENGTH = 1.5

TOY_NUM_SAMPLES = 1500
TOY_STEPS = [0, 150, 350, 600, 1000]
TOY_LIMIT = 4.5
TOY_GRID_SIZE = 120
TOY_FIGSIZE = (15.6, 3.4)
TOY_POINT_SIZE = 4.0
TOY_POINT_COLOR = "#3F7CAC"
TOY_CONTOUR_LEVELS = 6
TOY_CONTOUR_COLOR = "#667085"
TOY_CONTOUR_ALPHA = 0.55

CHAIN_FIGSIZE = (15.6, 2.0)
CHAIN_VMIN = 0.0
CHAIN_VMAX = 1.0


def render_score_field(output_path: Path) -> None:
    """Render an exact score field for a smoothed 2D Gaussian mixture."""
    mixture = make_default_gmm()
    noise_variance = SCORE_NOISE_STD**2

    dense_axis = jnp.linspace(SCORE_DENSE_MIN, SCORE_DENSE_MAX, SCORE_DENSE_GRID_SIZE, dtype=jnp.float64)
    dense_x, dense_y = jnp.meshgrid(dense_axis, dense_axis)
    dense_points = jnp.stack([dense_x, dense_y], axis=-1)
    density = smoothed_gmm_density(dense_points, mixture, noise_variance=noise_variance)

    coarse_axis = jnp.linspace(SCORE_COARSE_MIN, SCORE_COARSE_MAX, SCORE_COARSE_GRID_SIZE, dtype=jnp.float64)
    coarse_x, coarse_y = jnp.meshgrid(coarse_axis, coarse_axis)
    coarse_points = jnp.stack([coarse_x, coarse_y], axis=-1)
    score = smoothed_gmm_score(coarse_points, mixture, noise_variance=noise_variance)
    assert_all_finite(density, score)

    fig, ax = plt.subplots(figsize=SCORE_FIGSIZE)
    filled = ax.contourf(
        to_plot_array(dense_x),
        to_plot_array(dense_y),
        to_plot_array(density),
        levels=SCORE_CONTOUR_LEVELS,
        cmap=SCORE_CONTOUR_CMAP,
    )
    ax.contour(
        to_plot_array(dense_x),
        to_plot_array(dense_y),
        to_plot_array(density),
        levels=SCORE_LINE_LEVELS,
        colors=SCORE_CONTOUR_COLOR,
        linewidths=SCORE_CONTOUR_LINEWIDTH,
        alpha=SCORE_CONTOUR_ALPHA,
    )
    score_quiver = ax.quiver(
        to_plot_array(coarse_x),
        to_plot_array(coarse_y),
        to_plot_array(score[..., 0]),
        to_plot_array(score[..., 1]),
        color=SCORE_VECTOR_COLOR,
        pivot="mid",
        scale=SCORE_VECTOR_SCALE,
        width=SCORE_VECTOR_WIDTH,
    )
    ax.quiverkey(
        score_quiver,
        SCORE_QUIVER_KEY_X,
        SCORE_QUIVER_KEY_Y,
        SCORE_QUIVER_KEY_LENGTH,
        r"score vector $\nabla_x \log p_t(x)$",
        labelpos="E",
        coordinates="axes",
    )
    ax.scatter(
        to_plot_array(mixture.means[:, 0]),
        to_plot_array(mixture.means[:, 1]),
        s=SCORE_MARKER_SIZE,
        color="white",
        edgecolor="black",
        zorder=3,
    )
    ax.set_title(r"Smoothed density $p_t(x)$ and exact score field $\nabla_x \log p_t(x)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal")
    cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"density value $p_t(x)$")
    save_figure(fig, output_path)


def render_forward_chain(output_path: Path) -> list:
    """Render representative states of the forward diffusion chain."""
    key = make_rng(DIFFUSION_SEED)
    x0 = load_reference_cat_image()
    schedule = build_linear_schedule(num_steps=NUM_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    forward_states = simulate_forward_chain(x0, schedule, key)

    fig, axes = plt.subplots(1, len(CHAIN_STEPS), figsize=CHAIN_FIGSIZE)
    for ax, timestep in zip(axes, CHAIN_STEPS, strict=True):
        display_state = state_to_display_grayscale(forward_states[timestep])
        ax.imshow(to_plot_array(display_state), cmap="gray", vmin=CHAIN_VMIN, vmax=CHAIN_VMAX, extent=image_extent(x0))
        ax.set_title(fr"$t={timestep}$")
        ax.axis("off")

    save_figure(fig, output_path)
    return forward_states


def render_reverse_chain(output_path: Path) -> None:
    """Render representative states of the oracle posterior-mean reverse chain."""
    key = make_rng(DIFFUSION_SEED)
    x0 = load_reference_cat_image()
    schedule = build_linear_schedule(num_steps=NUM_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    forward_states = simulate_forward_chain(x0, schedule, key)
    reverse_states = oracle_reverse_chain(x0=x0, x_T=forward_states[-1], schedule=schedule)
    assert len(reverse_states) == len(forward_states)

    fig, axes = plt.subplots(1, len(CHAIN_STEPS), figsize=CHAIN_FIGSIZE)
    for ax, timestep in zip(axes, reversed(CHAIN_STEPS), strict=True):
        display_state = state_to_display_grayscale(reverse_states[timestep])
        ax.imshow(to_plot_array(display_state), cmap="gray", vmin=CHAIN_VMIN, vmax=CHAIN_VMAX, extent=image_extent(x0))
        ax.set_title(fr"$t={timestep}$")
        ax.axis("off")

    save_figure(fig, output_path)


def render_particle_row(
    states: dict[int, jnp.ndarray],
    steps: list[int],
    mixture: GaussianMixture2D,
    schedule: DiffusionSchedule,
    output_path: Path,
) -> None:
    """Draw 2-D particles at the given steps over the contours of the exact p_t."""
    axis = jnp.linspace(-TOY_LIMIT, TOY_LIMIT, TOY_GRID_SIZE, dtype=jnp.float64)
    grid_x, grid_y = jnp.meshgrid(axis, axis)
    grid = jnp.stack([grid_x, grid_y], axis=-1)

    fig, axes = plt.subplots(1, len(steps), figsize=TOY_FIGSIZE)
    for ax, timestep in zip(axes, steps, strict=True):
        noisy = noisy_gmm_parameters(mixture, schedule, timestep)
        density = smoothed_gmm_density(grid, noisy, noise_variance=0.0)
        ax.contour(
            to_plot_array(grid_x),
            to_plot_array(grid_y),
            to_plot_array(density),
            levels=TOY_CONTOUR_LEVELS,
            colors=TOY_CONTOUR_COLOR,
            alpha=TOY_CONTOUR_ALPHA,
            linewidths=0.8,
        )
        points = to_plot_array(states[timestep])
        ax.scatter(points[:, 0], points[:, 1], s=TOY_POINT_SIZE, color=TOY_POINT_COLOR)
        ax.set_xlim(-TOY_LIMIT, TOY_LIMIT)
        ax.set_ylim(-TOY_LIMIT, TOY_LIMIT)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(fr"$t={timestep}$")
    save_figure(fig, output_path)


def render_toy_forward(output_path: Path) -> None:
    """Noise 2-D data samples directly with the closed form q(x_t | x_0)."""
    mixture = make_default_gmm()
    schedule = build_linear_schedule(num_steps=NUM_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    key_data, key_noise = random.split(make_rng(DIFFUSION_SEED))
    x0 = sample_gmm(mixture, TOY_NUM_SAMPLES, key_data)
    noise = random.normal(key_noise, x0.shape, dtype=x0.dtype)
    states = {t: direct_noisy_sample(x0, schedule, t, noise) for t in TOY_STEPS}
    render_particle_row(states, TOY_STEPS, mixture, schedule, output_path)


def render_toy_reverse(output_path: Path) -> None:
    """Run the real DDPM sampler (exact score) from pure noise on the same 2-D data."""
    mixture = make_default_gmm()
    schedule = build_linear_schedule(num_steps=NUM_STEPS, beta_start=BETA_START, beta_end=BETA_END)
    key_start, key_steps = random.split(make_rng(DIFFUSION_SEED + 1))
    x_T = random.normal(key_start, (TOY_NUM_SAMPLES, 2), dtype=jnp.float64)
    states = ddpm_reverse_sample(x_T, mixture, schedule, key_steps, keep_steps=TOY_STEPS)
    render_particle_row(states, list(reversed(TOY_STEPS)), mixture, schedule, output_path)


def main() -> None:
    """Generate all diffusion figures used by the deck."""
    apply_plot_style()
    render_score_field(generated_path("score_field_gaussian_mixture.png"))
    render_forward_chain(generated_path("diffusion_forward_chain_cat.png"))
    render_reverse_chain(generated_path("diffusion_reverse_chain_cat_oracle.png"))
    render_toy_forward(generated_path("diffusion_toy_forward.png"))
    render_toy_reverse(generated_path("diffusion_toy_reverse.png"))


if __name__ == "__main__":
    main()
