#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax[cpu]",
#   "matplotlib",
# ]
# ///

from __future__ import annotations

"""Generate the 1D flow-matching transport visualization for the deck."""

from pathlib import Path

import jax.numpy as jnp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import apply_plot_style, assert_all_finite, generated_path, save_figure, plt, to_plot_array
from flow_core import build_monotone_transport, transport_characteristics, velocity_field_grid

TRANSPORT_TIME_MIN = 0.03
TRANSPORT_TIME_MAX = 0.97
TRANSPORT_TIME_GRID_SIZE = 150
TRANSPORT_STATE_MIN = -4.4
TRANSPORT_STATE_MAX = 4.4
TRANSPORT_STATE_GRID_SIZE = 380

PDF_BAND_HALF_WIDTH = 0.13
LEFT_PDF_EDGE = -0.13
RIGHT_PDF_EDGE = 1.13
TRANSPORT_XMIN = -0.16
TRANSPORT_XMAX = 1.16
TRANSPORT_YMIN = -4.4
TRANSPORT_YMAX = 4.4

CHARACTERISTIC_QUANTILES = [0.04, 0.18, 0.34, 0.50, 0.68, 0.84, 0.96]
TRAJECTORY_TIME_STEPS = 120
ENDPOINT_POINT_QUANTILES = jnp.linspace(0.05, 0.95, 17, dtype=jnp.float64)
TRANSPORT_FIGSIZE = (11.8, 6.2)
CONTOUR_LEVELS = 18
LINE_LEVELS = 10
PDF_LABEL_Y = 3.95
LABEL_FONT_SIZE = 10
LEGEND_LOCATION = "lower right"
SOURCE_POINT_SIZE = 22.0
TARGET_POINT_SIZE = 22.0
SOURCE_POINT_COLOR = "#4f5d75"
TARGET_POINT_COLOR = "#0b7189"
TRAJECTORY_COLOR = "black"
TRAJECTORY_LINEWIDTH = 1.0
TRAJECTORY_ALPHA = 0.8
SOURCE_BAND_COLOR = "#6c757d"
TARGET_BAND_COLOR = "#0b7189"
SOURCE_BAND_ALPHA = 0.88
TARGET_BAND_ALPHA = 0.88
CONTOUR_CMAP = "coolwarm"
CONTOUR_LINE_COLOR = "white"
CONTOUR_LINE_WIDTH = 0.6
CONTOUR_LINE_ALPHA = 0.7
ENDPOINT_LINE_COLOR = "black"
ENDPOINT_LINE_WIDTH = 0.9
ENDPOINT_LINE_ALPHA = 0.7


def render_transport_field(output_path: Path) -> None:
    """Render a 1D source-to-target transport field over `(t, x)`.

    The filled contour shows the coupling-induced velocity field, the side bands
    show the endpoint densities, and the black curves show characteristic
    interpolation paths for selected quantiles.
    """
    transport = build_monotone_transport()
    time_grid = jnp.linspace(TRANSPORT_TIME_MIN, TRANSPORT_TIME_MAX, TRANSPORT_TIME_GRID_SIZE, dtype=jnp.float64)
    state_grid = jnp.linspace(TRANSPORT_STATE_MIN, TRANSPORT_STATE_MAX, TRANSPORT_STATE_GRID_SIZE, dtype=jnp.float64)
    field = velocity_field_grid(transport, time_grid=time_grid, state_grid=state_grid)

    source_peak = float(jnp.max(transport.source_pdf))
    target_peak = float(jnp.max(transport.target_pdf))
    source_band = LEFT_PDF_EDGE + PDF_BAND_HALF_WIDTH * transport.source_pdf / source_peak
    target_band = RIGHT_PDF_EDGE - PDF_BAND_HALF_WIDTH * transport.target_pdf / target_peak

    characteristic_quantiles = jnp.array(CHARACTERISTIC_QUANTILES, dtype=jnp.float64)
    source_chars, target_chars = transport_characteristics(transport, characteristic_quantiles)
    trajectory_times = jnp.linspace(0.0, 1.0, TRAJECTORY_TIME_STEPS, dtype=jnp.float64)
    source_points, target_points = transport_characteristics(transport, ENDPOINT_POINT_QUANTILES)

    fig, ax = plt.subplots(figsize=TRANSPORT_FIGSIZE)
    filled = ax.contourf(
        to_plot_array(time_grid),
        to_plot_array(state_grid),
        to_plot_array(field.T),
        levels=CONTOUR_LEVELS,
        cmap=CONTOUR_CMAP,
    )
    ax.contour(
        to_plot_array(time_grid),
        to_plot_array(state_grid),
        to_plot_array(field.T),
        levels=LINE_LEVELS,
        colors=CONTOUR_LINE_COLOR,
        linewidths=CONTOUR_LINE_WIDTH,
        alpha=CONTOUR_LINE_ALPHA,
    )

    ax.fill_betweenx(
        to_plot_array(transport.source_support),
        LEFT_PDF_EDGE,
        to_plot_array(source_band),
        color=SOURCE_BAND_COLOR,
        alpha=SOURCE_BAND_ALPHA,
    )
    ax.fill_betweenx(
        to_plot_array(transport.source_support),
        to_plot_array(target_band),
        RIGHT_PDF_EDGE,
        color=TARGET_BAND_COLOR,
        alpha=TARGET_BAND_ALPHA,
    )

    for source_value, target_value in zip(source_chars, target_chars, strict=True):
        trajectory = (1.0 - trajectory_times) * source_value + trajectory_times * target_value
        ax.plot(
            to_plot_array(trajectory_times),
            to_plot_array(trajectory),
            color=TRAJECTORY_COLOR,
            linewidth=TRAJECTORY_LINEWIDTH,
            alpha=TRAJECTORY_ALPHA,
        )

    ax.scatter(
        to_plot_array(jnp.zeros_like(source_points)),
        to_plot_array(source_points),
        s=SOURCE_POINT_SIZE,
        color=SOURCE_POINT_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax.scatter(
        to_plot_array(jnp.ones_like(target_points)),
        to_plot_array(target_points),
        s=TARGET_POINT_SIZE,
        color=TARGET_POINT_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )

    finite_field_mask = jnp.isfinite(field)
    assert bool(jnp.any(finite_field_mask))
    assert_all_finite(field[finite_field_mask], source_band, target_band, source_chars, target_chars, source_points, target_points)
    ax.axvline(0.0, color=ENDPOINT_LINE_COLOR, linewidth=ENDPOINT_LINE_WIDTH, alpha=ENDPOINT_LINE_ALPHA)
    ax.axvline(1.0, color=ENDPOINT_LINE_COLOR, linewidth=ENDPOINT_LINE_WIDTH, alpha=ENDPOINT_LINE_ALPHA)
    ax.set_xlim(TRANSPORT_XMIN, TRANSPORT_XMAX)
    ax.set_ylim(TRANSPORT_YMIN, TRANSPORT_YMAX)
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"state $x$")
    ax.set_title(r"1D flow-matching transport field $u(x,t)$ between a Gaussian source and a bimodal target")
    ax.text(LEFT_PDF_EDGE + 0.015, PDF_LABEL_Y, r"$p_0(x)$", fontsize=LABEL_FONT_SIZE, ha="center", va="top")
    ax.text(RIGHT_PDF_EDGE - 0.015, PDF_LABEL_Y, r"$p_1(x)$", fontsize=LABEL_FONT_SIZE, ha="center", va="top")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=SOURCE_POINT_COLOR, markeredgecolor="white", label=r"source samples $x_0$"),
        Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=TARGET_POINT_COLOR, markeredgecolor="white", label=r"target samples $x_1$"),
        Line2D([0], [0], color=TRAJECTORY_COLOR, linewidth=TRAJECTORY_LINEWIDTH, label=r"straight paths $x_t=(1-t)x_0+t x_1$"),
        Patch(facecolor=SOURCE_BAND_COLOR, alpha=SOURCE_BAND_ALPHA, label=r"source density $p_0(x)$"),
        Patch(facecolor=TARGET_BAND_COLOR, alpha=TARGET_BAND_ALPHA, label=r"target density $p_1(x)$"),
    ]
    ax.legend(handles=legend_handles, loc=LEGEND_LOCATION, frameon=True)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(r"velocity field $u(x,t)$")
    save_figure(fig, output_path)


def main() -> None:
    """Generate all flow-related figures used by the deck."""
    apply_plot_style()
    render_transport_field(generated_path("flow_matching_1d_transport_field.png"))


if __name__ == "__main__":
    main()
