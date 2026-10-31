from __future__ import annotations

"""Output paths, plot defaults and array helpers for the flow-matching figures."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
GENERATED_DIR = ROOT_DIR / "assets" / "generated"

DEFAULT_FIGURE_DPI = 150
DEFAULT_AXES_TITLE_SIZE = 12
DEFAULT_AXES_LABEL_SIZE = 11
DEFAULT_TICK_LABEL_SIZE = 9
DEFAULT_LEGEND_SIZE = 9
DEFAULT_FONT_SIZE = 10
DEFAULT_SAVE_DPI = 220


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def generated_path(filename: str) -> Path:
    """Return the output path for a generated asset under `assets/generated/`."""
    return ensure_directory(GENERATED_DIR) / filename


def apply_plot_style() -> None:
    """Install one consistent Matplotlib style for all generated figures."""
    plt.rcParams.update(
        {
            "figure.dpi": DEFAULT_FIGURE_DPI,
            "axes.titlesize": DEFAULT_AXES_TITLE_SIZE,
            "axes.labelsize": DEFAULT_AXES_LABEL_SIZE,
            "xtick.labelsize": DEFAULT_TICK_LABEL_SIZE,
            "ytick.labelsize": DEFAULT_TICK_LABEL_SIZE,
            "legend.fontsize": DEFAULT_LEGEND_SIZE,
            "font.size": DEFAULT_FONT_SIZE,
            "axes.grid": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Persist a figure with a fixed export DPI and close it afterwards."""
    ensure_directory(path.parent)
    fig.savefig(path, dpi=DEFAULT_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def assert_all_finite(*arrays: jax.Array) -> None:
    """Assert that every supplied array contains only finite values."""
    for array in arrays:
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(array))))


def to_plot_array(array: jax.Array):
    """Move a JAX array to host memory for Matplotlib consumption."""
    return jax.device_get(array)
