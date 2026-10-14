from __future__ import annotations

"""Shared utilities for deterministic scientific figure generation.

The helpers in this module are intentionally lightweight:
- filesystem and output helpers,
- plotting defaults,
- deterministic reference-image loading,
- small display conversions used by the diffusion figures.
"""

from pathlib import Path
from typing import Iterable

import jax
from jax import random
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

CAT_IMAGE_DOWNSAMPLE_STEP = 3

DISPLAY_OFFSET = 1.0
DISPLAY_SCALE = 0.5
DISPLAY_MIN = 0.0
DISPLAY_MAX = 1.0


def make_rng(seed: int) -> jax.Array:
    """Create a deterministic JAX PRNG key from a single integer seed."""
    return random.PRNGKey(seed)


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


def normalize_to_unit_interval(image: jax.Array) -> jax.Array:
    """Affinely rescale an array so its values lie in `[0, 1]`."""
    image = jnp.asarray(image, dtype=jnp.float64)
    image_min = float(jnp.min(image))
    image_max = float(jnp.max(image))
    assert image_min == image_min and image_max == image_max
    assert image_max > image_min
    return (image - image_min) / (image_max - image_min)


def rgb_to_grayscale(image: jax.Array) -> jax.Array:
    """Convert an RGB image to grayscale with standard luminance weights."""
    rgb = jnp.asarray(image[..., :3], dtype=jnp.float64)
    if float(jnp.max(rgb)) > 1.0:
        rgb = rgb / 255.0
    grayscale = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return normalize_to_unit_interval(grayscale)


def grayscale_to_model_space(image: jax.Array) -> jax.Array:
    """Map grayscale values from `[0, 1]` to the diffusion model space `[-1, 1]`."""
    image = jnp.asarray(image, dtype=jnp.float64)
    return image * 2.0 - 1.0


def load_reference_cat_image(downsample_step: int = CAT_IMAGE_DOWNSAMPLE_STEP) -> jax.Array:
    """Load a built-in cat image and map it into model space.

    The source is `skimage.data.chelsea()`, a standard offline sample image
    packaged with scikit-image. The returned array is grayscale and scaled to
    `[-1, 1]` so the diffusion process can operate on it directly.
    """
    from skimage import data

    cat_rgb = jnp.asarray(data.chelsea(), dtype=jnp.float64)
    cat_gray = rgb_to_grayscale(cat_rgb)
    cat_small = cat_gray[::downsample_step, ::downsample_step]
    return grayscale_to_model_space(cat_small)


def state_to_display_grayscale(state: jax.Array) -> jax.Array:
    """Map an internal diffusion state from `[-1, 1]` to display grayscale."""
    state = jnp.asarray(state, dtype=jnp.float64)
    return jnp.clip(DISPLAY_SCALE * (state + DISPLAY_OFFSET), DISPLAY_MIN, DISPLAY_MAX)


def select_indices(num_steps: int, chosen: Iterable[int]) -> list[int]:
    """Validate and sort a chosen set of representative timesteps."""
    indices = sorted({int(step) for step in chosen})
    assert indices[0] == 0
    assert indices[-1] == num_steps
    assert all(0 <= idx <= num_steps for idx in indices)
    return indices


def assert_all_finite(*arrays: jax.Array) -> None:
    """Assert that every supplied array contains only finite values."""
    for array in arrays:
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(array))))


def image_extent(image: jax.Array) -> tuple[float, float, float, float]:
    """Return a Matplotlib image extent matching the array shape."""
    height, width = image.shape
    return (0.0, float(width), float(height), 0.0)


def to_plot_array(array: jax.Array):
    """Move a JAX array to host memory for Matplotlib consumption."""
    return jax.device_get(array)
