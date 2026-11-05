# /// script
# dependencies = ["matplotlib", "numpy", "pillow"]
# ///
"""Augmentation figures for the geometric-deep-learning deck.

Two figures: label-preserving views of one cat photo, and transformations that
change the label, drawn with text so no extra image source is needed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image, ImageEnhance, ImageFilter

SEED = 20261029
FIGURE_DPI = 130
NAVY = "#10284b"
RED = "#b5473c"
GREEN = "#27864a"

EXPECTED_FILENAMES = ("augmentation-grid.png", "label-changing-transforms.png")

IMAGE_SIZE = 420
ROTATION_DEGREES = 15
NOISE_SIGMA = 13.0
BLUR_RADIUS = 3.0
COLOR_ENHANCE = 0.45
CROP_BOX = (45, 20, 365, 340)
FILL_COLOR = (225, 222, 219)
GLYPH_SIZE = 1.6

SOURCE_IMAGE = Path(__file__).resolve().parents[1] / "assets" / "source" / "cat.png"


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _augmentation_grid(path: Path, rng: np.random.Generator) -> None:
    image = Image.open(SOURCE_IMAGE).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    crop = image.crop(CROP_BOX).resize(image.size)
    noisy = np.asarray(image, dtype=np.float32) + rng.normal(0, NOISE_SIGMA, (IMAGE_SIZE, IMAGE_SIZE, 3))
    variants = (
        ("original", image),
        (rf"rotate ${ROTATION_DEGREES}^\circ$", image.rotate(ROTATION_DEGREES, resample=Image.Resampling.BICUBIC, fillcolor=FILL_COLOR)),
        ("horizontal flip", image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
        ("crop + resize", crop),
        ("color shift", ImageEnhance.Color(image).enhance(COLOR_ENHANCE)),
        ("sensor noise", Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))),
        ("mild blur", image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))),
    )
    figure, axes = plt.subplots(2, 4, figsize=(11.2, 6.0))
    for axis, (title, variant) in zip(axes.flat, variants, strict=False):
        axis.imshow(variant)
        axis.set_title(title, color=NAVY, fontsize=12, weight="bold")
        axis.axis("off")
    axes.flat[-1].axis("off")
    figure.subplots_adjust(wspace=0.04, hspace=0.16)
    _save(figure, path)


def _glyph(axis: plt.Axes, glyph: str, color: str, *, mirror_x: bool, mirror_y: bool) -> None:
    """Draw ``glyph`` centred at the origin, optionally reflected through an axis.

    A text path is reflected exactly, whereas text rotation depends on font metrics.
    """
    outline = TextPath((0, 0), glyph, size=GLYPH_SIZE)
    box = outline.get_extents()
    centre = Affine2D().translate(-(box.x0 + box.x1) / 2, -(box.y0 + box.y1) / 2)
    reflect = Affine2D().scale(-1.0 if mirror_x else 1.0, -1.0 if mirror_y else 1.0)
    axis.add_patch(PathPatch(outline, color=color, transform=centre + reflect + axis.transData))


def _label_changing(path: Path) -> None:
    """Glyph pairs where the transformation turns one valid label into another."""
    cases = (
        ("6", "rotate $180^\\circ$", "the digit becomes a 9", True, True),
        ("b", "horizontal flip", "the letter becomes a d", True, False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for (before, after), (glyph, transform, outcome, mirror_x, mirror_y) in zip(axes, cases, strict=True):
        for axis in (before, after):
            axis.set(xlim=(-GLYPH_SIZE, GLYPH_SIZE), ylim=(-GLYPH_SIZE, GLYPH_SIZE))
            axis.set_aspect("equal")
            axis.axis("off")
        _glyph(before, glyph, NAVY, mirror_x=False, mirror_y=False)
        before.set_title("original", color=NAVY, fontsize=12)
        _glyph(after, glyph, RED, mirror_x=mirror_x, mirror_y=mirror_y)
        after.set_title(f"{transform}: {outcome}", color=RED, fontsize=12)
    figure.subplots_adjust(wspace=0.1, hspace=0.35)
    _save(figure, path)


def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_directory / name for name in EXPECTED_FILENAMES)
    rng = np.random.default_rng(SEED)
    _augmentation_grid(outputs[0], rng)
    _label_changing(outputs[1])
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    return outputs


def main() -> None:
    generate_figures(Path(__file__).resolve().parents[1] / "assets" / "generated")


if __name__ == "__main__":
    main()
