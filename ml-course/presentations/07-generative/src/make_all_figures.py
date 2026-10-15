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

"""Generate every figure needed by the diffusion-vs-flow deck."""

from make_diffusion_figures import main as make_diffusion_figures
from make_flow_figures import main as make_flow_figures


def main() -> None:
    """Run the diffusion and flow figure pipelines in sequence."""
    make_diffusion_figures()
    make_flow_figures()


if __name__ == "__main__":
    main()
