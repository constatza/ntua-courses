"""Integrating the exact averaged field must transport N(0,1) onto the target mixture."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from flow_core import (
    DEFAULT_TARGET,
    integrate_field,
    marginal_velocity,
    sample_independent_pairs,
    sample_target,
)

NUM_SAMPLES = 20000
NUM_STEPS = 100
MOMENT_TOLERANCE = 0.05


def test_flow_pushes_gaussian_onto_target() -> None:
    x0 = random.normal(random.PRNGKey(0), (NUM_SAMPLES,), dtype=jnp.float64)
    x1 = integrate_field(x0, NUM_STEPS)[-1]
    reference = sample_target(random.PRNGKey(1), NUM_SAMPLES)
    assert abs(float(x1.mean() - reference.mean())) < MOMENT_TOLERANCE
    assert abs(float(x1.std() - reference.std())) < MOMENT_TOLERANCE


def test_field_matches_pairwise_average() -> None:
    """Binned average of x_1 - x_0 over sampled pairs equals the closed-form field."""
    t, half_width, x_center = 0.5, 0.05, 0.3
    x0, x1 = sample_independent_pairs(random.PRNGKey(2), 400000)
    xt = (1.0 - t) * x0 + t * x1
    mask = jnp.abs(xt - x_center) < half_width
    empirical = float(jnp.mean((x1 - x0)[mask]))
    exact = float(marginal_velocity(jnp.array([x_center]), t, DEFAULT_TARGET)[0])
    assert abs(empirical - exact) < MOMENT_TOLERANCE
