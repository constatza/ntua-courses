"""The toy sampler must reproduce the data distribution without ever seeing x_0."""

from __future__ import annotations

import jax.numpy as jnp
from jax import random

from diffusion_core import (
    build_linear_schedule,
    ddpm_reverse_sample,
    make_default_gmm,
    sample_gmm,
)

NUM_STEPS = 400
NUM_SAMPLES = 2000
NEAR_COMPONENT_RADIUS = 1.5
MIN_FRACTION_NEAR = 0.95
MEAN_TOLERANCE = 0.15


def test_ddpm_with_exact_score_recovers_mixture() -> None:
    mixture = make_default_gmm()
    schedule = build_linear_schedule(num_steps=NUM_STEPS, beta_start=2.5e-4, beta_end=5.0e-2)
    key_start, key_steps = random.split(random.PRNGKey(0))
    x_T = random.normal(key_start, (NUM_SAMPLES, 2), dtype=jnp.float64)
    samples = ddpm_reverse_sample(x_T, mixture, schedule, key_steps, keep_steps=[])[0]

    distances = jnp.linalg.norm(samples[:, None, :] - mixture.means[None], axis=-1)
    assert float(jnp.mean(distances.min(axis=1) < NEAR_COMPONENT_RADIUS)) > MIN_FRACTION_NEAR

    data = sample_gmm(mixture, NUM_SAMPLES, random.PRNGKey(1))
    assert float(jnp.max(jnp.abs(samples.mean(axis=0) - data.mean(axis=0)))) < MEAN_TOLERANCE
