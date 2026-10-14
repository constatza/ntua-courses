from __future__ import annotations

"""Exact 1D transport objects for the flow-matching figure.

The figure uses a monotone quantile coupling between:
- a standard Gaussian source density `p_0`,
- a bimodal Gaussian-mixture target density `p_1`.

This gives a clean 1D setting where the interpolation paths and the induced
velocity field can be visualized without training any model.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv

jax.config.update("jax_enable_x64", True)

Array = jax.Array

DEFAULT_MIXTURE_WEIGHTS = jnp.array([0.56, 0.44], dtype=jnp.float64)
DEFAULT_MIXTURE_MEANS = jnp.array([-1.6, 1.9], dtype=jnp.float64)
DEFAULT_MIXTURE_STDS = jnp.array([0.42, 0.68], dtype=jnp.float64)
DEFAULT_PROBABILITY_MIN = 1.0e-4
DEFAULT_PROBABILITY_MAX = 1.0 - DEFAULT_PROBABILITY_MIN
DEFAULT_SUPPORT_MIN = -4.8
DEFAULT_SUPPORT_MAX = 4.8
DEFAULT_SUPPORT_SIZE = 6000
DEFAULT_NUM_QUANTILES = 2600


@dataclass(frozen=True)
class Transport1D:
    """Container for a monotone 1D source-to-target transport construction."""
    probabilities: Array
    source_samples: Array
    target_samples: Array
    velocities: Array
    source_support: Array
    source_pdf: Array
    target_pdf: Array


def normal_pdf(x: Array, mean: float, std: float) -> Array:
    """Evaluate a 1D Gaussian density."""
    variance = std**2
    return jnp.exp(-0.5 * ((x - mean) ** 2) / variance) / jnp.sqrt(2.0 * jnp.pi * variance)


def normal_cdf(x: Array, mean: float, std: float) -> Array:
    """Evaluate a 1D Gaussian cumulative distribution function."""
    return 0.5 * (1.0 + erf((x - mean) / (std * jnp.sqrt(2.0))))


def normal_ppf(probabilities: Array) -> Array:
    """Evaluate the inverse CDF of the standard normal distribution."""
    return jnp.sqrt(2.0) * erfinv(2.0 * probabilities - 1.0)


def target_mixture_pdf(
    x: Array,
    weights: Array = DEFAULT_MIXTURE_WEIGHTS,
    means: Array = DEFAULT_MIXTURE_MEANS,
    stds: Array = DEFAULT_MIXTURE_STDS,
) -> Array:
    """Evaluate the target bimodal mixture density."""
    density = jnp.zeros_like(x, dtype=jnp.float64)
    for weight, mean, std in zip(weights, means, stds, strict=True):
        density += weight * normal_pdf(x, float(mean), float(std))
    return density


def target_mixture_cdf(
    x: Array,
    weights: Array = DEFAULT_MIXTURE_WEIGHTS,
    means: Array = DEFAULT_MIXTURE_MEANS,
    stds: Array = DEFAULT_MIXTURE_STDS,
) -> Array:
    """Evaluate the CDF of the target bimodal mixture."""
    cumulative = jnp.zeros_like(x, dtype=jnp.float64)
    for weight, mean, std in zip(weights, means, stds, strict=True):
        cumulative += weight * normal_cdf(x, float(mean), float(std))
    return cumulative


def build_monotone_transport(
    num_quantiles: int = DEFAULT_NUM_QUANTILES,
    support_min: float = DEFAULT_SUPPORT_MIN,
    support_max: float = DEFAULT_SUPPORT_MAX,
    support_size: int = DEFAULT_SUPPORT_SIZE,
    probability_min: float = DEFAULT_PROBABILITY_MIN,
    probability_max: float = DEFAULT_PROBABILITY_MAX,
) -> Transport1D:
    """Construct a deterministic monotone coupling by matching quantiles.

    The source sample at quantile `p` is `Phi^{-1}(p)`. The target sample at the
    same quantile is found by numerically inverting the target mixture CDF.
    """
    probabilities = jnp.linspace(probability_min, probability_max, num_quantiles, dtype=jnp.float64)
    source_samples = normal_ppf(probabilities)

    support = jnp.linspace(support_min, support_max, support_size, dtype=jnp.float64)
    target_cdf = target_mixture_cdf(support)
    target_samples = jnp.interp(probabilities, target_cdf, support)
    velocities = target_samples - source_samples

    source_pdf = normal_pdf(support, mean=0.0, std=1.0)
    target_pdf = target_mixture_pdf(support)

    assert bool(jnp.all(jnp.diff(source_samples) > 0.0))
    assert bool(jnp.all(jnp.diff(target_samples) > 0.0))
    assert bool(jnp.all(jnp.isfinite(velocities)))

    return Transport1D(
        probabilities=probabilities,
        source_samples=source_samples,
        target_samples=target_samples,
        velocities=velocities,
        source_support=support,
        source_pdf=source_pdf,
        target_pdf=target_pdf,
    )


def interpolation_state(transport: Transport1D, t: float) -> Array:
    """Evaluate the straight interpolation state `x_t = (1-t)x_0 + t x_1`."""
    assert 0.0 <= t <= 1.0
    xt = (1.0 - t) * transport.source_samples + t * transport.target_samples
    assert bool(jnp.all(jnp.isfinite(xt)))
    return xt


def transport_characteristics(transport: Transport1D, chosen_quantiles: Array) -> tuple[Array, Array]:
    """Extract representative source-target pairs at selected quantiles."""
    source = jnp.interp(chosen_quantiles, transport.probabilities, transport.source_samples)
    target = jnp.interp(chosen_quantiles, transport.probabilities, transport.target_samples)
    return source, target


def velocity_field_grid(
    transport: Transport1D,
    time_grid: Array,
    state_grid: Array,
) -> Array:
    """Sample the induced 1D field `u(x,t)` on a regular `(t, x)` grid.

    For each time value we evaluate the monotone interpolation path and then
    interpolate the pairwise velocities onto a state grid.
    """
    field_rows = []
    for time_value in time_grid:
        xt = interpolation_state(transport, float(time_value))
        row = jnp.interp(state_grid, xt, transport.velocities, left=jnp.nan, right=jnp.nan)
        field_rows.append(row)
    field = jnp.stack(field_rows, axis=0)
    return field
