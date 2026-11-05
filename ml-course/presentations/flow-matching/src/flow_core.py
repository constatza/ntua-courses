from __future__ import annotations

"""Exact 1D flow-matching objects, with no trained model.

Setting (time convention of this deck: t=0 is noise, t=1 is data):
- source `p_0` is a standard Gaussian,
- target `p_1` is a bimodal Gaussian mixture,
- the coupling is *independent*: x_0 ~ p_0 and x_1 ~ p_1 are drawn separately,
- each pair follows the straight line x_t = (1-t) x_0 + t x_1 with velocity x_1 - x_0.

Because every piece is Gaussian, the averaged field
`u(x, t) = E[x_1 - x_0 | x_t = x]` is available in closed form, so the figures
show the exact field that an ideal regression would learn.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

Array = jax.Array


@dataclass(frozen=True, slots=True)
class TargetMixture1D:
    """Bimodal Gaussian mixture used as the data distribution `p_1`."""

    weights: Array
    means: Array
    stds: Array


DEFAULT_TARGET = TargetMixture1D(
    weights=jnp.array([0.56, 0.44], dtype=jnp.float64),
    means=jnp.array([-1.6, 1.9], dtype=jnp.float64),
    stds=jnp.array([0.42, 0.68], dtype=jnp.float64),
)


def normal_pdf(x: Array, mean: Array | float, variance: Array | float) -> Array:
    """Evaluate a 1D Gaussian density with the given variance."""
    return jnp.exp(-0.5 * (x - mean) ** 2 / variance) / jnp.sqrt(2.0 * jnp.pi * variance)


def target_pdf(x: Array, target: TargetMixture1D = DEFAULT_TARGET) -> Array:
    """Density of the data distribution `p_1`."""
    total = jnp.zeros_like(x, dtype=jnp.float64)
    for weight, mean, std in zip(target.weights, target.means, target.stds, strict=True):
        total += weight * normal_pdf(x, mean, std**2)
    return total


def sample_target(key: Array, num: int, target: TargetMixture1D = DEFAULT_TARGET) -> Array:
    """Draw `num` samples from `p_1`."""
    key_component, key_noise = random.split(key)
    components = random.categorical(key_component, jnp.log(target.weights), shape=(num,))
    noise = random.normal(key_noise, shape=(num,), dtype=jnp.float64)
    return target.means[components] + target.stds[components] * noise


def sample_independent_pairs(key: Array, num: int) -> tuple[Array, Array]:
    """Independent coupling: x_0 from N(0,1) and x_1 from p_1, drawn separately."""
    key_source, key_target = random.split(key)
    x0 = random.normal(key_source, shape=(num,), dtype=jnp.float64)
    return x0, sample_target(key_target, num)


def straight_path(x0: Array, x1: Array, t: Array) -> Array:
    """Points `(1-t) x_0 + t x_1` for each time in `t`; result shape `(len(t), len(x0))`."""
    return (1.0 - t[:, None]) * x0[None, :] + t[:, None] * x1[None, :]


def marginal_velocity(x: Array, t: float, target: TargetMixture1D = DEFAULT_TARGET) -> Array:
    """Exact averaged field `E[x_1 - x_0 | x_t = x]` under the independent coupling.

    Given mixture component k, (x_0, x_1, x_t) are jointly Gaussian, so the
    conditional mean of `x_1 - x_0` is linear in `x_t`. The components are then
    averaged with their posterior responsibilities at `(x, t)`.
    """
    variance_t = (1.0 - t) ** 2 + t**2 * target.stds**2
    mean_t = t * target.means
    gain = (t * target.stds**2 - (1.0 - t)) / variance_t
    component_field = target.means[None, :] + gain[None, :] * (x[:, None] - mean_t[None, :])
    weights = target.weights[None, :] * normal_pdf(x[:, None], mean_t[None, :], variance_t[None, :])
    responsibilities = weights / jnp.sum(weights, axis=1, keepdims=True)
    return jnp.sum(responsibilities * component_field, axis=1)


def integrate_field(x_start: Array, num_steps: int) -> Array:
    """Integrate dx/dt = u(x, t) from t=0 to 1 with the midpoint rule.

    Returns the trajectory with shape `(num_steps + 1, len(x_start))`.
    """
    dt = 1.0 / num_steps
    states = [x_start]
    x = x_start
    for step in range(num_steps):
        t = step * dt
        half = x + 0.5 * dt * marginal_velocity(x, t)
        x = x + dt * marginal_velocity(half, t + 0.5 * dt)
        states.append(x)
    return jnp.stack(states, axis=0)
