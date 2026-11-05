from __future__ import annotations

"""Closed-form diffusion helpers used by the educational figures.

This module only implements quantities that can be written exactly:
- the forward schedule,
- forward and posterior means,
- oracle reverse denoising,
- analytic scores for a smoothed Gaussian mixture,
- a real DDPM sampler driven by the exact (analytic) score of a Gaussian
  mixture, so no network has to be trained.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

Array = jax.Array

GMM_DEFAULT_WEIGHTS = jnp.array([0.35, 0.30, 0.35], dtype=jnp.float64)
GMM_DEFAULT_MEANS = jnp.array(
    [
        [-2.0, -0.7],
        [0.0, 1.9],
        [2.1, -0.8],
    ],
    dtype=jnp.float64,
)
GMM_DEFAULT_VARIANCES = jnp.array([0.28, 0.40, 0.30], dtype=jnp.float64)


@dataclass(frozen=True, slots=True)
class DiffusionSchedule:
    """Discrete DDPM schedule with per-step and cumulative coefficients."""
    beta: Array
    alpha: Array
    alpha_bar: Array

    @property
    def num_steps(self) -> int:
        return int(self.beta.shape[0] - 1)


@dataclass(frozen=True, slots=True)
class GaussianMixture2D:
    """Simple isotropic Gaussian mixture used for the score-field figure."""
    weights: Array
    means: Array
    variances: Array


def build_linear_schedule(num_steps: int, beta_start: float, beta_end: float) -> DiffusionSchedule:
    """Build a linear DDPM variance schedule.

    `beta[0]` is set to zero so that arrays are indexed directly by timestep.
    """
    assert num_steps >= 2
    beta_body = jnp.linspace(beta_start, beta_end, num_steps, dtype=jnp.float64)
    beta = jnp.concatenate((jnp.array([0.0], dtype=jnp.float64), beta_body))
    alpha = 1.0 - beta
    alpha_bar = jnp.cumprod(alpha)
    assert beta.shape == alpha.shape == alpha_bar.shape
    assert bool(jnp.all(beta[1:] > 0.0))
    assert bool(jnp.all(beta[1:] < 1.0))
    assert bool(jnp.isclose(alpha_bar[0], 1.0))
    assert bool(jnp.all(alpha_bar[1:] < alpha_bar[:-1]))
    return DiffusionSchedule(beta=beta, alpha=alpha, alpha_bar=alpha_bar)


def direct_noisy_sample(
    x0: Array,
    schedule: DiffusionSchedule,
    timestep: int,
    noise: Array,
) -> Array:
    """Sample `x_t` directly from the closed form `q(x_t | x_0)`."""
    assert 0 <= timestep <= schedule.num_steps
    assert noise.shape == x0.shape
    signal_scale = jnp.sqrt(schedule.alpha_bar[timestep])
    noise_scale = jnp.sqrt(1.0 - schedule.alpha_bar[timestep])
    xt = signal_scale * x0 + noise_scale * noise
    assert bool(jnp.all(jnp.isfinite(xt)))
    return xt


def forward_step(x_prev: Array, beta_t: float, key: Array) -> Array:
    """Apply one exact forward diffusion step."""
    assert 0.0 < beta_t < 1.0
    epsilon_t = random.normal(key, shape=x_prev.shape, dtype=x_prev.dtype)
    xt = jnp.sqrt(1.0 - beta_t) * x_prev + jnp.sqrt(beta_t) * epsilon_t
    assert bool(jnp.all(jnp.isfinite(xt)))
    return xt


def simulate_forward_chain(
    x0: Array,
    schedule: DiffusionSchedule,
    key: Array,
) -> list[Array]:
    """Simulate the full forward Markov chain from `x_0` to `x_T`."""
    states = [jnp.asarray(x0, dtype=jnp.float64)]
    x_t = jnp.asarray(x0, dtype=jnp.float64)
    keys = random.split(key, schedule.num_steps)
    for timestep in range(1, schedule.num_steps + 1):
        x_t = forward_step(x_t, float(schedule.beta[timestep]), keys[timestep - 1])
        states.append(x_t)
    assert len(states) == schedule.num_steps + 1
    return states


def posterior_variance(schedule: DiffusionSchedule, timestep: int) -> float:
    """Return the exact scalar posterior variance coefficient `\tilde{\beta}_t`."""
    assert 1 <= timestep <= schedule.num_steps
    alpha_bar_t = schedule.alpha_bar[timestep]
    alpha_bar_prev = schedule.alpha_bar[timestep - 1]
    beta_t = schedule.beta[timestep]
    variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    assert variance >= 0.0
    return float(variance)


def posterior_mean(x_t: Array, x0: Array, schedule: DiffusionSchedule, timestep: int) -> Array:
    """Return the exact posterior mean of `q(x_{t-1} | x_t, x_0)`."""
    assert 1 <= timestep <= schedule.num_steps
    beta_t = schedule.beta[timestep]
    alpha_t = schedule.alpha[timestep]
    alpha_bar_t = schedule.alpha_bar[timestep]
    alpha_bar_prev = schedule.alpha_bar[timestep - 1]

    coeff_x0 = jnp.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coeff_xt = jnp.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    mean = coeff_x0 * x0 + coeff_xt * x_t
    assert bool(jnp.all(jnp.isfinite(mean)))
    return mean


def oracle_reverse_chain(
    x0: Array,
    x_T: Array,
    schedule: DiffusionSchedule,
) -> list[Array]:
    """Build an oracle reverse chain using posterior means at every step.

    This is not a learned sampler. It visualizes what the exact reverse update
    looks like when the clean sample `x_0` is known.
    """
    states: list[Array] = [jnp.asarray(x_T, dtype=jnp.float64)]
    x_t = jnp.asarray(x_T, dtype=jnp.float64)
    for timestep in range(schedule.num_steps, 0, -1):
        x_t = posterior_mean(x_t, x0, schedule, timestep)
        states.append(x_t)
    reversed_states = list(reversed(states))
    assert len(reversed_states) == schedule.num_steps + 1
    return reversed_states


def make_default_gmm() -> GaussianMixture2D:
    """Return the default 2D mixture used for the score figure."""
    weights = GMM_DEFAULT_WEIGHTS
    means = GMM_DEFAULT_MEANS
    variances = GMM_DEFAULT_VARIANCES
    assert bool(jnp.isclose(jnp.sum(weights), 1.0))
    return GaussianMixture2D(weights=weights, means=means, variances=variances)


def gaussian_density_isotropic(points: Array, mean: Array, variance: float) -> Array:
    """Evaluate an isotropic Gaussian density on a batch of 2D points."""
    dim = mean.shape[0]
    centered = points - mean
    exponent = -0.5 * jnp.sum(centered * centered, axis=-1) / variance
    normalizer = (2.0 * jnp.pi * variance) ** (dim / 2.0)
    density = jnp.exp(exponent) / normalizer
    assert bool(jnp.all(jnp.isfinite(density)))
    return density


def smoothed_gmm_density(points: Array, mixture: GaussianMixture2D, noise_variance: float) -> Array:
    """Evaluate the mixture density after additional isotropic Gaussian smoothing."""
    total = jnp.zeros(points.shape[:-1], dtype=jnp.float64)
    for weight, mean, variance in zip(mixture.weights, mixture.means, mixture.variances, strict=True):
        total += weight * gaussian_density_isotropic(points, mean, variance + noise_variance)
    assert bool(jnp.all(total > 0.0))
    return total


def smoothed_gmm_score(points: Array, mixture: GaussianMixture2D, noise_variance: float) -> Array:
    """Compute the exact score of the smoothed Gaussian mixture.

    Each component remains Gaussian after smoothing, so the mixture score is the
    responsibility-weighted average of the component scores.
    """
    component_densities = []
    component_scores = []
    for weight, mean, variance in zip(mixture.weights, mixture.means, mixture.variances, strict=True):
        effective_variance = variance + noise_variance
        density = weight * gaussian_density_isotropic(points, mean, effective_variance)
        score = (mean - points) / effective_variance
        component_densities.append(density)
        component_scores.append(score)

    stacked_densities = jnp.stack(component_densities, axis=0)
    total_density = jnp.sum(stacked_densities, axis=0)
    responsibilities = stacked_densities / total_density[None, ...]
    stacked_scores = jnp.stack(component_scores, axis=0)
    score = jnp.sum(responsibilities[..., None] * stacked_scores, axis=0)
    assert bool(jnp.all(jnp.isfinite(score)))
    return score


def sample_gmm(mixture: GaussianMixture2D, num_samples: int, key: Array) -> Array:
    """Draw samples from the isotropic Gaussian mixture."""
    key_component, key_noise = random.split(key)
    components = random.categorical(key_component, jnp.log(mixture.weights), shape=(num_samples,))
    stds = jnp.sqrt(mixture.variances[components])[:, None]
    noise = random.normal(key_noise, shape=(num_samples, mixture.means.shape[1]), dtype=jnp.float64)
    samples = mixture.means[components] + stds * noise
    assert bool(jnp.all(jnp.isfinite(samples)))
    return samples


def noisy_gmm_parameters(
    mixture: GaussianMixture2D,
    schedule: DiffusionSchedule,
    timestep: int,
) -> GaussianMixture2D:
    """Return the mixture p_t of x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps.

    Each component stays Gaussian: its mean shrinks by sqrt(abar_t) and its
    variance becomes abar_t * v_k + (1 - abar_t).
    """
    alpha_bar_t = schedule.alpha_bar[timestep]
    return GaussianMixture2D(
        weights=mixture.weights,
        means=jnp.sqrt(alpha_bar_t) * mixture.means,
        variances=alpha_bar_t * mixture.variances + (1.0 - alpha_bar_t),
    )


def noisy_gmm_score(
    points: Array,
    mixture: GaussianMixture2D,
    schedule: DiffusionSchedule,
    timestep: int,
) -> Array:
    """Exact score of p_t for the diffusion of a Gaussian mixture."""
    noisy = noisy_gmm_parameters(mixture, schedule, timestep)
    return smoothed_gmm_score(points, noisy, noise_variance=0.0)


def ddpm_reverse_sample(
    x_T: Array,
    mixture: GaussianMixture2D,
    schedule: DiffusionSchedule,
    key: Array,
    keep_steps: list[int],
) -> dict[int, Array]:
    """Run the DDPM sampler from pure noise with the exact score as the denoiser.

    The noise prediction is eps_hat = -sigma_t * score, plugged into the DDPM mean
    formula; the added noise has the posterior variance. The true data point is
    never used, so this is a real sampler (with an ideal network).
    """
    kept: dict[int, Array] = {}
    x_t = jnp.asarray(x_T, dtype=jnp.float64)
    keys = random.split(key, schedule.num_steps)
    for timestep in range(schedule.num_steps, 0, -1):
        if timestep in keep_steps:
            kept[timestep] = x_t
        sigma_t = jnp.sqrt(1.0 - schedule.alpha_bar[timestep])
        eps_hat = -sigma_t * noisy_gmm_score(x_t, mixture, schedule, timestep)
        beta_t = schedule.beta[timestep]
        mean = (x_t - beta_t / sigma_t * eps_hat) / jnp.sqrt(schedule.alpha[timestep])
        std = jnp.sqrt(posterior_variance(schedule, timestep)) if timestep > 1 else 0.0
        x_t = mean + std * random.normal(keys[timestep - 1], x_t.shape, dtype=jnp.float64)
    kept[0] = x_t
    assert set(keep_steps) - {0} <= set(kept)
    return kept
