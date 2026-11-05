from __future__ import annotations

"""A tiny 1D PINN: -u'' = (k pi)^2 sin(k pi x) on (0, 1), u(0) = u(1) = 0.

The exact solution is sin(k pi x). The trial function u(x) = x (1 - x) N(x) makes
the boundary conditions hold for every network N, so only the interior residual is
trained. Training is full-batch Adam on a fixed grid of collocation points.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

Array = jax.Array
Params = list[tuple[Array, Array]]

HIDDEN_WIDTH = 24
NUM_COLLOCATION = 64
NUM_TEST = 400
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1.0e-8


@dataclass(frozen=True, slots=True)
class Problem:
    """Frequency `k` of the manufactured solution sin(k pi x)."""

    frequency: int

    def exact(self, x: Array) -> Array:
        return jnp.sin(self.frequency * jnp.pi * x)

    def forcing(self, x: Array) -> Array:
        return (self.frequency * jnp.pi) ** 2 * jnp.sin(self.frequency * jnp.pi * x)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Snapshots of one training run; all arrays are indexed by the recorded steps."""

    steps: list[int]
    residual_loss: list[float]
    relative_error: list[float]
    predictions: dict[int, Array]
    test_points: Array


def init_params(key: Array, width: int = HIDDEN_WIDTH) -> Params:
    sizes = [1, width, width, 1]
    keys = random.split(key, len(sizes) - 1)
    params: Params = []
    for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:], strict=True):
        weight = random.normal(k, (n_in, n_out), dtype=jnp.float64) / jnp.sqrt(n_in)
        params.append((weight, jnp.zeros(n_out, dtype=jnp.float64)))
    return params


def network(params: Params, x: Array) -> Array:
    """Scalar network N(x) for a scalar input x."""
    h = jnp.atleast_1d(x)
    for weight, bias in params[:-1]:
        h = jnp.tanh(h @ weight + bias)
    weight, bias = params[-1]
    return (h @ weight + bias)[0]


def trial_solution(params: Params, x: Array) -> Array:
    """Hard-constrained solution u(x) = x (1 - x) N(x)."""
    return x * (1.0 - x) * network(params, x)


def residual(params: Params, x: Array, problem: Problem) -> Array:
    """r(x) = -u''(x) - f(x), with u'' from automatic differentiation."""
    second = jax.grad(jax.grad(trial_solution, argnums=1), argnums=1)(params, x)
    return -second - problem.forcing(x)


def residual_loss(params: Params, points: Array, problem: Problem) -> Array:
    values = jax.vmap(lambda x: residual(params, x, problem))(points)
    return jnp.mean(values**2)


def relative_l2_error(params: Params, points: Array, problem: Problem) -> Array:
    prediction = jax.vmap(lambda x: trial_solution(params, x))(points)
    reference = problem.exact(points)
    return jnp.linalg.norm(prediction - reference) / jnp.linalg.norm(reference)


def train(
    problem: Problem,
    seed: int,
    num_steps: int,
    learning_rate: float,
    snapshot_steps: list[int],
) -> TrainingRun:
    """Train with full-batch Adam on a fixed collocation grid and record snapshots."""
    params = init_params(random.PRNGKey(seed))
    collocation = jnp.linspace(0.0, 1.0, NUM_COLLOCATION + 2, dtype=jnp.float64)[1:-1]
    test_points = jnp.linspace(0.0, 1.0, NUM_TEST, dtype=jnp.float64)
    loss_and_grad = jax.jit(jax.value_and_grad(lambda p: residual_loss(p, collocation, problem)))
    error_fn = jax.jit(lambda p: relative_l2_error(p, test_points, problem))
    predict = jax.jit(jax.vmap(lambda p, x: trial_solution(p, x), in_axes=(None, 0)))

    first = jax.tree_util.tree_map(jnp.zeros_like, params)
    second = jax.tree_util.tree_map(jnp.zeros_like, params)
    steps: list[int] = []
    losses: list[float] = []
    errors: list[float] = []
    predictions: dict[int, Array] = {}
    for step in range(num_steps + 1):
        loss, grads = loss_and_grad(params)
        if step % 10 == 0 or step in snapshot_steps:
            steps.append(step)
            losses.append(float(loss))
            errors.append(float(error_fn(params)))
        if step in snapshot_steps:
            predictions[step] = predict(params, test_points)
        first = jax.tree_util.tree_map(lambda m, g: ADAM_BETA1 * m + (1 - ADAM_BETA1) * g, first, grads)
        second = jax.tree_util.tree_map(lambda v, g: ADAM_BETA2 * v + (1 - ADAM_BETA2) * g * g, second, grads)
        scale1 = 1.0 - ADAM_BETA1 ** (step + 1)
        scale2 = 1.0 - ADAM_BETA2 ** (step + 1)
        params = jax.tree_util.tree_map(
            lambda p, m, v: p - learning_rate * (m / scale1) / (jnp.sqrt(v / scale2) + ADAM_EPS),
            params, first, second,
        )
    return TrainingRun(steps, losses, errors, predictions, test_points)
