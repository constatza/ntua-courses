"""First-order update rules shared by every optimization figure.

Conventions match the slides: momentum accumulates ``v = beta v + g`` and the
step is ``theta -= eta v``; adaptive methods divide by ``sqrt(second) + eps``.
Iterations are 1-based so that Adam's bias correction ``1 - beta**t`` is exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
# A gradient oracle may be stochastic, so it receives the generator explicitly.
GradientOracle = Callable[[Vector, np.random.Generator], Vector]


class OptimizerName(str, Enum):
    GD = "GD"
    MOMENTUM = "Momentum"
    NESTEROV = "Nesterov"
    ADAGRAD = "AdaGrad"
    RMSPROP = "RMSProp"
    ADAM = "Adam"


@dataclass(frozen=True, slots=True)
class Constant:
    rate: float

    def at(self, iteration: int) -> float:
        return self.rate


@dataclass(frozen=True, slots=True)
class PowerDecay:
    """eta_t = eta_0 / t**alpha; alpha in (1/2, 1] is the classical stochastic range."""

    rate: float
    alpha: float

    def at(self, iteration: int) -> float:
        return self.rate / iteration**self.alpha


@dataclass(frozen=True, slots=True)
class WarmupCosine:
    """Linear warm-up to ``peak`` over ``warmup`` updates, then cosine decay to zero at ``total``."""

    peak: float
    warmup: int
    total: int

    def at(self, iteration: int) -> float:
        if iteration <= self.warmup:
            return self.peak * iteration / self.warmup
        progress = (iteration - self.warmup) / (self.total - self.warmup)
        return 0.5 * self.peak * (1.0 + math.cos(math.pi * min(progress, 1.0)))


Schedule = Constant | PowerDecay | WarmupCosine


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: OptimizerName
    schedule: Schedule
    first_decay: float = 0.9
    second_decay: float = 0.999
    epsilon: float = 1e-8


@dataclass(frozen=True, slots=True)
class OptimizerState:
    first_moment: Vector
    second_moment: Vector

    @staticmethod
    def zeros(size: int) -> OptimizerState:
        return OptimizerState(np.zeros(size), np.zeros(size))


def optimizer_direction(
    name: OptimizerName,
    gradient: Vector,
    state: OptimizerState,
    iteration: int,
    first_decay: float,
    second_decay: float,
    epsilon: float,
) -> tuple[Vector, OptimizerState]:
    """Return the direction multiplied by the learning rate, and the updated state."""
    first, second = state.first_moment, state.second_moment
    match name:
        case OptimizerName.GD:
            direction = gradient
        case OptimizerName.MOMENTUM | OptimizerName.NESTEROV:
            first = first_decay * first + gradient
            direction = first
        case OptimizerName.ADAGRAD:
            second = second + gradient**2
            direction = gradient / (np.sqrt(second) + epsilon)
        case OptimizerName.RMSPROP:
            second = second_decay * second + (1.0 - second_decay) * gradient**2
            direction = gradient / (np.sqrt(second) + epsilon)
        case OptimizerName.ADAM:
            first = first_decay * first + (1.0 - first_decay) * gradient
            second = second_decay * second + (1.0 - second_decay) * gradient**2
            corrected_first = first / (1.0 - first_decay**iteration)
            corrected_second = second / (1.0 - second_decay**iteration)
            direction = corrected_first / (np.sqrt(corrected_second) + epsilon)
    return direction, OptimizerState(first, second)


def run(
    config: OptimizerConfig,
    start: Vector,
    oracle: GradientOracle,
    steps: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Return the ``(steps + 1, d)`` trajectory, including the start."""
    parameters = start.copy()
    state = OptimizerState.zeros(parameters.size)
    trajectory = [parameters.copy()]
    for iteration in range(1, steps + 1):
        rate = config.schedule.at(iteration)
        query = parameters
        if config.name is OptimizerName.NESTEROV:
            # Evaluate the gradient where momentum alone would carry the iterate.
            query = parameters - rate * config.first_decay * state.first_moment
        direction, state = optimizer_direction(
            config.name,
            oracle(query, rng),
            state,
            iteration,
            config.first_decay,
            config.second_decay,
            config.epsilon,
        )
        parameters = parameters - rate * direction
        trajectory.append(parameters.copy())
    return np.asarray(trajectory)


def run_constant_gradient(
    config: OptimizerConfig, gradient: float, steps: int
) -> NDArray[np.float64]:
    """One-dimensional trajectory under a constant gradient, used for hand-checked references."""
    oracle: GradientOracle = lambda _parameters, _rng: np.array([gradient])
    trajectory = run(config, np.zeros(1), oracle, steps, np.random.default_rng(0))
    return trajectory[1:, 0]
