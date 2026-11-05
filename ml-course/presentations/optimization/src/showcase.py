"""Learning-rate sweeps on the small network: the evidence behind the "why Adam" frames.

Every method sees the same data, initial weights, mini-batch stream and update budget; only
the learning rate (and the method) changes. Scores are test losses averaged over seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from optimizers import Constant, OptimizerConfig, OptimizerName, run
from problems import MlpRegression

BATCH_SIZE = 32
EPOCHS = 40
SEEDS = 5
LEARNING_RATES = np.geomspace(1e-4, 1.0, 17)
METHODS = (
    OptimizerName.GD,
    OptimizerName.MOMENTUM,
    OptimizerName.RMSPROP,
    OptimizerName.ADAM,
)
MOMENTUM_BETA = 0.9
RMSPROP_RHO = 0.9


@dataclass(frozen=True, slots=True)
class Sweep:
    """Mean test loss after training, per method and learning rate, plus the best run's curve."""

    final_loss: dict[OptimizerName, NDArray[np.float64]]
    best_curve: dict[OptimizerName, NDArray[np.float64]]  # mean test loss after each epoch

    def best_rate(self, method: OptimizerName) -> float:
        return float(LEARNING_RATES[np.argmin(self.final_loss[method])])

    def best_loss(self, method: OptimizerName) -> float:
        return float(np.min(self.final_loss[method]))


def _config(method: OptimizerName, rate: float) -> OptimizerConfig:
    if method is OptimizerName.RMSPROP:
        return OptimizerConfig(method, Constant(rate), second_decay=RMSPROP_RHO)
    return OptimizerConfig(method, Constant(rate), first_decay=MOMENTUM_BETA)


def _epoch_losses(
    problem: MlpRegression, method: OptimizerName, rate: float, seed: int
) -> NDArray[np.float64]:
    """Test loss at the start and after every epoch; ``inf`` once training diverged."""
    steps_per_epoch = problem.train_targets.size // BATCH_SIZE
    start = problem.initial_parameters(np.random.default_rng(1000 + seed))
    with np.errstate(all="ignore"):
        trajectory = run(
            _config(method, rate),
            start,
            problem.batch_oracle(BATCH_SIZE),
            EPOCHS * steps_per_epoch,
            np.random.default_rng(seed),
        )
        return np.array(
            [problem.test_loss(trajectory[epoch * steps_per_epoch]) for epoch in range(EPOCHS + 1)]
        )


def sweep(problem: MlpRegression) -> Sweep:
    final_loss: dict[OptimizerName, NDArray[np.float64]] = {}
    best_curve: dict[OptimizerName, NDArray[np.float64]] = {}
    for method in METHODS:
        curves = np.array(
            [
                [_epoch_losses(problem, method, float(rate), seed) for seed in range(SEEDS)]
                for rate in LEARNING_RATES
            ]
        )  # (rate, seed, epoch + 1)
        mean_curves = curves.mean(axis=1)
        final_loss[method] = mean_curves[:, -1]
        best_curve[method] = mean_curves[int(np.argmin(mean_curves[:, -1]))]
    return Sweep(final_loss, best_curve)
