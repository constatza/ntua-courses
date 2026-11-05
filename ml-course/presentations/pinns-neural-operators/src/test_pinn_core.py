"""The PINN must solve the easy problem; the training must be reproducible."""

from __future__ import annotations

from pinn_core import Problem, train

STEPS = 3000
LEARNING_RATE = 1.0e-2
MAX_ERROR_LOW_FREQUENCY = 1.0e-3


def test_low_frequency_problem_is_solved() -> None:
    run = train(Problem(1), seed=0, num_steps=STEPS, learning_rate=LEARNING_RATE, snapshot_steps=[])
    assert run.relative_error[-1] < MAX_ERROR_LOW_FREQUENCY


def test_training_is_deterministic() -> None:
    first = train(Problem(1), seed=0, num_steps=50, learning_rate=LEARNING_RATE, snapshot_steps=[])
    second = train(Problem(1), seed=0, num_steps=50, learning_rate=LEARNING_RATE, snapshot_steps=[])
    assert first.relative_error == second.relative_error
