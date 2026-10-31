from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from make_optimizer_figures import (
    EXPECTED_FILENAMES,
    INITIAL_POINT,
    OptimizerName,
    generate_figures,
    quadratic_gradient,
    quadratic_value,
    run_constant_gradient_updates,
    run_optimizer,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OptimizerFigureTests(unittest.TestCase):
    def test_quadratic_gradient_matches_centered_difference(self) -> None:
        point = np.array([1.2, -0.7])
        step = 1e-6
        numerical = np.empty_like(point)
        for index in range(point.size):
            offset = np.zeros_like(point)
            offset[index] = step
            numerical[index] = (
                quadratic_value(point + offset) - quadratic_value(point - offset)
            ) / (2.0 * step)

        np.testing.assert_allclose(quadratic_gradient(point), numerical, rtol=1e-6, atol=1e-7)

    def test_each_optimizer_starts_together_and_reduces_the_objective(self) -> None:
        for optimizer in OptimizerName:
            with self.subTest(optimizer=optimizer.value):
                trajectory = run_optimizer(optimizer, steps=80, seed=20261029)
                np.testing.assert_allclose(trajectory[0], INITIAL_POINT)
                self.assertLess(quadratic_value(trajectory[-1]), quadratic_value(trajectory[0]))

    def test_stateful_updates_match_hand_calculated_constant_gradients(self) -> None:
        expected = {
            OptimizerName.MOMENTUM: np.array([-0.1, -0.29, -0.561]),
            OptimizerName.ADAGRAD: np.array(
                [-0.1, -0.17071067811865476, -0.22844570503761735]
            ),
            OptimizerName.RMSPROP: np.array(
                [-0.31622776601683794, -0.5456434998873998, -0.7377381874862244]
            ),
            OptimizerName.ADAM: np.array([-0.1, -0.2, -0.3]),
        }

        for optimizer, wanted in expected.items():
            with self.subTest(optimizer=optimizer.value):
                actual = run_constant_gradient_updates(
                    optimizer,
                    gradient=1.0,
                    steps=3,
                    learning_rate=0.1,
                    first_decay=0.9,
                    second_decay=0.9 if optimizer is OptimizerName.RMSPROP else 0.999,
                    epsilon=0.0,
                )
                np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-12)

    def test_deterministic_comparison_reaches_the_minimum_by_objective_and_gradient(self) -> None:
        deterministic = tuple(
            optimizer for optimizer in OptimizerName if optimizer is not OptimizerName.SGD
        )
        for optimizer in deterministic:
            with self.subTest(optimizer=optimizer.value):
                final = run_optimizer(optimizer, steps=1_000, seed=20261029)[-1]
                self.assertLess(quadratic_value(final), 1e-4)
                self.assertLess(np.linalg.norm(quadratic_gradient(final)), 2e-2)

    def test_generate_figures_writes_expected_nonempty_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = generate_figures(Path(directory))

            self.assertEqual(tuple(path.name for path in outputs), EXPECTED_FILENAMES)
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))

    def test_generate_figures_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_outputs = generate_figures(Path(first))
            second_outputs = generate_figures(Path(second))

            self.assertEqual(
                tuple(file_hash(path) for path in first_outputs),
                tuple(file_hash(path) for path in second_outputs),
            )


if __name__ == "__main__":
    unittest.main()
