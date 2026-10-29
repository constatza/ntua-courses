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
