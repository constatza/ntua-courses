from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("make_regularization_figures.py")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_generator():
    spec = importlib.util.spec_from_file_location("make_regularization_figures", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegularizationFigureTests(unittest.TestCase):
    def test_generator_writes_every_deterministic_figure(self) -> None:
        generator = load_generator()

        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_outputs = generator.generate_figures(Path(first))
            second_outputs = generator.generate_figures(Path(second))

            self.assertEqual(
                tuple(path.name for path in first_outputs),
                generator.EXPECTED_FILENAMES,
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in first_outputs))
            self.assertEqual(
                tuple(file_hash(path) for path in first_outputs),
                tuple(file_hash(path) for path in second_outputs),
            )

    def test_l2_solution_moves_toward_zero(self) -> None:
        generator = load_generator()
        unregularized = np.array([2.0, 1.0])

        regularized = generator.l2_solution(unregularized, strength=2.0)

        np.testing.assert_allclose(regularized, np.array([2.0 / 3.0, 1.0 / 3.0]))
        self.assertLess(np.linalg.norm(regularized), np.linalg.norm(unregularized))


if __name__ == "__main__":
    unittest.main()
