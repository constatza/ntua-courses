from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("make_generalization_figures.py")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_generator():
    spec = importlib.util.spec_from_file_location("make_generalization_figures", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeneralizationFigureTests(unittest.TestCase):
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

    def test_target_panels_separate_bias_from_variance(self) -> None:
        generator = load_generator()
        rng = np.random.default_rng(0)
        count = 4000

        low = generator.TargetPanel(generator.LOW_BIAS, generator.LOW_SPREAD).sample_weights(rng, count)
        biased = generator.TargetPanel(generator.HIGH_BIAS, generator.LOW_SPREAD).sample_weights(rng, count)
        scattered = generator.TargetPanel(generator.LOW_BIAS, generator.HIGH_SPREAD).sample_weights(rng, count)

        # Bias is the distance of the mean fit from w* = 0; variance is the scatter around the mean.
        self.assertLess(np.linalg.norm(low.mean(axis=0)), 0.05)
        self.assertAlmostEqual(np.linalg.norm(biased.mean(axis=0)), generator.HIGH_BIAS, delta=0.05)
        self.assertGreater(scattered.std(axis=0).min(), 2.0 * low.std(axis=0).max())
        self.assertAlmostEqual(biased.std(axis=0).mean(), low.std(axis=0).mean(), delta=0.02)

    def test_every_total_loss_is_exactly_data_plus_penalty(self) -> None:
        generator = load_generator()
        unregularized = generator.L2_UNREGULARIZED
        strength = generator.L2_STRENGTH

        landscape = generator.l2_landscape(unregularized, strength)
        curves = generator.one_weight_curves(generator.ONE_WEIGHT_UNREGULARIZED, strength)

        np.testing.assert_array_equal(landscape.total, landscape.data_loss + landscape.penalty)
        np.testing.assert_array_equal(curves.total, curves.data_loss + curves.penalty)

    def test_total_minimum_matches_the_closed_form_and_the_slide_numbers(self) -> None:
        generator = load_generator()
        unregularized = generator.L2_UNREGULARIZED
        strength = generator.L2_STRENGTH

        landscape = generator.l2_landscape(unregularized, strength)
        regularized = generator.l2_solution(unregularized, strength)
        curves = generator.one_weight_curves(generator.ONE_WEIGHT_UNREGULARIZED, strength)

        np.testing.assert_allclose(landscape.minimizer(), regularized, atol=0.02)
        data = 0.5 * np.sum((regularized - unregularized) ** 2)
        penalty = 0.5 * strength * np.sum(regularized**2)
        self.assertAlmostEqual(data, 10 / 9)
        self.assertAlmostEqual(penalty, 5 / 9)
        self.assertAlmostEqual(data + penalty, 5 / 3)
        best_weight = curves.weight[np.argmin(curves.total)]
        self.assertAlmostEqual(best_weight, generator.ONE_WEIGHT_UNREGULARIZED / (1 + strength), delta=0.01)

    def test_l1_optimum_sits_on_an_axis_and_l2_optimum_does_not(self) -> None:
        generator = load_generator()

        l1_point = generator.constrained_minimizer(generator.l1_ball_boundary())
        l2_point = generator.constrained_minimizer(generator.l2_ball_boundary())

        np.testing.assert_allclose(l1_point, [generator.BALL_RADIUS, 0.0], atol=1e-9)
        self.assertGreater(np.abs(l2_point).min(), 0.05)
        # The unconstrained best fit lies outside both shapes, so the optimum is on the boundary.
        centre = generator.GEOMETRY_CENTRE
        self.assertGreater(np.abs(centre).sum(), generator.BALL_RADIUS)
        self.assertGreater(np.linalg.norm(centre), generator.BALL_RADIUS)


if __name__ == "__main__":
    unittest.main()
