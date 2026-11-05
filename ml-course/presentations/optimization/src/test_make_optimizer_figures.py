from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from make_optimizer_figures import (
    EXPECTED_FILENAMES,
    SADDLE_CONFIGS,
    SEED,
    basin_outcomes,
    contraction_factor,
    escape_step,
    generate_figures,
    heavy_ball_parameters,
    MULTIMODAL_CONFIGS,
    nesterov_parameters,
    optimal_gd_rate,
    protocol_table,
    tuned_protocol,
)
from optimizers import (
    Constant,
    OptimizerConfig,
    OptimizerName,
    PowerDecay,
    WarmupCosine,
    run,
    run_constant_gradient,
)
from problems import (
    LAB,
    LAB_START,
    Basin,
    DoubleWell,
    Rosenbrock,
    StyblinskiTang,
    make_finite_sum_lab,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def centered_difference(value, point: np.ndarray, step: float = 1e-6) -> np.ndarray:
    numerical = np.empty_like(point)
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset[index] = step
        numerical[index] = (value(point + offset) - value(point - offset)) / (2.0 * step)
    return numerical


def contraction_per_update(config: OptimizerConfig, steps: int = 80) -> float:
    """Geometric-mean contraction of the parameter error over the second half of a run."""
    trajectory = run(config, LAB_START, LAB.oracle, steps, np.random.default_rng(SEED))
    errors = np.linalg.norm(trajectory, axis=1)
    half = steps // 2
    return float((errors[steps] / errors[half]) ** (1.0 / (steps - half)))


class GradientTests(unittest.TestCase):
    def test_analytic_gradients_match_centered_differences(self) -> None:
        cases = {
            "quadratic": (LAB.value, LAB.gradient, np.array([1.2, -0.7])),
            "double well": (DoubleWell().value, DoubleWell().gradient, np.array([0.4, -0.9])),
            "rosenbrock": (Rosenbrock().value, Rosenbrock().gradient, np.array([-0.8, 0.6])),
            "styblinski-tang": (
                StyblinskiTang().value,
                StyblinskiTang().gradient,
                np.array([1.3, -2.2]),
            ),
        }
        for name, (value, gradient, point) in cases.items():
            with self.subTest(problem=name):
                np.testing.assert_allclose(
                    gradient(point), centered_difference(value, point), rtol=1e-5, atol=1e-6
                )


class FiniteSumLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lab = make_finite_sum_lab(SEED)

    def test_full_batch_objective_is_the_quadratic_laboratory(self) -> None:
        point = np.array([1.7, -0.4])
        np.testing.assert_allclose(self.lab.full_gradient(point), LAB.gradient(point), atol=1e-12)
        np.testing.assert_allclose(self.lab.full_gradient(np.zeros(2)), np.zeros(2), atol=1e-12)

    def test_sampling_noise_does_not_vanish_at_the_minimizer(self) -> None:
        per_sample = self.lab.sample_gradients(np.zeros(2))
        self.assertGreater(per_sample.std(axis=0).min(), 0.5)

    def test_mini_batch_gradient_is_unbiased_with_variance_sigma_squared_over_b(self) -> None:
        point = np.array([1.0, 1.0])
        per_sample = self.lab.sample_gradients(point)
        rng = np.random.default_rng(0)
        for batch_size in (1, 16):
            estimates = np.array(
                [self.lab.batch_oracle(batch_size)(point, rng) for _ in range(20_000)]
            )
            with self.subTest(batch_size=batch_size):
                np.testing.assert_allclose(estimates.mean(axis=0), per_sample.mean(axis=0), atol=0.05)
                np.testing.assert_allclose(
                    estimates.var(axis=0), per_sample.var(axis=0) / batch_size, rtol=0.06
                )


class UpdateRuleTests(unittest.TestCase):
    def test_stateful_updates_match_hand_calculated_constant_gradients(self) -> None:
        expected = {
            OptimizerName.MOMENTUM: [-0.1, -0.29, -0.561],
            OptimizerName.ADAGRAD: [-0.1, -0.17071067811865476, -0.22844570503761735],
            OptimizerName.RMSPROP: [-0.31622776601683794, -0.5456434998873998, -0.7377381874862244],
            OptimizerName.ADAM: [-0.1, -0.2, -0.3],
        }
        for name, wanted in expected.items():
            with self.subTest(optimizer=name.value):
                config = OptimizerConfig(
                    name,
                    Constant(0.1),
                    second_decay=0.9 if name is OptimizerName.RMSPROP else 0.999,
                    epsilon=0.0,
                )
                np.testing.assert_allclose(
                    run_constant_gradient(config, 1.0, 3), wanted, rtol=1e-12, atol=1e-12
                )

    def test_nesterov_evaluates_the_gradient_at_the_look_ahead_point(self) -> None:
        # L(theta) = theta^2 / 2 so the gradient is theta; hand-calculated with eta=0.1, beta=0.9.
        config = OptimizerConfig(OptimizerName.NESTEROV, Constant(0.1), first_decay=0.9)
        trajectory = run(config, np.array([1.0]), lambda p, _rng: p, 3, np.random.default_rng(0))
        np.testing.assert_allclose(trajectory[1:, 0], [0.9, 0.729, 0.51759], rtol=1e-12)

    def test_adam_bias_correction_makes_the_first_step_a_full_learning_rate(self) -> None:
        config = OptimizerConfig(OptimizerName.ADAM, Constant(0.1), epsilon=0.0)
        np.testing.assert_allclose(run_constant_gradient(config, 37.0, 1), [-0.1], rtol=1e-12)

    def test_schedules(self) -> None:
        self.assertAlmostEqual(PowerDecay(0.4, 0.5).at(4), 0.2)
        warmup = WarmupCosine(peak=0.1, warmup=10, total=110)
        self.assertAlmostEqual(warmup.at(5), 0.05)
        self.assertAlmostEqual(warmup.at(10), 0.1)
        self.assertAlmostEqual(warmup.at(60), 0.05)
        self.assertAlmostEqual(warmup.at(110), 0.0)


class QuadraticTheoryTests(unittest.TestCase):
    """Every rate quoted on the slides is measured here, not only derived."""

    def setUp(self) -> None:
        self.lambda_min, self.lambda_max = LAB.eigenvalues

    def test_gradient_descent_is_stable_exactly_below_two_over_lambda_max(self) -> None:
        limit = 2.0 / self.lambda_max
        for factor, should_converge in ((0.95, True), (1.05, False)):
            with self.subTest(factor=factor):
                config = OptimizerConfig(OptimizerName.GD, Constant(factor * limit))
                final = run(config, LAB_START, LAB.oracle, 400, np.random.default_rng(0))[-1]
                self.assertEqual(np.linalg.norm(final) < np.linalg.norm(LAB_START), should_converge)

    def test_optimal_gradient_descent_rate_is_kappa_minus_one_over_kappa_plus_one(self) -> None:
        rate = optimal_gd_rate(self.lambda_min, self.lambda_max)
        kappa = self.lambda_max / self.lambda_min
        self.assertAlmostEqual(contraction_factor(rate, LAB.eigenvalues), (kappa - 1) / (kappa + 1))
        measured = contraction_per_update(OptimizerConfig(OptimizerName.GD, Constant(rate)))
        self.assertAlmostEqual(measured, (kappa - 1) / (kappa + 1), delta=0.01)

    def test_heavy_ball_contracts_at_root_beta(self) -> None:
        rate, beta = heavy_ball_parameters(self.lambda_min, self.lambda_max)
        config = OptimizerConfig(OptimizerName.MOMENTUM, Constant(rate), first_decay=beta)
        self.assertAlmostEqual(contraction_per_update(config), np.sqrt(beta), delta=0.03)

    def test_nesterov_contracts_near_one_minus_one_over_root_kappa(self) -> None:
        rate, beta = nesterov_parameters(self.lambda_min, self.lambda_max)
        config = OptimizerConfig(OptimizerName.NESTEROV, Constant(rate), first_decay=beta)
        kappa = self.lambda_max / self.lambda_min
        self.assertAlmostEqual(contraction_per_update(config), 1 - 1 / np.sqrt(kappa), delta=0.03)

    def test_newton_step_solves_a_quadratic_in_one_update(self) -> None:
        point = LAB_START - np.linalg.solve(LAB.hessian, LAB.gradient(LAB_START))
        np.testing.assert_allclose(point, np.zeros(2), atol=1e-12)


class ProtocolTests(unittest.TestCase):
    def test_tuned_gradient_descent_rate_is_near_the_theoretical_optimum(self) -> None:
        tuned = tuned_protocol()[OptimizerName.GD].config.schedule.rate
        best = optimal_gd_rate(*LAB.eigenvalues[[0, -1]])
        self.assertAlmostEqual(tuned / best, 1.0, delta=0.25)

    def test_every_method_makes_progress_and_the_table_lists_all_six(self) -> None:
        initial = LAB.value(LAB_START)
        for name, method in tuned_protocol().items():
            with self.subTest(optimizer=name.value):
                self.assertLess(method.final_gap, 1e-6 * initial)
        table = protocol_table(tuned_protocol())
        for name in OptimizerName:
            self.assertIn(name.value, table)


class NonconvexTests(unittest.TestCase):
    def test_double_well_has_two_minima_one_saddle_and_a_better_global_minimum(self) -> None:
        well = DoubleWell()
        kinds = [
            "minimum" if np.linalg.eigvalsh(well.hessian(p))[0] > 0 else "saddle"
            for p in well.stationary_points()
        ]
        self.assertEqual(sorted(kinds), ["minimum", "minimum", "saddle"])
        minima = [p for p in well.stationary_points() if np.linalg.eigvalsh(well.hessian(p))[0] > 0]
        self.assertNotAlmostEqual(well.value(minima[0]), well.value(minima[1]), places=2)

    def test_start_decides_the_basin_and_optimizers_agree(self) -> None:
        global_shares = []
        for _label, config, _color in MULTIMODAL_CONFIGS:
            outcomes = basin_outcomes(config)
            self.assertNotIn(Basin.UNSETTLED, outcomes)
            global_shares.append(np.mean([outcome is Basin.GLOBAL for outcome in outcomes]))
        self.assertTrue(all(0.2 < share < 0.4 for share in global_shares))
        self.assertLess(max(global_shares) - min(global_shares), 0.05)

    def test_momentum_and_adam_leave_a_saddle_sooner_than_gradient_descent(self) -> None:
        steps = {label: escape_step(config) for label, config, _color in SADDLE_CONFIGS}
        self.assertGreater(steps["GD"], 2 * steps["Momentum"])
        self.assertGreater(steps["Momentum"], steps["Adam"])


class FigureTests(unittest.TestCase):
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
