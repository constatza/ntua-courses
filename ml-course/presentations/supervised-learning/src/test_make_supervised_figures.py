from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from make_supervised_figures import (
    EXPECTED_FILENAMES,
    POLYNOMIAL_DEGREES,
    SEED,
    PREVALENCE,
    SCORE_SEPARATION,
    generate_figures,
    polynomial_sample,
    two_class_sample,
)
from supervised_math import (
    GaussianScores,
    augment,
    cross_entropy,
    cross_entropy_gradient,
    expected_cost,
    fit_logistic,
    least_squares,
    polynomial_design,
    standardize,
    trapezoid,
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FourPointExampleTests(unittest.TestCase):
    """The hand-worked slide: (0,1), (1,2), (2,2), (3,4) gives w = b = 0.9."""

    def setUp(self) -> None:
        self.x = np.array([0.0, 1.0, 2.0, 3.0])
        self.y = np.array([1.0, 2.0, 2.0, 4.0])
        self.design = augment(self.x[:, None])

    def test_normal_equations_give_the_values_on_the_slide(self) -> None:
        np.testing.assert_allclose(self.design.T @ self.design, [[14, 6], [6, 4]])
        np.testing.assert_allclose(self.design.T @ self.y, [18, 9])
        np.testing.assert_allclose(np.linalg.solve(self.design.T @ self.design, self.design.T @ self.y), [0.9, 0.9])
        np.testing.assert_allclose(least_squares(self.design, self.y), [0.9, 0.9])

    def test_residuals_are_orthogonal_to_every_column_and_mse_is_0_175(self) -> None:
        beta = least_squares(self.design, self.y)
        residual = self.y - self.design @ beta
        np.testing.assert_allclose(residual, [0.1, 0.2, -0.7, 0.4], atol=1e-12)
        np.testing.assert_allclose(self.design.T @ residual, np.zeros(2), atol=1e-12)
        self.assertAlmostEqual(float(np.mean(residual**2)), 0.175)
        prediction = self.design @ beta
        self.assertAlmostEqual(float(self.y @ self.y), float(prediction @ prediction + residual @ residual))


class LeastSquaresClaimTests(unittest.TestCase):
    def test_normal_equations_square_the_condition_number(self) -> None:
        design = polynomial_design(np.linspace(-1.0, 1.0, 30), 5)
        self.assertAlmostEqual(
            np.linalg.cond(design.T @ design) / np.linalg.cond(design) ** 2, 1.0, delta=0.05
        )

    def test_mean_minimizes_squared_error_and_median_minimizes_absolute_error(self) -> None:
        targets = np.array([1.0, 2.0, 2.5, 3.0, 20.0])
        candidates = np.linspace(0.0, 25.0, 25_001)
        squared = ((targets[None, :] - candidates[:, None]) ** 2).mean(axis=1)
        absolute = np.abs(targets[None, :] - candidates[:, None]).mean(axis=1)
        self.assertAlmostEqual(candidates[squared.argmin()], targets.mean(), places=3)
        self.assertAlmostEqual(candidates[absolute.argmin()], np.median(targets), places=3)

    def test_polynomial_design_is_still_linear_in_the_coefficients(self) -> None:
        x = np.linspace(-1.0, 1.0, 20)
        design = polynomial_design(x, 3)
        coefficients = np.array([0.5, -1.0, 2.0, 0.25])
        np.testing.assert_allclose(design @ coefficients, np.polyval(coefficients[::-1], x))


class LogisticClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        features, self.labels = two_class_sample(np.random.default_rng(SEED))
        self.design = augment(features)

    def test_gradient_formula_matches_finite_differences(self) -> None:
        beta = np.array([0.4, -0.3, 0.1])
        numerical = np.empty(3)
        for index in range(3):
            offset = np.zeros(3)
            offset[index] = 1e-6
            numerical[index] = (
                cross_entropy(self.design, self.labels, beta + offset)
                - cross_entropy(self.design, self.labels, beta - offset)
            ) / 2e-6
        np.testing.assert_allclose(
            cross_entropy_gradient(self.design, self.labels, beta), numerical, rtol=1e-5, atol=1e-7
        )

    def test_newton_reaches_a_stationary_point_so_the_figure_shows_a_fitted_model(self) -> None:
        beta = fit_logistic(self.design, self.labels)
        np.testing.assert_allclose(cross_entropy_gradient(self.design, self.labels, beta), 0.0, atol=1e-9)

    def test_cost_sensitive_threshold_matches_the_boxed_rule(self) -> None:
        for cost_fp, cost_fn in ((1.0, 1.0), (1.0, 5.0), (4.0, 1.0)):
            threshold = cost_fp / (cost_fp + cost_fn)
            for probability in np.linspace(0.01, 0.99, 99):
                predict_positive = expected_cost(probability, True, cost_fp, cost_fn) <= expected_cost(
                    probability, False, cost_fp, cost_fn
                )
                if abs(probability - threshold) > 1e-9:
                    self.assertEqual(predict_positive, probability > threshold)


class LeakageExampleTests(unittest.TestCase):
    def test_statistics_over_all_data_hide_a_shifted_test_point(self) -> None:
        train, test = np.array([1.0, 2.0, 3.0, 4.0]), 10.0
        everything = np.append(train, test)
        leaked = standardize(np.array([test]), everything.mean(), everything.std())[0]
        honest = standardize(np.array([test]), train.mean(), train.std())[0]
        self.assertAlmostEqual(everything.mean(), 4.0)
        self.assertAlmostEqual(train.mean(), 2.5)
        self.assertAlmostEqual(leaked, 1.897, places=3)
        self.assertAlmostEqual(honest, 6.708, places=3)


class CurveTests(unittest.TestCase):
    def test_roc_area_matches_the_closed_form_and_pr_baseline_is_the_prevalence(self) -> None:
        model = GaussianScores(SCORE_SEPARATION, PREVALENCE)
        thresholds = np.linspace(10.0, -8.0, 4000)
        area = trapezoid(model.true_positive_rate(thresholds), model.false_positive_rate(thresholds))
        self.assertAlmostEqual(area, model.auc(), places=3)
        self.assertAlmostEqual(model.auc(), 0.9214, places=3)
        far_below = np.array([-30.0])  # every example predicted positive
        self.assertAlmostEqual(float(model.precision(far_below)[0]), PREVALENCE, places=6)


class PolynomialTests(unittest.TestCase):
    def test_training_error_falls_with_degree_but_the_flexible_fit_generalizes_worst(self) -> None:
        sample = polynomial_sample(np.random.default_rng(SEED))
        errors = {degree: sample.rmse(degree) for degree in POLYNOMIAL_DEGREES}
        train = [errors[d][0] for d in POLYNOMIAL_DEGREES]
        self.assertEqual(train, sorted(train, reverse=True))
        self.assertGreater(errors[9][1], 3 * errors[3][1])
        self.assertLess(errors[3][1], errors[1][1])


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
