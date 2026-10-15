# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy"]
# ///

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntroductionMathTest(unittest.TestCase):
    def test_sigmoid_maps_zero_to_half_and_preserves_shape(self) -> None:
        figures = load_module(
            "introduction_figures", "presentations/01-ml-introduction/src/make_figures.py"
        )
        values = np.array([-2.0, 0.0, 2.0])
        result = figures.sigmoid(values)
        self.assertEqual(result.shape, values.shape)
        self.assertAlmostEqual(float(result[1]), 0.5)
        np.testing.assert_allclose(result[0], 1.0 - result[2])


class ArchitectureMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.figures = load_module(
            "architecture_figures", "presentations/02-neural-architectures/src/make_figures.py"
        )

    def test_cross_correlation_matches_hand_calculation(self) -> None:
        image = np.array(
            [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]]
        )
        kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
        expected = np.array([[0.0, -1.0], [-1.0, 1.0]])
        np.testing.assert_allclose(self.figures.cross_correlate2d(image, kernel), expected)

    def test_attention_is_row_normalized_and_matches_uniform_case(self) -> None:
        queries = np.zeros((2, 2))
        keys = np.zeros((3, 2))
        values = np.array([[1.0, 0.0], [3.0, 2.0], [2.0, 4.0]])
        output, weights = self.figures.scaled_dot_product_attention(queries, keys, values)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(2))
        np.testing.assert_allclose(output, np.array([[2.0, 2.0], [2.0, 2.0]]))

    def test_convolution_output_size_uses_floor(self) -> None:
        self.assertEqual(
            self.figures.convolution_output_size(size=8, kernel=3, stride=2, padding=1),
            4,
        )


class TrainingMathTest(unittest.TestCase):
    def test_two_layer_gradient_matches_finite_difference(self) -> None:
        figures = load_module(
            "training_figures", "presentations/03-neural-training/src/make_figures.py"
        )
        x = np.array([0.4, -0.7])
        target = np.array([0.3])
        w1 = np.array([[0.2, -0.3], [0.5, 0.1]])
        b1 = np.array([0.1, -0.2])
        w2 = np.array([[0.6, -0.4]])
        b2 = np.array([0.05])

        gradients = figures.two_layer_gradients(x, target, w1, b1, w2, b2)
        epsilon = 1e-6
        numerical = np.empty_like(w1)
        for row, column in np.ndindex(w1.shape):
            plus = w1.copy()
            minus = w1.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numerical[row, column] = (
                figures.two_layer_loss(x, target, plus, b1, w2, b2)
                - figures.two_layer_loss(x, target, minus, b1, w2, b2)
            ) / (2.0 * epsilon)

        np.testing.assert_allclose(gradients["w1"], numerical, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
