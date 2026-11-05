"""Closed-form and Newton computations behind the supervised-learning figures and slides."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


def augment(features: Matrix) -> Matrix:
    """Append the column of ones that carries the intercept."""
    return np.hstack((features, np.ones((features.shape[0], 1))))


def least_squares(design: Matrix, targets: Vector) -> Vector:
    """Minimizer of ||X beta - y||^2 via QR, never forming an inverse."""
    q, r = np.linalg.qr(design)
    return np.linalg.solve(r, q.T @ targets)


def polynomial_design(x: Vector, degree: int) -> Matrix:
    return np.vander(x, degree + 1, increasing=True)


def sigmoid(scores: Vector) -> Vector:
    return np.exp(-np.logaddexp(0.0, -scores))


def cross_entropy(design: Matrix, labels: Vector, beta: Vector) -> float:
    scores = design @ beta
    return float(np.mean(np.logaddexp(0.0, scores) - labels * scores))


def cross_entropy_gradient(design: Matrix, labels: Vector, beta: Vector) -> Vector:
    """(1/n) X' (p_hat - y): the same 'prediction minus target' form as least squares."""
    return design.T @ (sigmoid(design @ beta) - labels) / design.shape[0]


def fit_logistic(design: Matrix, labels: Vector, iterations: int = 50) -> Vector:
    """Newton's method; the objective is convex but has no closed-form minimizer."""
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        probabilities = sigmoid(design @ beta)
        weights = probabilities * (1.0 - probabilities)
        hessian = (design * weights[:, None]).T @ design / design.shape[0]
        beta = beta - np.linalg.solve(hessian, cross_entropy_gradient(design, labels, beta))
    return beta


def normal_cdf(z: float | Vector) -> float | Vector:
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class GaussianScores:
    """Scores N(0, 1) for negatives and N(separation, 1) for positives."""

    separation: float
    prevalence: float

    def true_positive_rate(self, threshold: Vector) -> Vector:
        return 1.0 - normal_cdf(threshold - self.separation)

    def false_positive_rate(self, threshold: Vector) -> Vector:
        return 1.0 - normal_cdf(threshold)

    def precision(self, threshold: Vector) -> Vector:
        positives = self.prevalence * self.true_positive_rate(threshold)
        negatives = (1.0 - self.prevalence) * self.false_positive_rate(threshold)
        return positives / (positives + negatives)

    def auc(self) -> float:
        """P(score of a random positive exceeds that of a random negative)."""
        return float(normal_cdf(self.separation / math.sqrt(2.0)))


def trapezoid(y: Vector, x: Vector) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def standardize(values: Vector, mean: float, std: float) -> Vector:
    return (values - mean) / std


def expected_cost(probability: float, predict_positive: bool, cost_fp: float, cost_fn: float) -> float:
    """Expected cost of one decision when the true P(y=1) is ``probability``."""
    return cost_fp * (1.0 - probability) if predict_positive else cost_fn * probability
