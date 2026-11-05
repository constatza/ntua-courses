"""Objectives used by the optimization figures.

Each problem exposes ``value`` and ``gradient`` and, when sampling matters, a
stochastic gradient oracle with the signature required by ``optimizers.run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


def rotated_hessian(eigenvalues: tuple[float, float], angle_degrees: float) -> Matrix:
    angle = np.deg2rad(angle_degrees)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return rotation @ np.diag(eigenvalues) @ rotation.T


@dataclass(frozen=True, slots=True)
class Quadratic:
    """L(theta) = 1/2 theta' H theta, minimizer 0, minimum value 0."""

    hessian: Matrix

    def value(self, parameters: Vector) -> float:
        return float(0.5 * parameters @ self.hessian @ parameters)

    def gradient(self, parameters: Vector) -> Vector:
        return self.hessian @ parameters

    def oracle(self, parameters: Vector, _rng: np.random.Generator) -> Vector:
        return self.gradient(parameters)

    @property
    def eigenvalues(self) -> Vector:
        return np.linalg.eigvalsh(self.hessian)

    @property
    def condition_number(self) -> float:
        eigenvalues = self.eigenvalues
        return float(eigenvalues[-1] / eigenvalues[0])


# The course-wide quadratic laboratory: kappa = 18, principal axes rotated by 28 degrees.
LAB_EIGENVALUES = (1.0, 18.0)
LAB = Quadratic(rotated_hessian(LAB_EIGENVALUES, 28.0))
LAB_START: Vector = np.array((-3.0, 2.6))


@dataclass(frozen=True, slots=True)
class FiniteSumLab:
    """Least squares whose full-batch objective equals the quadratic laboratory.

    ``L(theta) = (1/2n) ||A theta - b||^2`` with ``A'A/n = H`` and ``A'b = 0``, so the
    minimizer is exactly zero and ``L - L* = 1/2 theta' H theta``. The targets are
    nonzero, so per-sample gradients stay nonzero at the minimizer: mini-batch noise
    is real sampling noise that does not vanish at the optimum.
    """

    features: Matrix
    targets: Vector

    @property
    def size(self) -> int:
        return self.features.shape[0]

    def sample_gradients(self, parameters: Vector) -> Matrix:
        residual = self.features @ parameters - self.targets
        return self.features * residual[:, None]

    def full_gradient(self, parameters: Vector) -> Vector:
        return self.sample_gradients(parameters).mean(axis=0)

    def batch_oracle(self, batch_size: int):
        """Uniform i.i.d. sampling with replacement, so the variance is exactly sigma^2 / b."""

        def oracle(parameters: Vector, rng: np.random.Generator) -> Vector:
            indices = rng.integers(self.size, size=batch_size)
            return self.sample_gradients(parameters)[indices].mean(axis=0)

        return oracle


def make_finite_sum_lab(
    seed: int, samples: int = 256, target_noise: float = 0.6
) -> FiniteSumLab:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(samples, 2))
    raw -= raw.mean(axis=0)
    cholesky = np.linalg.cholesky(raw.T @ raw / samples)
    white = raw @ np.linalg.inv(cholesky).T  # white' white / n = I
    eigenvalues, eigenvectors = np.linalg.eigh(LAB.hessian)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    features = white @ root  # features' features / n = H
    noise = rng.normal(0.0, target_noise, size=samples)
    projection = features @ np.linalg.solve(features.T @ features, features.T)
    targets = noise - projection @ noise  # orthogonal to every feature column
    return FiniteSumLab(features, targets)


@dataclass(frozen=True, slots=True)
class DoubleWell:
    """f(x, y) = (x^2 - 1)^2 + 0.3 x + y^2 / 2: two minima, one saddle, one global."""

    tilt: float = 0.3

    def value(self, parameters: Vector) -> float:
        x, y = parameters
        return float((x**2 - 1.0) ** 2 + self.tilt * x + 0.5 * y**2)

    def gradient(self, parameters: Vector) -> Vector:
        x, y = parameters
        return np.array((4.0 * x * (x**2 - 1.0) + self.tilt, y))

    def hessian(self, parameters: Vector) -> Matrix:
        x, _ = parameters
        return np.diag((12.0 * x**2 - 4.0, 1.0))

    def oracle(self, parameters: Vector, _rng: np.random.Generator) -> Vector:
        return self.gradient(parameters)

    def stationary_points(self) -> tuple[Vector, ...]:
        roots = np.roots([4.0, 0.0, -4.0, self.tilt]).real
        return tuple(np.array((x, 0.0)) for x in sorted(roots))


@dataclass(frozen=True, slots=True)
class Rosenbrock:
    """f(x, y) = (1 - x)^2 + c (y - x^2)^2, a curved narrow valley with minimum at (1, 1)."""

    curvature: float = 10.0

    def value(self, parameters: Vector) -> float:
        x, y = parameters
        return float((1.0 - x) ** 2 + self.curvature * (y - x**2) ** 2)

    def gradient(self, parameters: Vector) -> Vector:
        x, y = parameters
        return np.array(
            (
                -2.0 * (1.0 - x) - 4.0 * self.curvature * x * (y - x**2),
                2.0 * self.curvature * (y - x**2),
            )
        )

    def oracle(self, parameters: Vector, _rng: np.random.Generator) -> Vector:
        return self.gradient(parameters)


@dataclass(frozen=True, slots=True)
class StyblinskiTang:
    """f(x) = 1/2 sum_i (x_i^4 - 16 x_i^2 + 5 x_i): four minima in 2-D, one of them global.

    A standard multimodal benchmark: each coordinate has a deep and a shallow well, so the
    four combinations are separated by saddles and the global minimum is the deep-deep one.
    """

    def value(self, parameters: Vector) -> float:
        return float(0.5 * np.sum(parameters**4 - 16.0 * parameters**2 + 5.0 * parameters))

    def gradient(self, parameters: Vector) -> Vector:
        return 2.0 * parameters**3 - 16.0 * parameters + 2.5

    def oracle(self, parameters: Vector, _rng: np.random.Generator) -> Vector:
        return self.gradient(parameters)

    @property
    def wells(self) -> tuple[float, float]:
        """(deep, shallow) coordinate of the one-dimensional minima."""
        roots = np.sort(np.roots([2.0, 0.0, -16.0, 2.5]).real)
        return float(roots[0]), float(roots[2])


class Basin(str, Enum):
    GLOBAL = "global minimum"
    MIXED = "deep in one coordinate"
    SHALLOW = "shallow in both"
    UNSETTLED = "not converged"


BASIN_TOLERANCE = 0.5


def classify_basin(problem: StyblinskiTang, point: Vector) -> Basin:
    deep, shallow = problem.wells
    in_deep = np.abs(point - deep) < BASIN_TOLERANCE
    in_shallow = np.abs(point - shallow) < BASIN_TOLERANCE
    if not np.all(in_deep | in_shallow):
        return Basin.UNSETTLED
    match int(in_deep.sum()):
        case 2:
            return Basin.GLOBAL
        case 1:
            return Basin.MIXED
        case _:
            return Basin.SHALLOW


@dataclass(frozen=True, slots=True)
class MlpRegression:
    """A small tanh network ``input -> HIDDEN -> HIDDEN -> 1`` fitted to noisy samples.

    The target is ``sin(2 x1) + sin(3 u)`` where the second input is ``x2 = second_scale * u``.
    A small ``second_scale`` gives the first layer weights of very different gradient size
    (the weights on ``x2`` see gradients ``second_scale`` times smaller), which is the situation
    per-parameter scaling is meant for. All parameters live in one flat vector so that the
    optimizers of ``optimizers.py`` apply unchanged.
    """

    train_inputs: Matrix
    train_targets: Vector
    test_inputs: Matrix
    test_targets: Vector
    hidden: int = 24

    @property
    def size(self) -> int:
        h = self.hidden
        return 2 * h + h + h * h + h + h + 1

    def _unpack(self, flat: Vector) -> tuple[Matrix, Vector, Matrix, Vector, Matrix, Vector]:
        h = self.hidden
        cuts = np.cumsum((2 * h, h, h * h, h, h))
        w1, b1, w2, b2, w3, b3 = np.split(flat, cuts)
        return w1.reshape(2, h), b1, w2.reshape(h, h), b2, w3.reshape(h, 1), b3

    def initial_parameters(self, rng: np.random.Generator) -> Vector:
        h = self.hidden
        pieces = (
            rng.normal(0.0, 1.0, 2 * h),
            np.zeros(h),
            rng.normal(0.0, 1.0 / np.sqrt(h), h * h),
            np.zeros(h),
            rng.normal(0.0, 1.0 / np.sqrt(h), h),
            np.zeros(1),
        )
        return np.concatenate(pieces)

    def loss_and_gradient(
        self, flat: Vector, inputs: Matrix, targets: Vector
    ) -> tuple[float, Vector]:
        """Half mean squared error and its gradient by backpropagation."""
        w1, b1, w2, b2, w3, b3 = self._unpack(flat)
        a1 = np.tanh(inputs @ w1 + b1)
        a2 = np.tanh(a1 @ w2 + b2)
        residual = (a2 @ w3)[:, 0] + b3[0] - targets
        count = targets.size
        d3 = (residual / count)[:, None]
        d2 = (d3 @ w3.T) * (1.0 - a2**2)
        d1 = (d2 @ w2.T) * (1.0 - a1**2)
        gradient = np.concatenate(
            (
                (inputs.T @ d1).ravel(), d1.sum(axis=0),
                (a1.T @ d2).ravel(), d2.sum(axis=0),
                (a2.T @ d3).ravel(), d3.sum(axis=0),
            )
        )
        return float(0.5 * np.mean(residual**2)), gradient

    def test_loss(self, flat: Vector) -> float:
        loss = self.loss_and_gradient(flat, self.test_inputs, self.test_targets)[0]
        return loss if np.isfinite(loss) else float("inf")

    def batch_oracle(self, batch_size: int):
        def oracle(flat: Vector, rng: np.random.Generator) -> Vector:
            indices = rng.integers(self.train_targets.size, size=batch_size)
            return self.loss_and_gradient(
                flat, self.train_inputs[indices], self.train_targets[indices]
            )[1]

        return oracle


NOISE_STD = 0.1


def make_mlp_regression(
    second_scale: float, seed: int, train: int = 256, test: int = 512
) -> MlpRegression:
    """Noisy samples of the two-input target; ``second_scale`` sets the size of input two."""
    rng = np.random.default_rng(seed)

    def draw(count: int) -> tuple[Matrix, Vector]:
        first = rng.uniform(-3.0, 3.0, count)
        second = rng.uniform(-1.0, 1.0, count)
        targets = np.sin(2.0 * first) + np.sin(3.0 * second) + NOISE_STD * rng.normal(size=count)
        return np.stack((first, second_scale * second), axis=1), targets

    train_inputs, train_targets = draw(train)
    test_inputs, test_targets = draw(test)
    return MlpRegression(train_inputs, train_targets, test_inputs, test_targets)
