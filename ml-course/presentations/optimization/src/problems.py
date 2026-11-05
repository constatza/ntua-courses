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
