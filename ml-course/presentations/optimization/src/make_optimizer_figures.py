from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from optimizers import (
    Constant,
    GradientOracle,
    OptimizerConfig,
    OptimizerName,
    PowerDecay,
    WarmupCosine,
    run,
)
from problems import (
    LAB,
    LAB_START,
    Basin,
    DoubleWell,
    Quadratic,
    Rosenbrock,
    StyblinskiTang,
    classify_basin,
    make_finite_sum_lab,
    rotated_hessian,
)

SEED = 20261029
FIGURE_DPI = 220
NAVY = "#10284b"
BLUE = "#3f7cac"
ORANGE = "#d97706"
GREEN = "#27864a"
PURPLE = "#7656a3"
RED = "#b5473c"
GRAY = "#667085"

METHOD_COLORS = {
    OptimizerName.GD: ORANGE,
    OptimizerName.MOMENTUM: GREEN,
    OptimizerName.NESTEROV: NAVY,
    OptimizerName.ADAGRAD: BLUE,
    OptimizerName.RMSPROP: PURPLE,
    OptimizerName.ADAM: RED,
}

PROTOCOL_STEPS = 200
TAIL_WINDOW = 20  # tuning scores the worst gap in the last updates so an oscillating method cannot win on lucky phase
LEARNING_RATE_GRID = np.geomspace(1e-3, 10.0, 49)
MOMENTUM_GRID = (0.5, 0.7, 0.9)
RMSPROP_DECAY = 0.9
DISPLAY_FLOOR = 1e-15  # below this a printed value is rounding noise, not a result
PLOT_FLOOR = 1e-16

EXPECTED_FILENAMES = (
    "landscapes.png",
    "objective-geometry.png",
    "learning-rates.png",
    "contraction.png",
    "conditioning.png",
    "gd-vs-sgd.png",
    "schedules.png",
    "momentum.png",
    "diagonal-scaling.png",
    "optimizer-comparison.png",
    "optimizer-progress.png",
    "protocol-table.tex",
    "rosenbrock.png",
    "multimodal.png",
    "saddle-escape.png",
)

Trajectory = NDArray[np.float64]


# --------------------------------------------------------------------------- tuning


@dataclass(frozen=True, slots=True)
class TunedMethod:
    config: OptimizerConfig
    final_gap: float
    final_gradient_norm: float


def _config_candidates(name: OptimizerName) -> tuple[tuple[float, float], ...]:
    """(learning rate, decay) pairs searched for one method; decay is beta or rho."""
    match name:
        case OptimizerName.MOMENTUM | OptimizerName.NESTEROV:
            decays = MOMENTUM_GRID
        case OptimizerName.RMSPROP:
            decays = (RMSPROP_DECAY,)
        case _:
            decays = (0.9,)
    return tuple((float(rate), decay) for decay in decays for rate in LEARNING_RATE_GRID)


def _make_config(name: OptimizerName, rate: float, decay: float) -> OptimizerConfig:
    if name is OptimizerName.RMSPROP:
        return OptimizerConfig(name, Constant(rate), second_decay=decay)
    return OptimizerConfig(name, Constant(rate), first_decay=decay)


def tune_on_quadratic(
    problem: Quadratic,
    name: OptimizerName,
    steps: int = PROTOCOL_STEPS,
    start: NDArray[np.float64] = LAB_START,
) -> TunedMethod:
    """Pick the constant learning rate (and momentum beta) with the smallest final gap."""
    rng = np.random.default_rng(SEED)
    best: TunedMethod | None = None
    best_score = np.inf
    with np.errstate(all="ignore"):
        for rate, decay in _config_candidates(name):
            config = _make_config(name, rate, decay)
            trajectory = run(config, start, problem.oracle, steps, rng)
            score = max(problem.value(point) for point in trajectory[-TAIL_WINDOW:])
            if np.isfinite(score) and (best is None or score < best_score):
                best_score = score
                final = trajectory[-1]
                best = TunedMethod(
                    config, problem.value(final), float(np.linalg.norm(problem.gradient(final)))
                )
    assert best is not None, f"no stable learning rate found for {name.value}"
    return best


@cache
def tuned_protocol() -> dict[OptimizerName, TunedMethod]:
    return {name: tune_on_quadratic(LAB, name) for name in OptimizerName}


def _scientific(value: float) -> str:
    if value < DISPLAY_FLOOR:
        return rf"$<10^{{{int(np.log10(DISPLAY_FLOOR))}}}$"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def protocol_table(tuned: dict[OptimizerName, TunedMethod]) -> str:
    rows = []
    for name, method in tuned.items():
        config = method.config
        decay = config.second_decay if name is OptimizerName.RMSPROP else config.first_decay
        decay_text = "--" if name in (OptimizerName.GD, OptimizerName.ADAGRAD) else f"{decay:g}"
        if name is OptimizerName.ADAM:
            decay_text = "0.9 / 0.999"
        rows.append(
            f"{name.value} & ${config.schedule.rate:.3g}$ & {decay_text} & "
            f"{_scientific(method.final_gap)} & {_scientific(method.final_gradient_norm)} \\\\"
        )
    header = (
        r"\begin{tabular}{lrrrr}" + "\n" + r"\toprule" + "\n"
        r"Method & $\eta$ & $\beta$, $\rho$ & $L-L^\star$ & $\|\nabla L\|_2$ \\" + "\n"
        r"\midrule" + "\n"
    )
    return header + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"


# --------------------------------------------------------------------------- plotting helpers


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _clean(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10)


def _quadratic_axis(axis: plt.Axes, problem: Quadratic = LAB) -> None:
    horizontal, vertical = np.meshgrid(np.linspace(-4.2, 4.2, 360), np.linspace(-3.6, 3.6, 320))
    points = np.stack((horizontal, vertical), axis=-1)
    values = 0.5 * np.einsum("...i,ij,...j->...", points, problem.hessian, points)
    axis.contour(
        horizontal, vertical, values, levels=np.geomspace(0.35, 110.0, 13),
        colors=BLUE, linewidths=1.0, alpha=0.7,
    )
    axis.scatter(0.0, 0.0, s=80, color=GREEN, edgecolor="white", linewidth=1.0, zorder=8)
    axis.set_xlim(-4.2, 4.2)
    axis.set_ylim(-3.6, 3.6)
    axis.set_xlabel(r"parameter $\theta_1$", fontsize=13)
    axis.set_ylabel(r"parameter $\theta_2$", fontsize=13)
    axis.set_aspect("equal", adjustable="box")
    _clean(axis)


def _path(
    axis: plt.Axes, trajectory: Trajectory, label: str, color: str, marker: str = "o"
) -> None:
    axis.plot(
        trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2.0, marker=marker,
        markersize=3.4, markevery=max(1, len(trajectory) // 14), label=label, zorder=6,
    )
    axis.scatter(*trajectory[0], s=64, color=color, edgecolor="white", linewidth=0.8, zorder=7)


def _gap_axis(axis: plt.Axes, steps: int, bottom: float = PLOT_FLOOR, top: float = 1e3) -> None:
    axis.set_yscale("log")
    axis.set_xlim(0, steps)
    axis.set_ylim(bottom, top)
    axis.set_xlabel("update", fontsize=13)
    axis.set_ylabel(r"objective gap $L(\boldsymbol{\theta}_t)-L^\star$", fontsize=13)
    axis.grid(alpha=0.2)
    _clean(axis)


def _gaps(values: Callable[[NDArray[np.float64]], float], trajectory: Trajectory) -> NDArray[np.float64]:
    return np.maximum([values(point) for point in trajectory], PLOT_FLOOR)


def _config(name: OptimizerName, rate: float, decay: float = 0.9) -> OptimizerConfig:
    return _make_config(name, rate, decay)


def _deterministic(
    config: OptimizerConfig, problem: Quadratic, steps: int, start: Trajectory = LAB_START
) -> Trajectory:
    return run(config, start, problem.oracle, steps, np.random.default_rng(SEED))


# --------------------------------------------------------------------------- figures


def _landscapes(path: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.8, 4.9))
    _quadratic_axis(left)
    left.set_title("convex: one valley", color=NAVY, fontsize=15, weight="bold")
    left.annotate("minimum", (0.0, 0.0), (0.7, 0.9), color=GREEN, fontsize=12, weight="bold")

    well = DoubleWell()
    horizontal, vertical = np.meshgrid(np.linspace(-1.9, 1.9, 300), np.linspace(-1.7, 1.7, 300))
    values = (horizontal**2 - 1.0) ** 2 + well.tilt * horizontal + 0.5 * vertical**2
    right.contour(horizontal, vertical, values, levels=np.linspace(-0.3, 4.5, 22),
                  colors=BLUE, linewidths=1.0, alpha=0.7)
    minima, saddles = [], []
    for point in well.stationary_points():
        eigenvalues = np.linalg.eigvalsh(well.hessian(point))
        (minima if eigenvalues[0] > 0 else saddles).append(point)
    for point in minima:
        right.scatter(*point, s=80, color=GREEN, edgecolor="white", zorder=8)
    for point in saddles:
        right.scatter(*point, s=110, color=RED, marker="X", edgecolor="white", zorder=8)
    global_min = min(minima, key=well.value)
    local_min = max(minima, key=well.value)
    right.annotate("global minimum", global_min, (-1.8, -1.3), color=GREEN, fontsize=11, weight="bold")
    right.annotate("local minimum", local_min, (0.55, -1.3), color=GREEN, fontsize=11, weight="bold")
    right.annotate("saddle", saddles[0], (-0.45, 1.3), color=RED, fontsize=11, weight="bold")
    config = OptimizerConfig(OptimizerName.GD, Constant(0.04))
    for start, color in (((-0.6, 1.4), GRAY), ((0.5, 1.4), ORANGE)):
        trajectory = run(config, np.array(start), well.oracle, 200, np.random.default_rng(SEED))
        right.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2.0, zorder=6)
        right.scatter(*trajectory[0], s=56, color=color, edgecolor="white", zorder=7)
    right.set_title("nonconvex: start decides the outcome", color=NAVY, fontsize=15, weight="bold")
    right.set_xlabel(r"parameter $\theta_1$", fontsize=13)
    right.set_ylabel(r"parameter $\theta_2$", fontsize=13)
    right.set_aspect("equal", adjustable="box")
    _clean(right)
    figure.subplots_adjust(wspace=0.28)
    _save(figure, path)


def _objective_geometry(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    _quadratic_axis(axis)
    point = np.array((1.5, 1.55))
    direction = LAB.gradient(point) / np.linalg.norm(LAB.gradient(point))
    axis.scatter(*point, s=85, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=8)
    for sign, color in ((1.0, RED), (-1.0, GREEN)):
        axis.arrow(point[0], point[1], sign * 0.9 * direction[0], sign * 0.9 * direction[1],
                   width=0.025, head_width=0.17, color=color, length_includes_head=True, zorder=9)
    axis.text(point[0] + 0.45, point[1] + 0.75, r"$\nabla L$", color=RED, fontsize=15)
    axis.text(point[0] - 1.35, point[1] - 0.7, r"$-\nabla L$", color=GREEN, fontsize=15)
    _save(figure, path)


def _learning_rates(path: Path) -> None:
    lambda_max = LAB.eigenvalues[-1]
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    settings = (
        (0.008, "too small", GRAY),
        (0.075, "useful", GREEN),
        (1.05 * 2.0 / lambda_max, r"just above $2/\lambda_{\max}$", RED),
    )
    for rate, label, color in settings:
        trajectory = _deterministic(OptimizerConfig(OptimizerName.GD, Constant(rate)), LAB, 36)
        axis.semilogy([LAB.value(p) for p in trajectory], color=color, linewidth=3.0,
                      label=rf"$\eta={rate:.3f}$: {label}")
    axis.set_xlabel("update", fontsize=14)
    axis.set_ylabel(r"objective $L(\boldsymbol{\theta}_t)$", fontsize=14)
    axis.set_title(rf"$\lambda_{{\max}}={lambda_max:.0f}$, stability limit "
                   rf"$2/\lambda_{{\max}}={2.0 / lambda_max:.3f}$", fontsize=13, color=NAVY)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=12, loc="upper left")
    _clean(axis)
    _save(figure, path)


def contraction_factor(rate: float, eigenvalues: Sequence[float]) -> float:
    return max(abs(1.0 - rate * value) for value in eigenvalues)


def optimal_gd_rate(lambda_min: float, lambda_max: float) -> float:
    return 2.0 / (lambda_max + lambda_min)


def _contraction(path: Path) -> None:
    lambda_min, lambda_max = LAB.eigenvalues
    rates = np.linspace(0.0, 0.15, 600)
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.plot(rates, np.abs(1 - rates * lambda_min), color=GRAY, linewidth=1.8,
              label=rf"$|1-\eta\lambda_{{\min}}|$, $\lambda_{{\min}}={lambda_min:.0f}$")
    axis.plot(rates, np.abs(1 - rates * lambda_max), color=BLUE, linewidth=1.8,
              label=rf"$|1-\eta\lambda_{{\max}}|$, $\lambda_{{\max}}={lambda_max:.0f}$")
    axis.plot(rates, [contraction_factor(r, (lambda_min, lambda_max)) for r in rates],
              color=ORANGE, linewidth=3.4, label=r"$\rho(\eta)=\max_j|1-\eta\lambda_j|$")
    best = optimal_gd_rate(lambda_min, lambda_max)
    axis.axhline(1.0, color=RED, linewidth=1.2, linestyle=":")
    axis.scatter([best], [(lambda_max - lambda_min) / (lambda_max + lambda_min)], s=90,
                 color=GREEN, zorder=8)
    axis.annotate(rf"$\eta^\star=2/(\lambda_{{\min}}+\lambda_{{\max}})$" "\n"
                  rf"$\rho^\star=(\kappa-1)/(\kappa+1)={(lambda_max - lambda_min) / (lambda_max + lambda_min):.2f}$",
                  (best, 0.89), (0.085, 0.3), color=GREEN, fontsize=13,
                  arrowprops={"arrowstyle": "->", "color": GREEN})
    axis.text(0.147, 1.03, "diverges", color=RED, fontsize=12, ha="right")
    axis.set_xlim(0.0, 0.15)
    axis.set_ylim(0.0, 1.5)
    axis.set_xlabel(r"learning rate $\eta$", fontsize=14)
    axis.set_ylabel("error contraction per update", fontsize=14)
    axis.legend(frameon=False, fontsize=11, loc="upper center")
    axis.grid(alpha=0.2)
    _clean(axis)
    _save(figure, path)


CONDITIONING_RATE = 0.10
CONDITIONING_STEPS = 25


def _conditioning(path: Path) -> None:
    mean_curvature = float(LAB.eigenvalues.mean())
    round_problem = Quadratic(mean_curvature * np.eye(2))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))
    for axis, problem, title in (
        (axes[0], round_problem, r"$\kappa=1$"),
        (axes[1], LAB, rf"$\kappa={LAB.condition_number:.0f}$"),
    ):
        _quadratic_axis(axis, problem)
        config = OptimizerConfig(OptimizerName.GD, Constant(CONDITIONING_RATE))
        rho = contraction_factor(CONDITIONING_RATE, problem.eigenvalues)
        _path(axis, _deterministic(config, problem, CONDITIONING_STEPS),
              rf"GD, $\eta={CONDITIONING_RATE}$, $\rho={rho:.2f}$", ORANGE)
        axis.set_title(title, color=NAVY, fontsize=15, weight="bold")
        axis.legend(frameon=False, fontsize=11, loc="lower right")
    figure.subplots_adjust(wspace=0.28)
    _save(figure, path)


SGD_RATE = 0.02
SGD_STEPS = 600
SGD_BATCH_SIZES = (1, 16)


def _gd_vs_sgd(path: Path) -> None:
    lab = make_finite_sum_lab(SEED)
    runs: list[tuple[str, str, Trajectory]] = [
        ("full-batch GD", ORANGE,
         _deterministic(OptimizerConfig(OptimizerName.GD, Constant(SGD_RATE)), LAB, SGD_STEPS)),
    ]
    for batch_size, color in zip(SGD_BATCH_SIZES, (GRAY, GREEN), strict=True):
        trajectory = run(OptimizerConfig(OptimizerName.GD, Constant(SGD_RATE)), LAB_START,
                         lab.batch_oracle(batch_size), SGD_STEPS, np.random.default_rng(SEED))
        runs.append((f"mini-batch $b={batch_size}$", color, trajectory))
    figure, (paths, gaps) = plt.subplots(1, 2, figsize=(11.2, 4.9))
    _quadratic_axis(paths)
    for label, color, trajectory in runs:
        _path(paths, trajectory, label, color)
        gaps.semilogy(_gaps(LAB.value, trajectory), color=color, linewidth=1.8, label=label)
    _gap_axis(gaps, SGD_STEPS, bottom=1e-8, top=1e3)
    gaps.legend(frameon=False, fontsize=11, loc="upper right")
    figure.subplots_adjust(wspace=0.3)
    _save(figure, path)


def _schedules(path: Path) -> None:
    total, peak = 1000, 0.1
    updates = np.arange(1, total + 1)
    curves = (
        ("constant", Constant(peak), GRAY),
        (r"$\eta_0/\sqrt{t}$", PowerDecay(peak * 4.0, 0.5), BLUE),
        (r"$\eta_0/t$", PowerDecay(peak * 20.0, 1.0), PURPLE),
        ("warm-up + cosine", WarmupCosine(peak, 50, total), GREEN),
    )
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    for label, schedule, color in curves:
        axis.plot(updates, [schedule.at(int(t)) for t in updates], color=color, linewidth=2.6,
                  label=label)
    axis.set_ylim(0.0, 0.13)
    axis.set_xlim(0, total)
    axis.set_xlabel("update $t$", fontsize=14)
    axis.set_ylabel(r"learning rate $\eta_t$", fontsize=14)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=12)
    _clean(axis)
    _save(figure, path)


def heavy_ball_parameters(lambda_min: float, lambda_max: float) -> tuple[float, float]:
    """Polyak's optimal (eta, beta) for ``v = beta v + g``, ``theta -= eta v`` on a quadratic."""
    root_kappa = np.sqrt(lambda_max / lambda_min)
    rate = 4.0 / (np.sqrt(lambda_max) + np.sqrt(lambda_min)) ** 2
    return float(rate), float(((root_kappa - 1.0) / (root_kappa + 1.0)) ** 2)


def nesterov_parameters(lambda_min: float, lambda_max: float) -> tuple[float, float]:
    root_kappa = np.sqrt(lambda_max / lambda_min)
    return 1.0 / lambda_max, float((root_kappa - 1.0) / (root_kappa + 1.0))


MOMENTUM_STEPS = 60


def _momentum(path: Path) -> None:
    lambda_min, lambda_max = LAB.eigenvalues
    kappa = lambda_max / lambda_min
    heavy_rate, heavy_beta = heavy_ball_parameters(lambda_min, lambda_max)
    nesterov_rate, nesterov_beta = nesterov_parameters(lambda_min, lambda_max)
    methods = (
        (rf"GD, $\rho={(kappa - 1) / (kappa + 1):.2f}$",
         OptimizerConfig(OptimizerName.GD, Constant(optimal_gd_rate(lambda_min, lambda_max))), ORANGE),
        (rf"momentum, $\rho\approx\sqrt{{\beta}}={np.sqrt(heavy_beta):.2f}$",
         OptimizerConfig(OptimizerName.MOMENTUM, Constant(heavy_rate), first_decay=heavy_beta), GREEN),
        (rf"Nesterov, $\rho\approx1-1/\sqrt{{\kappa}}={1 - 1 / np.sqrt(kappa):.2f}$",
         OptimizerConfig(OptimizerName.NESTEROV, Constant(nesterov_rate), first_decay=nesterov_beta), NAVY),
    )
    figure, gaps = plt.subplots(figsize=(8.8, 5.2))
    for label, config, color in methods:
        trajectory = _deterministic(config, LAB, MOMENTUM_STEPS)
        gaps.semilogy(_gaps(LAB.value, trajectory), color=color, linewidth=2.6, label=label)
    _gap_axis(gaps, MOMENTUM_STEPS, bottom=1e-14, top=1e3)
    gaps.legend(frameon=False, fontsize=12, loc="lower left")
    _save(figure, path)


DIAGONAL_STEPS = 100


def _diagonal_scaling(path: Path) -> None:
    names = (OptimizerName.GD, OptimizerName.ADAGRAD, OptimizerName.RMSPROP, OptimizerName.ADAM)
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    for axis, angle, title in (
        (axes[0], 0.0, "curvature axes aligned with parameters"),
        (axes[1], 45.0, r"axes rotated by $45^\circ$"),
    ):
        problem = Quadratic(rotated_hessian((1.0, 18.0), angle))
        for name in names:
            tuned = tune_on_quadratic(problem, name, steps=DIAGONAL_STEPS)
            trajectory = _deterministic(tuned.config, problem, DIAGONAL_STEPS)
            axis.semilogy(_gaps(problem.value, trajectory), color=METHOD_COLORS[name],
                          linewidth=2.2, label=name.value)
        _gap_axis(axis, DIAGONAL_STEPS, bottom=1e-14, top=1e3)
        axis.set_title(title, color=NAVY, fontsize=14, weight="bold")
    axes[1].set_ylabel("")
    axes[0].legend(frameon=False, fontsize=11, loc="lower left")
    figure.subplots_adjust(wspace=0.08)
    _save(figure, path)


def _optimizer_comparison(path: Path) -> None:
    tuned = tuned_protocol()
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 7.0), sharex=True, sharey=True)
    for axis, (name, method) in zip(axes.flat, tuned.items(), strict=True):
        _quadratic_axis(axis)
        _path(axis, _deterministic(method.config, LAB, PROTOCOL_STEPS), name.value,
              METHOD_COLORS[name])
        axis.set_title(name.value, color=METHOD_COLORS[name], fontsize=15, weight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
    figure.subplots_adjust(wspace=0.1, hspace=0.25)
    _save(figure, path)


def _optimizer_progress(path: Path) -> None:
    tuned = tuned_protocol()
    figure, (gaps, gradients) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    for name, method in tuned.items():
        trajectory = _deterministic(method.config, LAB, PROTOCOL_STEPS)
        gaps.semilogy(_gaps(LAB.value, trajectory), color=METHOD_COLORS[name], linewidth=2.1,
                      label=name.value)
        gradients.semilogy(
            np.maximum([np.linalg.norm(LAB.gradient(p)) for p in trajectory], PLOT_FLOOR),
            color=METHOD_COLORS[name], linewidth=2.1, label=name.value)
    _gap_axis(gaps, PROTOCOL_STEPS, bottom=1e-15, top=1e3)
    gradients.set_yscale("log")
    gradients.set_xlim(0, PROTOCOL_STEPS)
    gradients.set_ylim(1e-15, 1e2)
    gradients.set_xlabel("update", fontsize=13)
    gradients.set_ylabel(r"gradient norm $\|\nabla L(\boldsymbol{\theta}_t)\|_2$", fontsize=13)
    gradients.grid(alpha=0.2)
    _clean(gradients)
    handles, labels = gaps.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=11)
    figure.subplots_adjust(wspace=0.3, bottom=0.25)
    _save(figure, path)


def _tune_generic(
    problem: Rosenbrock,
    name: OptimizerName,
    start: NDArray[np.float64],
    steps: int,
    grid: NDArray[np.float64],
    reference: float,
) -> tuple[OptimizerConfig, Trajectory]:
    rng = np.random.default_rng(SEED)
    best: tuple[float, OptimizerConfig, Trajectory] | None = None
    with np.errstate(all="ignore"):
        for decay in (MOMENTUM_GRID if name in (OptimizerName.MOMENTUM, OptimizerName.NESTEROV) else (0.9,)):
            for rate in grid:
                config = _make_config(name, float(rate), decay)
                trajectory = run(config, start, problem.oracle, steps, rng)
                score = max(problem.value(point) for point in trajectory[-TAIL_WINDOW:]) - reference
                if np.isfinite(score) and (best is None or score < best[0]):
                    best = (score, config, trajectory)
    assert best is not None
    return best[1], best[2]


ROSENBROCK_STEPS = 400
ROSENBROCK_START = np.array((-1.2, 1.0))


def _rosenbrock(path: Path) -> None:
    problem = Rosenbrock()
    figure, (paths, gaps) = plt.subplots(1, 2, figsize=(11.4, 4.9))
    horizontal, vertical = np.meshgrid(np.linspace(-1.6, 1.6, 320), np.linspace(-0.6, 2.4, 320))
    values = (1 - horizontal) ** 2 + problem.curvature * (vertical - horizontal**2) ** 2
    paths.contour(horizontal, vertical, values, levels=np.geomspace(0.05, 600.0, 14),
                  colors=BLUE, linewidths=1.0, alpha=0.7)
    paths.scatter(1.0, 1.0, s=80, color=GREEN, edgecolor="white", zorder=8)
    for name in (OptimizerName.GD, OptimizerName.MOMENTUM, OptimizerName.ADAM):
        _, trajectory = _tune_generic(problem, name, ROSENBROCK_START, ROSENBROCK_STEPS,
                                      np.geomspace(1e-4, 0.5, 60), 0.0)
        paths.plot(trajectory[:, 0], trajectory[:, 1], color=METHOD_COLORS[name], linewidth=1.8,
                   label=name.value, zorder=6)
        gaps.semilogy(np.maximum([problem.value(p) for p in trajectory], PLOT_FLOOR),
                      color=METHOD_COLORS[name], linewidth=2.1, label=name.value)
    paths.scatter(*ROSENBROCK_START, s=60, color=GRAY, edgecolor="white", zorder=7)
    paths.set_xlabel(r"parameter $\theta_1$", fontsize=13)
    paths.set_ylabel(r"parameter $\theta_2$", fontsize=13)
    paths.set_aspect("equal", adjustable="box")
    paths.legend(frameon=False, fontsize=10, loc="upper left")
    _clean(paths)
    _gap_axis(gaps, ROSENBROCK_STEPS, bottom=1e-14, top=1e2)
    gaps.legend(frameon=False, fontsize=11, loc="lower left")
    figure.subplots_adjust(wspace=0.3)
    _save(figure, path)


MULTIMODAL_STARTS = 300
MULTIMODAL_STEPS = 600
MULTIMODAL_CONFIGS = (
    ("GD", OptimizerConfig(OptimizerName.GD, Constant(0.01)), ORANGE),
    ("Momentum", OptimizerConfig(OptimizerName.MOMENTUM, Constant(0.01), first_decay=0.9), GREEN),
    ("Adam", OptimizerConfig(OptimizerName.ADAM, Constant(0.1)), RED),
)
BASIN_COLORS = {
    Basin.GLOBAL: GREEN,
    Basin.MIXED: ORANGE,
    Basin.SHALLOW: RED,
    Basin.UNSETTLED: GRAY,
}


def multimodal_starts() -> NDArray[np.float64]:
    return np.random.default_rng(SEED).uniform(-4.0, 4.0, size=(MULTIMODAL_STARTS, 2))


def basin_outcomes(config: OptimizerConfig) -> list[Basin]:
    problem = StyblinskiTang()
    rng = np.random.default_rng(SEED)
    return [
        classify_basin(problem, run(config, start, problem.oracle, MULTIMODAL_STEPS, rng)[-1])
        for start in multimodal_starts()
    ]


def _multimodal(path: Path) -> None:
    problem = StyblinskiTang()
    starts = multimodal_starts()
    outcomes = {label: basin_outcomes(config) for label, config, _ in MULTIMODAL_CONFIGS}
    figure, (basins, bars) = plt.subplots(1, 2, figsize=(11.6, 5.0), gridspec_kw={"width_ratios": (1.15, 1.0)})
    horizontal, vertical = np.meshgrid(np.linspace(-4.5, 4.5, 300), np.linspace(-4.5, 4.5, 300))
    values = 0.5 * sum(c**4 - 16.0 * c**2 + 5.0 * c for c in (horizontal, vertical))
    basins.contour(horizontal, vertical, values, levels=18, colors=BLUE, linewidths=0.9, alpha=0.6)
    for basin in Basin:
        mask = [outcome is basin for outcome in outcomes["GD"]]
        if any(mask):
            basins.scatter(starts[mask, 0], starts[mask, 1], s=22, color=BASIN_COLORS[basin],
                           edgecolor="white", linewidth=0.4, label=basin.value, zorder=5)
    deep, shallow = problem.wells
    for x in (deep, shallow):
        for y in (deep, shallow):
            basins.scatter(x, y, s=70, marker="*", color=NAVY, zorder=6)
    basins.set_xlabel(r"parameter $\theta_1$", fontsize=13)
    basins.set_ylabel(r"parameter $\theta_2$", fontsize=13)
    basins.set_title("starting point, coloured by where GD ends", color=NAVY, fontsize=13, weight="bold")
    basins.set_aspect("equal", adjustable="box")
    _clean(basins)
    positions = np.arange(len(MULTIMODAL_CONFIGS))
    bottoms = np.zeros(len(MULTIMODAL_CONFIGS))
    for basin in Basin:
        shares = np.array([np.mean([o is basin for o in outcomes[label]])
                           for label, _, _ in MULTIMODAL_CONFIGS])
        if shares.any():
            bars.bar(positions, shares, bottom=bottoms, color=BASIN_COLORS[basin], label=basin.value)
        bottoms += shares
    bars.set_xticks(positions, [label for label, _, _ in MULTIMODAL_CONFIGS], fontsize=12)
    bars.set_ylabel("share of random starts", fontsize=13)
    bars.set_ylim(0, 1)
    bars.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _clean(bars)
    figure.subplots_adjust(wspace=0.3)
    _save(figure, path)


SADDLE_STEPS = 300
SADDLE_OFFSET = 1e-6
SADDLE_ESCAPE_DISTANCE = 0.3
SADDLE_CONFIGS = (
    ("GD", OptimizerConfig(OptimizerName.GD, Constant(0.02)), ORANGE),
    ("Momentum", OptimizerConfig(OptimizerName.MOMENTUM, Constant(0.02), first_decay=0.9), GREEN),
    ("Adam", OptimizerConfig(OptimizerName.ADAM, Constant(0.02)), RED),
)


def saddle_distance(config: OptimizerConfig) -> NDArray[np.float64]:
    """Distance along theta_1 from the double-well saddle, starting a tiny offset away from it."""
    well = DoubleWell()
    saddle = next(p for p in well.stationary_points() if np.linalg.eigvalsh(well.hessian(p))[0] < 0)
    start = np.array((saddle[0] + SADDLE_OFFSET, 0.5))
    trajectory = run(config, start, well.oracle, SADDLE_STEPS, np.random.default_rng(SEED))
    return np.abs(trajectory[:, 0] - saddle[0])


def escape_step(config: OptimizerConfig) -> int:
    distance = saddle_distance(config)
    assert (distance > SADDLE_ESCAPE_DISTANCE).any(), "iterate never left the saddle"
    return int(np.argmax(distance > SADDLE_ESCAPE_DISTANCE))


def _saddle_escape(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    for label, config, color in SADDLE_CONFIGS:
        axis.semilogy(np.maximum(saddle_distance(config), 1e-9), color=color, linewidth=2.6,
                      label=f"{label}: leaves after {escape_step(config)} updates")
    axis.axhline(SADDLE_ESCAPE_DISTANCE, color=GRAY, linestyle=":", linewidth=1.4)
    axis.set_xlim(0, SADDLE_STEPS)
    axis.set_ylim(1e-7, 3.0)
    axis.set_xlabel("update", fontsize=14)
    axis.set_ylabel(r"distance from the saddle along $\theta_1$", fontsize=14)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=12, loc="lower right")
    _clean(axis)
    _save(figure, path)



def generate_figures(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_directory / name for name in EXPECTED_FILENAMES}
    _landscapes(outputs["landscapes.png"])
    _objective_geometry(outputs["objective-geometry.png"])
    _learning_rates(outputs["learning-rates.png"])
    _contraction(outputs["contraction.png"])
    _conditioning(outputs["conditioning.png"])
    _gd_vs_sgd(outputs["gd-vs-sgd.png"])
    _schedules(outputs["schedules.png"])
    _momentum(outputs["momentum.png"])
    _diagonal_scaling(outputs["diagonal-scaling.png"])
    _optimizer_comparison(outputs["optimizer-comparison.png"])
    _optimizer_progress(outputs["optimizer-progress.png"])
    outputs["protocol-table.tex"].write_text(protocol_table(tuned_protocol()), encoding="utf-8")
    _rosenbrock(outputs["rosenbrock.png"])
    _multimodal(outputs["multimodal.png"])
    _saddle_escape(outputs["saddle-escape.png"])
    return tuple(outputs.values())


def main() -> None:
    """Run cheap optimizer experiments and write all figures and the protocol table."""
    output_directory = Path(__file__).resolve().parents[1] / "assets" / "generated"
    generate_figures(output_directory)


if __name__ == "__main__":
    main()
