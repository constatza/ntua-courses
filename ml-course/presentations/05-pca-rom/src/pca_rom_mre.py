# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib", "scipy"]
# ///
"""Minimal reproducible example of the PCA-Galerkin ROM methodology.

Implements exactly the equations from the pca-rom.tex presentation:
  - Step 1: high-fidelity solves for training parameters
  - Step 2: mean and centred snapshot matrix
  - Step 3: PCA basis via covariance matrix eigenproblem
  - Step 4: reduced operators  K_r = V_rT K V_r,  f_r = V_rT (f - K u_mean)
  - Step 5: reduced solve + decode

Also demonstrates that the (f - K u_mean) term is necessary by comparing
the correct formulation against a naive wrong one (f_r = V_rT f).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.linalg import solve

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "assets"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Problem definition  K(ξ) u(ξ) = f(ξ)   with D = 3
# ---------------------------------------------------------------------------

_K0 = np.array([[4.0, -1.0, 0.0],
                 [-1.0, 4.0, -1.0],
                 [0.0, -1.0, 4.0]])   # 1D-Poisson tridiagonal, SPD

_K1 = np.diag([2.0, 1.0, 0.5])       # diagonal perturbation, PSD

_f0 = np.array([1.0, 0.0, 0.0])      # base load at node 1
_f1 = np.array([0.0, 0.0, 1.0])      # ξ1-varying load at node 3
_f2 = np.array([0.0, 1.0, 0.0])      # ξ2-varying load at node 2 (second parameter)


def stiffness(xi: np.ndarray) -> np.ndarray:
    """K(ξ) = K0 + ξ[0] K1  ∈ ℝ^{3×3}, SPD for ξ[0] ≥ 0."""
    return _K0 + xi[0] * _K1


def force(xi: np.ndarray) -> np.ndarray:
    """f(ξ) = f0 + ξ[0] f1 + ξ[1] f2  ∈ ℝ^3."""
    return _f0 + xi[0] * _f1 + xi[1] * _f2


def hf_solve(xi: np.ndarray) -> np.ndarray:
    """Exact high-fidelity solution u(ξ) = K(ξ)⁻¹ f(ξ)."""
    return solve(stiffness(xi), force(xi), assume_a="pos")


# ---------------------------------------------------------------------------
# Sampling: 50/50 random split on [0,3]×[0,2]  (2-parameter domain)
# ---------------------------------------------------------------------------

def make_parameter_split(
    n_train: int,
    n_deploy: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_train + n_deploy iid uniform samples from [0,3]×[0,2] and split.

    Args:
        n_train: number of training parameters.
        n_deploy: number of deployment parameters.
        rng: seeded NumPy random generator.

    Returns:
        Tuple (xi_train, xi_deploy), each ∈ ℝ^{N × 2}.
    """
    xi_all = rng.uniform([0.0, 0.0], [3.0, 2.0], size=(n_train + n_deploy, 2))
    return xi_all[:n_train], xi_all[n_train:]


# ---------------------------------------------------------------------------
# High-dimensional snapshot collection (for eigenvalue decay visualization)
# ---------------------------------------------------------------------------

def generate_high_dim_snapshots(
    D: int = 30,
    n_train: int = 200,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate snapshots for a high-dimensional system (moving Gaussian).

    Used to showcase eigenvalue decay and explained variance in systems with
    more dimensions and gradual mode truncation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    h = 1.0 / (D + 1)
    x = np.linspace(h, 1.0 - h, D)

    # We sample:
    #   xi_0 (center of Gaussian) in [0.15, 0.85]
    #   xi_1 (width of Gaussian) in [0.05, 0.15]
    xi_0 = rng.uniform(0.15, 0.85, size=n_train)
    xi_1 = rng.uniform(0.05, 0.15, size=n_train)

    U = np.zeros((n_train, D))
    for j in range(n_train):
        U[j, :] = np.exp(-0.5 * ((x - xi_0[j]) / xi_1[j])**2)

    return U


# ---------------------------------------------------------------------------
# High-fidelity snapshot collection
# ---------------------------------------------------------------------------

def collect_snapshots(xi_arr: np.ndarray) -> np.ndarray:
    """Solve HF system for each parameter and stack into snapshot matrix.

    Args:
        xi_arr: 1-D array of N parameter values.

    Returns:
        U ∈ ℝ^{N × D} with U[j, :] = u(ξ_j)ᵀ.
    """
    return np.stack([hf_solve(xi) for xi in xi_arr], axis=0)


# ---------------------------------------------------------------------------
# PCA basis construction  (tex: "PCA Basis Construction" frame)
# ---------------------------------------------------------------------------

def build_pca_basis(
    U_train: np.ndarray,
    r: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA basis from training snapshots following the tex equations.

    C = (1/N_train) U_cᵀ U_c,   C v_i = λ_i v_i,   V_r = [v_1, …, v_r]

    Args:
        U_train: snapshot matrix ∈ ℝ^{N_train × D}.
        r: reduced dimension.

    Returns:
        Tuple (u_mean, V_r, eigenvalues):
          u_mean  ∈ ℝ^D
          V_r     ∈ ℝ^{D × r}  (columns are leading eigenvectors)
          eigvals ∈ ℝ^D        (all eigenvalues, descending)
    """
    n_train = U_train.shape[0]

    # tex eq: ū = (1/N_train) Σ u(ξ_j)
    u_mean = U_train.mean(axis=0)

    # tex eq: (U_c)[j, :] = (u(ξ_j) - ū)ᵀ
    U_c = U_train - u_mean[np.newaxis, :]

    # tex eq: C = (1/N_train) U_cᵀ U_c
    C = (U_c.T @ U_c) / n_train

    # tex eq: C v_i = λ_i v_i  — eigh returns ascending order, so reverse
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = eigvals[::-1]       # descending
    eigvecs = eigvecs[:, ::-1]    # matching columns

    V_r = eigvecs[:, :r]          # ℝ^{D × r}
    return u_mean, V_r, eigvals


# ---------------------------------------------------------------------------
# Encoder / decoder  (tex: "PCA Basis Construction" frame)
# ---------------------------------------------------------------------------

def encode(u: np.ndarray, u_mean: np.ndarray, V_r: np.ndarray) -> np.ndarray:
    """Enc_r: u_r = V_rᵀ (u - ū)."""
    return V_r.T @ (u - u_mean)


def decode(u_r: np.ndarray, u_mean: np.ndarray, V_r: np.ndarray) -> np.ndarray:
    """Dec_r: û = ū + V_r u_r."""
    return u_mean + V_r @ u_r


# ---------------------------------------------------------------------------
# Reduced operators and ROM solve  (tex: "PCA-ROM: Galerkin Projection" frame)
# ---------------------------------------------------------------------------

def rom_solve_correct(
    xi: float,
    u_mean: np.ndarray,
    V_r: np.ndarray,
) -> np.ndarray:
    """ROM solve using the correct formulation from the tex.

    Reduced operators:
      K_r(ξ) = V_rᵀ K(ξ) V_r
      f_r(ξ) = V_rᵀ (f(ξ) - K(ξ) ū)     ← the (f - K ū) term

    Args:
        xi: parameter value.
        u_mean: mean high-fidelity solution ū ∈ ℝ^D.
        V_r: reduced basis ∈ ℝ^{D × r}.

    Returns:
        Reconstructed state û(ξ) ∈ ℝ^D.
    """
    K_xi = stiffness(xi)
    f_xi = force(xi)

    K_r = V_r.T @ K_xi @ V_r                          # ℝ^{r × r}
    f_r = V_r.T @ (f_xi - K_xi @ u_mean)              # ℝ^r

    u_r = solve(K_r, f_r, assume_a="pos")
    return decode(u_r, u_mean, V_r)


def rom_solve_wrong(
    xi: float,
    u_mean: np.ndarray,
    V_r: np.ndarray,
) -> np.ndarray:
    """ROM solve with the WRONG formulation: f_r = V_rᵀ f  (missing K ū term).

    Args:
        xi: parameter value.
        u_mean: mean high-fidelity solution ū ∈ ℝ^D.
        V_r: reduced basis ∈ ℝ^{D × r}.

    Returns:
        (Incorrectly) reconstructed state.
    """
    K_xi = stiffness(xi)
    f_xi = force(xi)

    K_r = V_r.T @ K_xi @ V_r
    f_r = V_r.T @ f_xi                                # ← wrong: ignores K ū shift

    u_r = solve(K_r, f_r, assume_a="pos")
    return decode(u_r, u_mean, V_r)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def relative_errors(U_hat: np.ndarray, U_ref: np.ndarray) -> np.ndarray:
    """‖û - u‖ / ‖u‖ for each row.

    Args:
        U_hat: reconstructed snapshots ∈ ℝ^{N × D}.
        U_ref: reference (HF) snapshots ∈ ℝ^{N × D}.

    Returns:
        1-D array of N relative errors.
    """
    return np.linalg.norm(U_hat - U_ref, axis=1) / np.linalg.norm(U_ref, axis=1)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_COLOURS = {"train": "#1f77b4", "deploy": "#ff7f0e"}
_ALPHA = 0.35
_MS = 12


def plot_3d_original_space(
    U_train: np.ndarray,
    U_deploy: np.ndarray,
    u_mean: np.ndarray,
    V_r: np.ndarray,
    save_path: pathlib.Path,
) -> None:
    """3D scatter of all HF solutions coloured by train/deploy split.

    Args:
        U_train: training snapshots ∈ ℝ^{N_train × 3}.
        U_deploy: deployment snapshots ∈ ℝ^{N_deploy × 3}.
        u_mean: mean solution ∈ ℝ^3.
        V_r: reduced basis ∈ ℝ^{3 × r}.
        save_path: output file path.
    """
    fig = plt.figure(figsize=(7, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.scatter(*U_train.T, color=_COLOURS["train"], alpha=_ALPHA, s=_MS,
               label=f"Train (N={len(U_train)})", depthshade=False)
    ax.scatter(*U_deploy.T, color=_COLOURS["deploy"], alpha=_ALPHA, s=_MS,
               label=f"Deploy (N={len(U_deploy)})", depthshade=False)

    # Build full 3-column PC frame: V_r gives PC1, PC2; PC3 is the normal.
    v3 = np.cross(V_r[:, 0], V_r[:, 1])
    v3 /= np.linalg.norm(v3)
    V_full = np.column_stack([V_r, v3])   # D×3 orthonormal frame

    scale = 0.08  # arrow length relative to data spread
    spread = np.ptp(np.vstack([U_train, U_deploy]), axis=0).max()
    arrow_len = scale * spread
    for i in range(3):
        tip = u_mean + arrow_len * V_full[:, i]
        ax.quiver(*u_mean, *(tip - u_mean),
                  color=f"C{i + 2}", linewidth=2, arrow_length_ratio=0.3,
                  label=f"PC{i + 1}")

    ax.scatter(*u_mean, color="black", s=50, marker="x", zorder=5, label="Mean")

    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$u_2$")
    ax.set_zlabel(r"$u_3$")
    ax.set_title("High-fidelity solutions in $\\mathbb{R}^3$")
    ax.legend(loc="upper left", fontsize=8)

    # The curve lives mostly in the u1-u3 plane (u2 barely varies).
    # Looking from azim≈160° (from the -u1/+u2 side) at elev≈30° shows
    # u3 rising, u1 falling, and u2 as a shallow depth axis.
    ax.view_init(elev=30, azim=-60)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


def plot_reduced_space(
    U_train: np.ndarray,
    U_deploy: np.ndarray,
    u_mean: np.ndarray,
    V_r: np.ndarray,
    save_path: pathlib.Path,
) -> None:
    """2D scatter of all solutions in reduced coordinates, coloured train/deploy.

    Args:
        U_train: training snapshots ∈ ℝ^{N_train × D}.
        U_deploy: deployment snapshots ∈ ℝ^{N_deploy × D}.
        u_mean: mean solution ∈ ℝ^D.
        V_r: reduced basis ∈ ℝ^{D × r}, r ≥ 2.
        save_path: output file path.
    """
    # Encode all snapshots: u_r = V_rᵀ (u - ū)
    Ur_train = (U_train - u_mean) @ V_r      # ℝ^{N_train × r}
    Ur_deploy = (U_deploy - u_mean) @ V_r    # ℝ^{N_deploy × r}

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Ur_train[:, 0], Ur_train[:, 1],
               color=_COLOURS["train"], alpha=_ALPHA, s=_MS,
               label=f"Train (N={len(U_train)})")
    ax.scatter(Ur_deploy[:, 0], Ur_deploy[:, 1],
               color=_COLOURS["deploy"], alpha=_ALPHA, s=_MS,
               label=f"Deploy (N={len(U_deploy)})")
    ax.scatter(0, 0, color="black", s=80, marker="x", zorder=5, label="Origin (mean)")

    ax.set_xlabel("PC$_1$")
    ax.set_ylabel("PC$_2$")
    ax.set_title("Solutions in 2-D reduced space")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


def plot_eigenvalues(
    eigvals: np.ndarray,
    save_path: pathlib.Path,
    max_modes: int = 10,
) -> None:
    """Bar chart of descending PCA eigenvalues (showing only the leading modes).

    Args:
        eigvals: all D eigenvalues in descending order.
        save_path: output file path.
        max_modes: maximum number of leading modes to display.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_vals = eigvals[:max_modes]
    x = np.arange(1, len(plot_vals) + 1)
    
    ax.bar(x, plot_vals, color="#1f77b4", edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xlabel("Mode $i$")
    ax.set_ylabel("$\\lambda_i$")
    ax.set_title(f"PCA eigenvalues (top {max_modes})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


def plot_explained_variance(
    eigvals: np.ndarray,
    save_path: pathlib.Path,
    max_modes: int = 15,
) -> None:
    """Cumulative explained variance η(r) = Σ_{i≤r} λ_i / Σ λ_i.

    Args:
        eigvals: all D eigenvalues in descending order.
        save_path: output file path.
        max_modes: maximum number of modes to display on the x-axis.
    """
    eta = np.cumsum(eigvals) / eigvals.sum()
    plot_eta = eta[:max_modes]
    r_vals = np.arange(1, len(plot_eta) + 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r_vals, plot_eta * 100, "o-", color="#1f77b4")
    ax.axhline(95, linestyle="--", color="#800020", linewidth=0.8, label="95 %")
    ax.set_xticks(r_vals)
    ax.set_xlabel("Reduced dimension $r$")
    ax.set_ylabel("Cumulative variance $\\eta(r)$ [%]")
    ax.set_title("Explained variance")
    ax.set_ylim(0, 102)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


def plot_reconstruction_errors(
    xi_train: np.ndarray,
    xi_deploy: np.ndarray,
    err_pca_train: np.ndarray,
    err_rom_train: np.ndarray,
    err_rom_deploy: np.ndarray,
    err_rom_wrong_deploy: np.ndarray,
    save_path: pathlib.Path,
) -> None:
    """Line plots of relative reconstruction error vs ξ.

    Args:
        xi_train: training parameter values.
        xi_deploy: deployment parameter values.
        err_pca_train: PCA encode-decode error on training set.
        err_rom_train: ROM error on training set.
        err_rom_deploy: ROM error on deployment set (correct formulation).
        err_rom_wrong_deploy: ROM error on deployment set (wrong formulation).
        save_path: output file path.
    """
    # xi arrays are N×2; sort by ξ1 for a clean x-axis
    xi1_train = xi_train[:, 0]
    xi1_deploy = xi_deploy[:, 0]
    idx_tr = np.argsort(xi1_train)
    idx_de = np.argsort(xi1_deploy)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(xi1_train[idx_tr], err_pca_train[idx_tr], "o", ms=4, alpha=0.7,
                color=_COLOURS["train"], label="PCA encode/decode (train)")
    ax.semilogy(xi1_train[idx_tr], err_rom_train[idx_tr], "s", ms=4, alpha=0.7,
                color="#2ca02c", label="ROM correct (train)")
    ax.semilogy(xi1_deploy[idx_de], err_rom_deploy[idx_de], "^", ms=4, alpha=0.7,
                color=_COLOURS["deploy"], label="ROM correct (deploy)")
    ax.semilogy(xi1_deploy[idx_de], err_rom_wrong_deploy[idx_de], "x", ms=6, alpha=0.9,
                color="#d62728", label="ROM WRONG $f_r = V_r^\\top f$ (deploy)")

    ax.set_xlabel("$\\xi_1$")
    ax.set_ylabel("Relative error $\\|\\hat{u} - u\\| / \\|u\\|$")
    ax.set_title("Reconstruction errors")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full PCA-ROM MRE pipeline."""
    RNG_SEED = 42
    N_TRAIN = 200
    N_DEPLOY = 200
    R = 2          # reduced dimension  (r=2 → 2D reduced space)

    rng = np.random.default_rng(RNG_SEED)

    # ------------------------------------------------------------------
    # 3D System (kept for spatial visualization & projection ROM)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("3D System Solver (for spatial visualization)")
    print("=" * 60)
    xi_train, xi_deploy = make_parameter_split(N_TRAIN, N_DEPLOY, rng)
    U_train = collect_snapshots(xi_train)      # ℝ^{N_train × 3}
    U_deploy = collect_snapshots(xi_deploy)    # ℝ^{N_deploy × 3}

    u_mean, V_r, eigvals = build_pca_basis(U_train, r=R)
    eta_r = eigvals[:R].sum() / eigvals.sum()

    print(f"  D = {U_train.shape[1]},  r = {R},  N_train = {N_TRAIN}")
    print(f"  Eigenvalues: {eigvals.round(6)}")
    print(f"  Explained variance r={R}: {eta_r * 100:.4f} %")
    print(f"  V_r orthonormality check  ‖V_rᵀ V_r - I‖ = "
          f"{np.linalg.norm(V_r.T @ V_r - np.eye(R)):.2e}")

    # PCA encode/decode reconstruction (no system solve — pure geometry)
    U_pca_train = np.stack([
        decode(encode(u, u_mean, V_r), u_mean, V_r)
        for u in U_train
    ])
    err_pca_train = relative_errors(U_pca_train, U_train)

    # ROM deployment — correct formulation
    U_rom_train = np.stack([rom_solve_correct(xi, u_mean, V_r) for xi in xi_train])
    U_rom_deploy = np.stack([rom_solve_correct(xi, u_mean, V_r) for xi in xi_deploy])

    err_rom_train = relative_errors(U_rom_train, U_train)
    err_rom_deploy = relative_errors(U_rom_deploy, U_deploy)

    # Wrong formulation (for comparison)
    U_rom_wrong_deploy = np.stack([rom_solve_wrong(xi, u_mean, V_r) for xi in xi_deploy])
    err_rom_wrong_deploy = relative_errors(U_rom_wrong_deploy, U_deploy)

    # Verification report
    print()
    print("=" * 60)
    print("Reconstruction errors (3D system)")
    print("=" * 60)
    fmt = "  {:<42s}  max = {:.2e}  mean = {:.2e}"
    print(fmt.format("PCA encode/decode (train):", err_pca_train.max(), err_pca_train.mean()))
    print(fmt.format("ROM correct (train):", err_rom_train.max(), err_rom_train.mean()))
    print(fmt.format("ROM correct (deploy):", err_rom_deploy.max(), err_rom_deploy.mean()))
    print(fmt.format("ROM WRONG f_r=V_rᵀf (deploy):", err_rom_wrong_deploy.max(), err_rom_wrong_deploy.mean()))
    print()

    if err_rom_deploy.max() > 0.10:
        print("WARNING: max deploy error exceeds 10% — check basis dimension r or problem setup.")
    else:
        print("OK: deploy errors < 10%.  Formulation verified.")

    ratio = err_rom_wrong_deploy.mean() / max(err_rom_deploy.mean(), 1e-16)
    print(f"     Wrong / correct mean error ratio: {ratio:.1f}×")

    # ------------------------------------------------------------------
    # High-dimensional System (for eigenvalue decay & explained variance plots)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("High-Dimensional System (D=30, moving Gaussian profiles)")
    print("=" * 60)
    D_hd = 30
    # Re-seeded RNG to ensure reproducibility
    rng_hd = np.random.default_rng(RNG_SEED)
    U_hd_train = generate_high_dim_snapshots(D=D_hd, n_train=N_TRAIN, rng=rng_hd)
    
    # Compute all eigenvalues for the high-dimensional system
    u_mean_hd, _, eigvals_hd = build_pca_basis(U_hd_train, r=1)
    print(f"  D = {D_hd},  N_train = {N_TRAIN}")
    print(f"  Eigenvalues (first 10): {eigvals_hd[:10].round(6)}")
    # Find r for 95% explained variance
    eta_hd = np.cumsum(eigvals_hd) / eigvals_hd.sum()
    r_95 = np.argmax(eta_hd >= 0.95) + 1
    print(f"  Explained variance 95% threshold achieved at r = {r_95} ({eta_hd[r_95-1]*100:.2f}%)")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Figures")
    print("=" * 60)
    plot_3d_original_space(U_train, U_deploy, u_mean, V_r,
                           FIGURES_DIR / "3d-original-space.png")
    plot_reduced_space(U_train, U_deploy, u_mean, V_r,
                       FIGURES_DIR / "2d-reduced-space.png")
    plot_eigenvalues(eigvals_hd, FIGURES_DIR / "eigenvalues.png")
    plot_explained_variance(eigvals_hd, FIGURES_DIR / "explained-variance.png")
    plot_reconstruction_errors(
        xi_train, xi_deploy,
        err_pca_train, err_rom_train, err_rom_deploy, err_rom_wrong_deploy,
        FIGURES_DIR / "reconstruction-error.png",
    )


if __name__ == "__main__":
    main()
