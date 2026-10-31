#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///

"""
Visualize a Gaussian-prior + Gaussian-likelihood Bayesian update.
Creates plots of prior, scaled likelihood, and posterior distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Constants
DEFAULT_PRIOR_MEAN: float = 0.0
DEFAULT_PRIOR_SD: float = 1.0
DEFAULT_DATA: float = 2.0
DEFAULT_LIKELIHOOD_SD: float = 0.5
THETA_RANGE_FACTOR: float = 4.0
NUM_POINTS: int = 1000

def compute_posterior(
    mu0: float, sigma0: float,
    data: float, sigma_lik: float
) -> tuple[float, float]:
    """Compute posterior mean and standard deviation for normal-normal update
    using explicit precisions.
    Prior ~ N(mu0, σ0²); Likelihood ~ N(data, σ_lik²).
    Returns (mu_post, sigma_post).
    """
    # convert variances to precisions
    tau0: float = 1.0 / (sigma0 * sigma0)
    tau_lik: float = 1.0 / (sigma_lik * sigma_lik)
    # posterior precision
    tau_post: float = tau0 + tau_lik
    # posterior mean (precision-weighted)
    mu_post: float = (tau0 * mu0 + tau_lik * data) / tau_post
    # posterior standard deviation
    sigma_post: float = (1.0 / tau_post) ** 0.5
    return mu_post, sigma_post


def plot_update(
    mu0: float = DEFAULT_PRIOR_MEAN, sigma0: float = DEFAULT_PRIOR_SD,
    data: float = DEFAULT_DATA, sigma_lik: float = DEFAULT_LIKELIHOOD_SD
) -> None:
    """Generate a plot of prior, scaled likelihood, and posterior densities."""
    mu_post, sigma_post = compute_posterior(mu0, sigma0, data, sigma_lik)

    theta_min = mu0 - THETA_RANGE_FACTOR * sigma0
    theta_max = data + THETA_RANGE_FACTOR * sigma_lik
    theta = np.linspace(theta_min, theta_max, NUM_POINTS)

    prior_pdf = norm(mu0, sigma0).pdf(theta)
    lik_pdf = norm(data, sigma_lik).pdf(theta)
    # scale the likelihood so its peak matches the prior peak (for visualization only)
    lik_scaled = lik_pdf * (prior_pdf.max() / lik_pdf.max())
    posterior_pdf = norm(mu_post, sigma_post).pdf(theta)

    plt.figure(figsize=(8, 5))
    plt.plot(theta, prior_pdf, label=f"Prior N({mu0:.2f}, {sigma0:.2f}²)")
    plt.plot(theta, lik_scaled, label=f"Scaled Likelihood N({data:.2f}, {sigma_lik:.2f}²)", linestyle='--')
    plt.plot(theta, posterior_pdf, label=f"Posterior N({mu_post:.2f}, {sigma_post:.2f}²)", linestyle='-.')
    plt.axvline(mu_post, color='gray', linestyle=':')
    plt.title("Gaussian–Gaussian Bayesian Update")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Density (scaled)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_filename = "gaussian_update.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

def main() -> None:
    """Main entry point."""
    plot_update()

if __name__ == "__main__":
    main()
