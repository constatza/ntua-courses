from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

OUT = Path(__file__).resolve().parent.parent
OUT.mkdir(parents=True, exist_ok=True)

# Example 1: Gaussian location parameter
mu0, tau0, sigma = 0.0, 0.5, 1.0
y = np.array([1.2, 1.7, 2.0, 2.3, 1.8])
n = y.size
ybar = y.mean()
post_var = 1.0 / (1.0 / tau0**2 + n / sigma**2)
post_sd = np.sqrt(post_var)
post_mean = post_var * (mu0 / tau0**2 + n * ybar / sigma**2)
like_sd = sigma / np.sqrt(n)

rng = np.random.default_rng(20260722)
T, B = 10_000, 1_000
proposal_sd = 0.45
chain = np.empty(T)
chain[0] = -0.5
accepted = 0

def log_target(theta: float) -> float:
    return -0.5 * ((theta - mu0) / tau0) ** 2 - 0.5 * np.sum(((y - theta) / sigma) ** 2)

for t in range(1, T):
    candidate = chain[t - 1] + rng.normal(0.0, proposal_sd)
    log_alpha = min(0.0, log_target(candidate) - log_target(chain[t - 1]))
    if np.log(rng.uniform()) <= log_alpha:
        chain[t] = candidate
        accepted += 1
    else:
        chain[t] = chain[t - 1]
retained = chain[B:]

xgrid = np.linspace(-1.5, 3.0, 800)
fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.plot(xgrid, norm.pdf(xgrid, mu0, tau0), label=r'Prior $p(\theta)$')
ax.plot(xgrid, norm.pdf(xgrid, ybar, like_sd), '--', label=r'Normalized likelihood')
ax.plot(xgrid, norm.pdf(xgrid, post_mean, post_sd), label=r'Posterior $p(\theta\mid y)$')
ax.set_xlabel(r'Parameter $\theta$')
ax.set_ylabel('Density')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig1.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(np.arange(T), chain, linewidth=0.6)
ax.axvline(B, linestyle=':', linewidth=1.5, label=rf'Burn-in boundary $B={B:,}$')
ax.axhline(post_mean, linestyle='--', linewidth=1.2, label='Analytic posterior mean')
ax.set_xlabel('Iteration')
ax.set_ylabel(r'$\theta^{(t)}$')
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(OUT / 'fig2.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 3.4))
kde = gaussian_kde(retained)
ax.hist(retained, bins=40, density=True, alpha=0.35, label='MH sample histogram')
ax.plot(xgrid, kde(xgrid), label='MH density estimate')
ax.plot(xgrid, norm.pdf(xgrid, post_mean, post_sd), '--', label='Analytic posterior')
ax.axvline(retained.mean(), linestyle=':', label='MH sample mean')
ax.set_xlabel(r'Parameter $\theta$')
ax.set_ylabel('Density')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig3.pdf')
plt.close(fig)

cumulative_mean = np.cumsum(retained) / np.arange(1, retained.size + 1)
fig, ax = plt.subplots(figsize=(7.0, 3.2))
ax.plot(np.arange(1, retained.size + 1), cumulative_mean, linewidth=0.8, label='Cumulative sample mean')
ax.axhline(post_mean, linestyle='--', label=r'Analytic $\mathbb{E}[\theta\mid y]$')
ax.set_xlabel('Number of retained states')
ax.set_ylabel('Cumulative sample mean')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'fig4.pdf')
plt.close(fig)

# Example 2: scalar linear model with slope close to 1/2
x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
y2 = np.array([-0.85, -1.05, 0.05, 0.55, 0.45, 1.75])
mu02, tau02, sigma2 = 0.0, 1.0, 0.40
sxx = float(np.sum(x * x))
sxy = float(np.sum(x * y2))
post_var2 = 1.0 / (1.0 / tau02**2 + sxx / sigma2**2)
post_sd2 = np.sqrt(post_var2)
post_mean2 = post_var2 * (mu02 / tau02**2 + sxy / sigma2**2)
z = norm.ppf(0.975)
prior_ci = (mu02 - z * tau02, mu02 + z * tau02)
post_ci = (post_mean2 - z * post_sd2, post_mean2 + z * post_sd2)
xx = np.linspace(-3, 3, 500)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
theta_grid = np.linspace(-3, 3, 700)
axes[0].plot(theta_grid, norm.pdf(theta_grid, mu02, tau02), label=r'Prior $p(\theta)$')
axes[0].plot(theta_grid, norm.pdf(theta_grid, post_mean2, post_sd2), label=r'Posterior $p(\theta\mid y)$')
axes[0].axvline(mu02, linestyle='--', label=r'$\mathbb{E}[\theta]$')
axes[0].axvline(post_mean2, linestyle=':', label=r'$\mathbb{E}[\theta\mid y]$')
axes[0].set_xlabel(r'Parameter $\theta$')
axes[0].set_ylabel('Density')
axes[0].set_title('(a) Parameter distributions')
axes[0].legend(frameon=False, fontsize=8)

axes[1].scatter(x, y2, label='Observations', zorder=3)
axes[1].plot(xx, xx * mu02, '--', label=r'$f(x,\mathbb{E}[\theta])$')
axes[1].fill_between(
    xx,
    np.minimum(xx * prior_ci[0], xx * prior_ci[1]),
    np.maximum(xx * prior_ci[0], xx * prior_ci[1]),
    alpha=0.18,
    label='Prior 95% model-response band',
)
axes[1].plot(xx, xx * post_mean2, label=r'$f(x,\mathbb{E}[\theta\mid y])$')
axes[1].fill_between(
    xx,
    np.minimum(xx * post_ci[0], xx * post_ci[1]),
    np.maximum(xx * post_ci[0], xx * post_ci[1]),
    alpha=0.25,
    label='Posterior 95% model-response band',
)
axes[1].axvline(0, linewidth=0.6)
axes[1].axhline(0, linewidth=0.6)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$y$')
axes[1].set_title('(b) Deterministic model responses')
axes[1].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / 'fig5.pdf')
plt.close(fig)

with (OUT / 'values.tex').open('w', encoding='utf-8') as file:
    file.write(f'\\newcommand{{\\NOne}}{{{n}}}\n')
    file.write(f'\\newcommand{{\\YBar}}{{{ybar:.2f}}}\n')
    file.write(f'\\newcommand{{\\PostMeanOne}}{{{post_mean:.5f}}}\n')
    file.write(f'\\newcommand{{\\PostSDOne}}{{{post_sd:.5f}}}\n')
    file.write(f'\\newcommand{{\\MHMean}}{{{retained.mean():.5f}}}\n')
    file.write(f'\\newcommand{{\\MHSD}}{{{retained.std(ddof=1):.5f}}}\n')
    file.write(f'\\newcommand{{\\AcceptRate}}{{{accepted / (T - 1):.3f}}}\n')
    file.write(f'\\newcommand{{\\TotalStates}}{{{T}}}\n')
    file.write(f'\\newcommand{{\\BurnIn}}{{{B}}}\n')
    file.write(f'\\newcommand{{\\RetainedStates}}{{{T-B}}}\n')
    file.write(f'\\newcommand{{\\Sxx}}{{{sxx:.2f}}}\n')
    file.write(f'\\newcommand{{\\Sxy}}{{{sxy:.2f}}}\n')
    file.write(f'\\newcommand{{\\PostMeanTwo}}{{{post_mean2:.5f}}}\n')
    file.write(f'\\newcommand{{\\PostSDTwo}}{{{post_sd2:.5f}}}\n')
    file.write(f'\\newcommand{{\\PriorCILow}}{{{prior_ci[0]:.5f}}}\n')
    file.write(f'\\newcommand{{\\PriorCIHigh}}{{{prior_ci[1]:.5f}}}\n')
    file.write(f'\\newcommand{{\\PostCILow}}{{{post_ci[0]:.5f}}}\n')
    file.write(f'\\newcommand{{\\PostCIHigh}}{{{post_ci[1]:.5f}}}\n')
