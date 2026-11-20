"""
Chapter 10 companion script: Maximum Likelihood Estimation

1) Plot the log-likelihood of candidate mu values for Normal data, confirming
   the peak sits exactly at the sample mean.
2) Show that fitting a linear regression by maximizing Gaussian log-likelihood
   gives the IDENTICAL answer to minimizing sum of squared errors directly --
   concretely demonstrating "MSE is Gaussian MLE."
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats

rng = np.random.default_rng(seed=13)

# ---------------------------------------------------------------------------
# 1) Log-likelihood hill for the Normal mean
# ---------------------------------------------------------------------------
true_mu, true_sigma = 10.0, 3.0
data = rng.normal(true_mu, true_sigma, size=50)
sample_mean = data.mean()

candidate_mus = np.linspace(true_mu - 5, true_mu + 5, 300)
log_likelihoods = np.array([
    np.sum(stats.norm.logpdf(data, loc=m, scale=true_sigma))
    for m in candidate_mus
])
best_mu_from_grid = candidate_mus[np.argmax(log_likelihoods)]

print("=== Part 1: Log-likelihood peak vs sample mean ===")
print(f"Sample mean (x-bar):              {sample_mean:.4f}")
print(f"Mu that maximizes log-likelihood: {best_mu_from_grid:.4f}")
print("(these should match closely -- the MLE for a Normal mean IS the sample mean)\n")

plt.figure(figsize=(8, 5))
plt.plot(candidate_mus, log_likelihoods, color="steelblue")
plt.axvline(sample_mean, color="red", linestyle="--", label=f"sample mean = {sample_mean:.2f}")
plt.xlabel("candidate mu")
plt.ylabel("log-likelihood of observed data")
plt.title("Log-likelihood as a function of candidate mu (Normal data)")
plt.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) MSE regression vs Gaussian-MLE regression -- same answer
# ---------------------------------------------------------------------------
n = 100
x = rng.uniform(0, 10, n)
true_slope, true_intercept, noise_sigma = 2.5, 1.0, 2.0
y = true_slope * x + true_intercept + rng.normal(0, noise_sigma, n)

# (a) plain least squares (minimize sum of squared errors)
slope_ls, intercept_ls = np.polyfit(x, y, 1)

# (b) maximize Gaussian log-likelihood directly over (slope, intercept, sigma)
def neg_log_likelihood(params):
    slope, intercept, log_sigma = params
    sigma = np.exp(log_sigma)  # keep sigma positive
    predictions = slope * x + intercept
    return -np.sum(stats.norm.logpdf(y, loc=predictions, scale=sigma))

result = optimize.minimize(neg_log_likelihood, x0=[0.0, 0.0, 0.0])
slope_mle, intercept_mle, _ = result.x

print("=== Part 2: MSE regression vs Gaussian MLE regression ===")
print(f"True slope={true_slope}, true intercept={true_intercept}")
print(f"Least-squares fit:   slope={slope_ls:.4f}, intercept={intercept_ls:.4f}")
print(f"Gaussian MLE fit:    slope={slope_mle:.4f}, intercept={intercept_mle:.4f}")
print("(these two fits should match almost exactly -- same optimization problem)")

plt.show()
