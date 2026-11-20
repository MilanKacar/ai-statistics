"""
Chapter 17 companion script: Bayesian Statistics

1) Beta-Binomial coin-bias example: compute the posterior by hand formula,
   confirm with scipy, and visualize how the posterior sharpens as more
   coin flips accumulate (prior's influence fading with more data).
2) Demonstrate Ridge regression == MAP estimation with a Gaussian prior, by
   fitting the same data two ways and comparing coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.linear_model import Ridge

rng = np.random.default_rng(seed=23)

# ---------------------------------------------------------------------------
# 1) Beta-Binomial posterior
# ---------------------------------------------------------------------------
prior_alpha, prior_beta = 2, 2  # Beta(2,2) prior
heads, tails = 7, 3

post_alpha = prior_alpha + heads
post_beta = prior_beta + tails
posterior_mean = post_alpha / (post_alpha + post_beta)

print("=== Part 1: Beta-Binomial posterior ===")
print(f"Prior: Beta({prior_alpha},{prior_beta}), mean={prior_alpha/(prior_alpha+prior_beta):.3f}")
print(f"Data: {heads} heads, {tails} tails")
print(f"Posterior: Beta({post_alpha},{post_beta}), mean={posterior_mean:.4f} "
      f"(hand calc: 9/14 = {9/14:.4f})")
print()

# Show posterior sharpening as more data accumulates (same true p=0.7 throughout)
true_p = 0.7
flip_counts = [0, 10, 50, 200, 1000]
p_grid = np.linspace(0, 1, 500)

fig1, ax1 = plt.subplots(figsize=(9, 5))
cumulative_heads, cumulative_tails = 0, 0
rng2 = np.random.default_rng(seed=1)
prev_n = 0
for n_flips in flip_counts:
    new_flips = n_flips - prev_n
    if new_flips > 0:
        outcomes = rng2.random(new_flips) < true_p
        cumulative_heads += outcomes.sum()
        cumulative_tails += (~outcomes).sum()
    prev_n = n_flips

    a = prior_alpha + cumulative_heads
    b = prior_beta + cumulative_tails
    ax1.plot(p_grid, stats.beta.pdf(p_grid, a, b), label=f"n={n_flips} flips")

ax1.axvline(true_p, color="black", linestyle="--", label=f"true p={true_p}")
ax1.set_title("Posterior sharpening as more coin flips accumulate\n(prior's influence fades with more data)")
ax1.set_xlabel("p (probability of heads)")
ax1.legend(fontsize=8)
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Ridge regression == MAP estimation with Gaussian prior
# ---------------------------------------------------------------------------
n = 80
x1 = rng.normal(0, 1, n)
x2 = rng.normal(0, 1, n)
true_b1, true_b2 = 3.0, -1.5
noise_sigma = 1.0
y = true_b1 * x1 + true_b2 * x2 + rng.normal(0, noise_sigma, n)
X = np.column_stack([x1, x2])

alpha_ridge = 5.0  # sklearn's "alpha" = our lambda
ridge = Ridge(alpha=alpha_ridge, fit_intercept=False).fit(X, y)

# MAP: maximize log(likelihood) + log(prior)
# likelihood: Gaussian errors with known noise_sigma
# prior: each beta_j ~ N(0, tau^2), where tau relates to lambda via lambda = sigma^2/tau^2
tau_squared = noise_sigma ** 2 / alpha_ridge

def neg_log_posterior(betas):
    residuals = y - X @ betas
    log_likelihood = -np.sum(residuals ** 2) / (2 * noise_sigma ** 2)
    log_prior = -np.sum(betas ** 2) / (2 * tau_squared)
    return -(log_likelihood + log_prior)

result = optimize.minimize(neg_log_posterior, x0=[0.0, 0.0])

print("=== Part 2: Ridge regression == MAP with Gaussian prior ===")
print(f"True coefficients: b1={true_b1}, b2={true_b2}")
print(f"Ridge (sklearn):    b1={ridge.coef_[0]:.4f}, b2={ridge.coef_[1]:.4f}")
print(f"MAP (manual optim): b1={result.x[0]:.4f}, b2={result.x[1]:.4f}")
print("(these two should match closely -- same underlying optimization problem)")

plt.show()
