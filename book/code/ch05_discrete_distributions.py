"""
Chapter 5 companion script: Discrete Distributions (Bernoulli, Binomial, Poisson)

1) Verify the hand-computed Binomial(n=4,p=0.5) P(X=2)=0.375 example.
2) Verify the hand-computed Poisson(lambda=3) P(X=5)=0.1008 example.
3) Plot PMFs for Binomial(n=10,p=0.3) and Poisson(lambda=3).
4) Simulate draws and overlay on the theoretical PMF to show they match.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(seed=7)

# ---------------------------------------------------------------------------
# Hand-computed examples, verified with scipy
# ---------------------------------------------------------------------------
p_binom_example = stats.binom.pmf(k=2, n=4, p=0.5)
p_poisson_example = stats.poisson.pmf(k=5, mu=3)

print("=== Verifying hand-computed examples ===")
print(f"Binomial(n=4, p=0.5): P(X=2) = {p_binom_example:.4f}  (hand calc: 0.375)")
print(f"Poisson(lambda=3):    P(X=5) = {p_poisson_example:.4f}  (hand calc: ~0.1008)")
print()

# ---------------------------------------------------------------------------
# Plot theoretical PMFs
# ---------------------------------------------------------------------------
n, p = 10, 0.3
lam = 3

k_binom = np.arange(0, n + 1)
pmf_binom = stats.binom.pmf(k_binom, n, p)

k_poisson = np.arange(0, 15)
pmf_poisson = stats.poisson.pmf(k_poisson, lam)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].bar(k_binom, pmf_binom, color="steelblue")
axes[0, 0].set_title(f"Binomial(n={n}, p={p}) -- theoretical PMF")
axes[0, 0].set_xlabel("k successes")

axes[0, 1].bar(k_poisson, pmf_poisson, color="darkorange")
axes[0, 1].set_title(f"Poisson(lambda={lam}) -- theoretical PMF")
axes[0, 1].set_xlabel("k events")

# ---------------------------------------------------------------------------
# Simulate and overlay on theory
# ---------------------------------------------------------------------------
n_draws = 50_000
binom_samples = rng.binomial(n, p, size=n_draws)
poisson_samples = rng.poisson(lam, size=n_draws)

axes[1, 0].hist(binom_samples, bins=np.arange(-0.5, n + 1.5, 1), density=True,
                color="lightsteelblue", edgecolor="white", label="simulated")
axes[1, 0].plot(k_binom, pmf_binom, "o-", color="red", label="theoretical")
axes[1, 0].set_title("Binomial: simulation vs theory")
axes[1, 0].legend()

axes[1, 1].hist(poisson_samples, bins=np.arange(-0.5, 15.5, 1), density=True,
                color="moccasin", edgecolor="white", label="simulated")
axes[1, 1].plot(k_poisson, pmf_poisson, "o-", color="red", label="theoretical")
axes[1, 1].set_title("Poisson: simulation vs theory")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print(f"Simulated Binomial mean={binom_samples.mean():.3f} (theory: {n*p})")
print(f"Simulated Poisson mean={poisson_samples.mean():.3f}, "
      f"variance={poisson_samples.var():.3f} (theory: both = {lam})")
