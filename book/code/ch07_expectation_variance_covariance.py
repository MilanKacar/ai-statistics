"""
Chapter 7 companion script: Expectation, Variance, Covariance, Correlation

1) Confirm E[X] for a die roll and a biased bet via simulation.
2) Generate (X, Y) pairs with different relationships (strong positive, strong
   negative, none, and a perfect-but-nonlinear parabola) and show how
   correlation captures linear relationships but misses the parabola entirely.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=3)

# ---------------------------------------------------------------------------
# 1) Expected value via simulation
# ---------------------------------------------------------------------------
n_trials = 1_000_000

die_rolls = rng.integers(1, 7, size=n_trials)
print("=== Expected value: die roll ===")
print(f"Simulated E[X] = {die_rolls.mean():.4f}  (theory: 3.5)")
print()

win = rng.random(n_trials) < 0.1
net_profit = np.where(win, 4, -1)
print("=== Expected value: biased bet ===")
print(f"Simulated E[net profit] = {net_profit.mean():.4f}  (theory: -0.5)")
print()

# ---------------------------------------------------------------------------
# 2) Correlation and its blind spot
# ---------------------------------------------------------------------------
n = 300
x = rng.normal(0, 1, n)

datasets = {
    "Strong positive (rho~0.9)": (x, 0.9 * x + rng.normal(0, 0.45, n)),
    "Strong negative (rho~-0.9)": (x, -0.9 * x + rng.normal(0, 0.45, n)),
    "No relationship (rho~0)": (x, rng.normal(0, 1, n)),
    "Perfect parabola (rho~0, but Y=X^2 exactly)": (x, x**2),
}

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, (title, (xi, yi)) in zip(axes, datasets.items()):
    rho = np.corrcoef(xi, yi)[0, 1]
    ax.scatter(xi, yi, alpha=0.5, s=15, color="steelblue")
    ax.set_title(f"{title}\ncomputed rho = {rho:.3f}")

plt.suptitle("Correlation measures LINEAR relationship strength only -- "
             "note the parabola's near-zero rho despite a perfect relationship")
plt.tight_layout()
plt.show()
