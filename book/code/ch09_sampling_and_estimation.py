"""
Chapter 9 companion script: Comparing Estimators (bias, variance)

We know the TRUE population mean (because we define the population). We then
compare three "center" estimators across thousands of repeated samples:
  1) sample mean
  2) sample median
  3) "first data point" (an estimator that's unbiased but bad in practice)
and compute each one's empirical bias and variance directly.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=5)

true_mean = 100.0
true_std = 15.0
population_size = 1_000_000
population = rng.normal(true_mean, true_std, population_size)

sample_sizes = [5, 30, 200]
n_repeats = 3000

estimators = {
    "sample mean": lambda s: s.mean(),
    "sample median": lambda s: np.median(s),
    "first data point": lambda s: s[0],
}

results = {name: {n: [] for n in sample_sizes} for name in estimators}

for n in sample_sizes:
    for _ in range(n_repeats):
        sample = rng.choice(population, size=n, replace=False)
        for name, fn in estimators.items():
            results[name][n].append(fn(sample))

print(f"TRUE population mean: {true_mean}\n")
print(f"{'estimator':<20}{'n':>6}{'avg estimate':>16}{'bias':>10}{'variance':>12}{'std dev':>10}")
for name in estimators:
    for n in sample_sizes:
        vals = np.array(results[name][n])
        bias = vals.mean() - true_mean
        print(f"{name:<20}{n:>6}{vals.mean():>16.3f}{bias:>10.3f}{vals.var():>12.3f}{vals.std():>10.3f}")

fig, axes = plt.subplots(len(estimators), len(sample_sizes), figsize=(13, 9), sharex=True)
for row, name in enumerate(estimators):
    for col, n in enumerate(sample_sizes):
        ax = axes[row, col]
        vals = np.array(results[name][n])
        ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(true_mean, color="red", linestyle="--")
        if row == 0:
            ax.set_title(f"n={n}")
        if col == 0:
            ax.set_ylabel(name)

plt.suptitle("Sampling distributions of three estimators across sample sizes\n"
             "(red dashed line = true population mean)")
plt.tight_layout()
plt.show()
