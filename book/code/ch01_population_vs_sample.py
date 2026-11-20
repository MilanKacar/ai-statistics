"""
Chapter 1 companion script: Population vs. Sample

We build a fake "population" of 10,000 people with a KNOWN true mean age
(something you could never know in real life). Then we repeatedly draw small
random samples from it and watch the sample mean bounce around the true mean.

What to look for when you run this:
1. The printed sample means for n=5 jump around a lot.
2. As n grows (5 -> 30 -> 200), the sample means cluster tighter around the
   true population mean. This shrinking spread as sample size grows is the
   seed of the Central Limit Theorem (Chapter 8).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)

# --- Step 1: create a population with a KNOWN true mean ---
POPULATION_SIZE = 10_000
true_mean_age = 34.0
true_std_age = 10.0
population = rng.normal(loc=true_mean_age, scale=true_std_age, size=POPULATION_SIZE)
population = np.clip(population, 18, 90)  # ages should be realistic

print(f"TRUE population mean age (mu): {population.mean():.3f}")
print(f"TRUE population std dev:        {population.std():.3f}")
print()

# --- Step 2: draw samples of different sizes and see how the sample mean behaves ---
sample_sizes = [5, 30, 200]
n_repeats = 2000  # how many independent samples we draw, to see the SPREAD of x-bar

fig, axes = plt.subplots(1, len(sample_sizes), figsize=(15, 4), sharey=True)

for ax, n in zip(axes, sample_sizes):
    sample_means = []
    for _ in range(n_repeats):
        sample = rng.choice(population, size=n, replace=False)
        sample_means.append(sample.mean())
    sample_means = np.array(sample_means)

    print(f"n={n:>4}: average of sample means = {sample_means.mean():.3f}, "
          f"spread (std) of sample means = {sample_means.std():.3f}")

    ax.hist(sample_means, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(true_mean_age, color="red", linestyle="--", label="true mean (mu)")
    ax.set_title(f"Sample means, n={n}")
    ax.set_xlabel("sample mean (x-bar)")
    ax.legend()

axes[0].set_ylabel("how often this x-bar value occurred")
plt.suptitle("As sample size grows, the sample mean clusters tighter around the true mean")
plt.tight_layout()
plt.show()
