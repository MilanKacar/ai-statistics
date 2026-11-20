"""
Chapter 8 companion script: The Central Limit Theorem

Start from a heavily right-skewed population (exponential distribution --
about as "not Normal" as a simple distribution gets). Draw many samples at
increasing sample sizes and plot the distribution of the sample mean at each
size. Watch it turn into a clean bell curve even though the original
population never looked anything like one.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=11)

# A skewed population: exponential distribution (long right tail)
population_scale = 50
true_mean = population_scale  # exponential(scale) has mean = scale
true_std = population_scale   # and std dev = scale too

sample_sizes = [1, 5, 30, 100]
n_repeats = 5000

fig, axes = plt.subplots(1, len(sample_sizes), figsize=(16, 4), sharey=False)

print("=== Central Limit Theorem demo ===")
print(f"Population (exponential): true mean = {true_mean}, true std dev = {true_std}")
print("Population shape: heavily right-skewed, NOT bell-shaped at all.\n")

for ax, n in zip(axes, sample_sizes):
    sample_means = np.array([
        rng.exponential(scale=population_scale, size=n).mean()
        for _ in range(n_repeats)
    ])

    predicted_se = true_std / np.sqrt(n)
    actual_se = sample_means.std()

    print(f"n={n:>4}: predicted standard error (sigma/sqrt(n)) = {predicted_se:.3f}, "
          f"actual observed standard error = {actual_se:.3f}")

    ax.hist(sample_means, bins=50, color="steelblue", edgecolor="white", density=True)
    ax.axvline(true_mean, color="red", linestyle="--")
    ax.set_title(f"Distribution of sample mean\n(n={n})")
    ax.set_xlabel("sample mean")

plt.suptitle("CLT in action: averaging a skewed population still converges to a bell curve")
plt.tight_layout()
plt.show()
