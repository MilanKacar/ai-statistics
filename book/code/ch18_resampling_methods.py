"""
Chapter 18 companion script: Bootstrap and Permutation Tests

1) Bootstrap confidence interval for the MEDIAN of a skewed dataset (no clean
   textbook formula exists for this), compared to a naive Normal-approximation
   interval around the mean to show how they diverge.
2) Permutation test comparing two groups' conversion rates.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=29)

# ---------------------------------------------------------------------------
# 1) Bootstrap CI for the median
# ---------------------------------------------------------------------------
# Skewed data: mostly modest values, a few large outliers (income-like, Chapter 2)
data = rng.lognormal(mean=3, sigma=0.8, size=200)
observed_median = np.median(data)

n_bootstrap = 10_000
bootstrap_medians = np.array([
    np.median(rng.choice(data, size=len(data), replace=True))
    for _ in range(n_bootstrap)
])

ci_lower, ci_upper = np.percentile(bootstrap_medians, [2.5, 97.5])

print("=== Part 1: Bootstrap CI for the median ===")
print(f"Observed median: {observed_median:.3f}")
print(f"Bootstrap 95% CI for median: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"(Note: data is right-skewed -- mean={data.mean():.3f} is noticeably "
      f"higher than the median, same lesson as Chapter 2's income example)")
print()

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.hist(bootstrap_medians, bins=60, color="steelblue", edgecolor="white")
ax1.axvline(observed_median, color="black", linestyle="-", label="observed median")
ax1.axvline(ci_lower, color="red", linestyle="--", label="95% bootstrap CI")
ax1.axvline(ci_upper, color="red", linestyle="--")
ax1.set_title("Bootstrap distribution of the sample median (10,000 resamples)")
ax1.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Permutation test: two-group conversion rate comparison
# ---------------------------------------------------------------------------
n_a, n_b = 300, 300
true_rate_a, true_rate_b = 0.10, 0.14  # B genuinely converts better
group_a = rng.random(n_a) < true_rate_a
group_b = rng.random(n_b) < true_rate_b

observed_diff = group_b.mean() - group_a.mean()

combined = np.concatenate([group_a, group_b]).astype(float)
n_total = len(combined)

n_permutations = 10_000
permuted_diffs = np.empty(n_permutations)
for i in range(n_permutations):
    shuffled = rng.permutation(combined)
    fake_a = shuffled[:n_a]
    fake_b = shuffled[n_a:]
    permuted_diffs[i] = fake_b.mean() - fake_a.mean()

p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))

print("=== Part 2: Permutation test on conversion rates ===")
print(f"Group A conversion rate: {group_a.mean():.4f}")
print(f"Group B conversion rate: {group_b.mean():.4f}")
print(f"Observed difference (B - A): {observed_diff:.4f}")
print(f"Permutation test p-value: {p_value:.4f}")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(permuted_diffs, bins=60, color="lightsteelblue", edgecolor="white",
          label="shuffled (null) differences")
ax2.axvline(observed_diff, color="red", linestyle="--", linewidth=2,
            label=f"observed difference = {observed_diff:.4f}")
ax2.set_title(f"Permutation test: null distribution of shuffled differences\np-value = {p_value:.4f}")
ax2.legend()
plt.tight_layout()
plt.show()
