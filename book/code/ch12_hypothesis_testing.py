"""
Chapter 12 companion script: Hypothesis Testing and p-values

1) Reproduce the conversion-rate worked example (hand z-test) and cross-check
   against scipy's proportions test.
2) Type-I-error demonstration: simulate 1000 experiments where H0 is
   deliberately TRUE, and confirm the false-positive rate lands near alpha.
3) p-hacking demonstration: run 20 unrelated tests on pure noise and see how
   often at least one comes out "significant" by chance alone.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(seed=71)

# ---------------------------------------------------------------------------
# 1) Conversion rate hypothesis test
# ---------------------------------------------------------------------------
p0 = 0.10        # null hypothesis: true rate is 10%
n = 400
observed_conversions = 52
p_hat = observed_conversions / n

se = np.sqrt(p0 * (1 - p0) / n)
z = (p_hat - p0) / se
p_value_two_sided = 2 * (1 - stats.norm.cdf(abs(z)))

print("=== Part 1: Conversion rate test ===")
print(f"Observed rate: {p_hat:.4f}, null rate: {p0}, SE={se:.4f}")
print(f"z = {z:.4f}")
print(f"two-sided p-value (hand calc) = {p_value_two_sided:.4f}")

# scipy's built-in one-proportion z-test equivalent via binomtest for comparison
binom_result = stats.binomtest(observed_conversions, n, p0, alternative="two-sided")
print(f"scipy binomtest p-value (exact, not normal-approx) = {binom_result.pvalue:.4f}")
print()

# ---------------------------------------------------------------------------
# 2) Type I error rate demonstration
# ---------------------------------------------------------------------------
alpha = 0.05
n_experiments = 1000
n_per_experiment = 400
true_p = p0  # H0 is TRUE in every single one of these simulated experiments

false_positives = 0
for _ in range(n_experiments):
    sample = rng.random(n_per_experiment) < true_p
    phat = sample.mean()
    se_i = np.sqrt(p0 * (1 - p0) / n_per_experiment)
    z_i = (phat - p0) / se_i
    p_i = 2 * (1 - stats.norm.cdf(abs(z_i)))
    if p_i < alpha:
        false_positives += 1

print("=== Part 2: Type I error rate (H0 is TRUE in every experiment) ===")
print(f"alpha = {alpha}")
print(f"False positives: {false_positives}/{n_experiments} = "
      f"{false_positives/n_experiments:.4f} (should be close to {alpha})")
print()

# ---------------------------------------------------------------------------
# 3) p-hacking / multiple comparisons demonstration
# ---------------------------------------------------------------------------
n_batches = 2000
n_tests_per_batch = 20
n_per_test = 100

at_least_one_significant = 0
all_p_values_first_batch = None

for batch in range(n_batches):
    p_values_this_batch = []
    for _ in range(n_tests_per_batch):
        # two independent, completely unrelated noise samples -- H0 is true
        group1 = rng.normal(0, 1, n_per_test)
        group2 = rng.normal(0, 1, n_per_test)
        t_stat, p_val = stats.ttest_ind(group1, group2)
        p_values_this_batch.append(p_val)
    if batch == 0:
        all_p_values_first_batch = p_values_this_batch
    if min(p_values_this_batch) < alpha:
        at_least_one_significant += 1

print("=== Part 3: p-hacking / multiple comparisons ===")
print(f"Running {n_tests_per_batch} unrelated tests per batch, {n_batches} batches, "
      f"all on pure noise (H0 true for every single test)")
print(f"Fraction of batches with AT LEAST ONE 'significant' (p<{alpha}) result: "
      f"{at_least_one_significant/n_batches:.4f}")
print(f"(theoretical: 1 - (1-{alpha})^{n_tests_per_batch} = "
      f"{1 - (1-alpha)**n_tests_per_batch:.4f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_p_values_first_batch, bins=20, color="steelblue", edgecolor="white")
ax.axvline(alpha, color="red", linestyle="--", label=f"alpha={alpha}")
ax.set_title(f"p-values from {n_tests_per_batch} unrelated tests on pure noise\n"
             "(uniformly distributed under H0 -- some always look 'significant')")
ax.set_xlabel("p-value")
ax.legend()
plt.tight_layout()
plt.show()
