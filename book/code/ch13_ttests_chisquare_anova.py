"""
Chapter 13 companion script: t-tests, Chi-square, and ANOVA

1) Paired t-test: workout program heart-rate example, hand formula vs scipy.
2) Chi-square goodness-of-fit: die-fairness example, hand formula vs scipy.
3) One-way ANOVA: three teaching methods example, hand formula vs scipy.
4) Plot the t-distribution (varying df), chi-square distribution (varying k),
   and F-distribution shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 1) Paired t-test
# ---------------------------------------------------------------------------
before = np.array([72, 75, 68, 80, 74, 77, 71, 79])
after = np.array([70, 73, 66, 76, 72, 74, 70, 75])
d = after - before

t_stat_hand = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
t_result_scipy = stats.ttest_rel(after, before)

print("=== Part 1: Paired t-test (heart rate before/after) ===")
print(f"differences: {d}, mean diff = {d.mean():.3f}, std diff = {d.std(ddof=1):.3f}")
print(f"hand-computed t = {t_stat_hand:.4f}")
print(f"scipy ttest_rel: t = {t_result_scipy.statistic:.4f}, p = {t_result_scipy.pvalue:.6f}")
print()

# ---------------------------------------------------------------------------
# 2) Chi-square goodness of fit
# ---------------------------------------------------------------------------
observed = np.array([12, 8, 9, 11, 7, 13])
expected = np.array([10, 10, 10, 10, 10, 10])

chi2_hand = np.sum((observed - expected) ** 2 / expected)
chi2_scipy = stats.chisquare(observed, expected)

print("=== Part 2: Chi-square goodness-of-fit (die fairness) ===")
print(f"hand-computed chi2 = {chi2_hand:.4f}")
print(f"scipy chisquare:    chi2 = {chi2_scipy.statistic:.4f}, p = {chi2_scipy.pvalue:.4f}")
print()

# ---------------------------------------------------------------------------
# 3) One-way ANOVA
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=42)
method_a = rng.normal(75, 5, 5)
method_b = rng.normal(78, 5, 5)
method_c = rng.normal(82, 5, 5)

grand_mean = np.concatenate([method_a, method_b, method_c]).mean()
groups = [method_a, method_b, method_c]

ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

df_between = len(groups) - 1
df_within = sum(len(g) for g in groups) - len(groups)

ms_between = ss_between / df_between
ms_within = ss_within / df_within
f_stat_hand = ms_between / ms_within

f_result_scipy = stats.f_oneway(method_a, method_b, method_c)

print("=== Part 3: One-way ANOVA (three teaching methods) ===")
print(f"Group means: A={method_a.mean():.2f}, B={method_b.mean():.2f}, C={method_c.mean():.2f}")
print(f"SS_between={ss_between:.3f}, SS_within={ss_within:.3f}")
print(f"hand-computed F = {f_stat_hand:.4f}")
print(f"scipy f_oneway:  F = {f_result_scipy.statistic:.4f}, p = {f_result_scipy.pvalue:.4f}")
print()

# ---------------------------------------------------------------------------
# 4) Visualize the three distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

x = np.linspace(-5, 5, 400)
axes[0].plot(x, stats.norm.pdf(x), "k--", label="Normal (df=inf)")
for df in [1, 3, 10, 30]:
    axes[0].plot(x, stats.t.pdf(x, df), label=f"t, df={df}")
axes[0].set_title("t-distribution: heavier tails at low df,\nconverges to Normal as df grows")
axes[0].legend(fontsize=8)

x2 = np.linspace(0, 20, 400)
for k in [1, 2, 5, 10]:
    axes[1].plot(x2, stats.chi2.pdf(x2, k), label=f"k={k}")
axes[1].set_title("Chi-square distribution\n(sum of k squared standard Normals)")
axes[1].legend(fontsize=8)

x3 = np.linspace(0.01, 5, 400)
for df1, df2 in [(2, 10), (5, 10), (10, 30)]:
    axes[2].plot(x3, stats.f.pdf(x3, df1, df2), label=f"df1={df1}, df2={df2}")
axes[2].set_title("F-distribution\n(ratio of two variance estimates)")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()
