"""
Chapter 11 companion script: Confidence Intervals

1) Build the support-ticket 95% CI example by hand-formula, both z-based and
   t-based, and compare to scipy's built-in interval function.
2) The key demonstration: repeat "sample from a KNOWN population, build a 95%
   CI" 100 times, and plot all 100 intervals. About 95 of them should contain
   the true mean; about 5 should miss it. This is what "95% confidence"
   actually means.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(seed=21)

# ---------------------------------------------------------------------------
# 1) Worked example: support ticket resolution time
# ---------------------------------------------------------------------------
n, x_bar, s = 36, 12.5, 4.2
se = s / np.sqrt(n)

z_crit = 1.96
ci_z = (x_bar - z_crit * se, x_bar + z_crit * se)

t_crit = stats.t.ppf(0.975, df=n - 1)  # two-sided 95% -> 97.5th percentile
ci_t = (x_bar - t_crit * se, x_bar + t_crit * se)

print("=== Part 1: Support ticket example ===")
print(f"SE = {se:.4f}")
print(f"z-based 95% CI (z={z_crit}):      [{ci_z[0]:.3f}, {ci_z[1]:.3f}]")
print(f"t-based 95% CI (t={t_crit:.3f}):   [{ci_t[0]:.3f}, {ci_t[1]:.3f}]")
print("(very close since n=36 is reasonably large)\n")

# ---------------------------------------------------------------------------
# 2) What "95% confidence" really means -- 100 repeated intervals
# ---------------------------------------------------------------------------
true_mu, true_sigma = 50.0, 10.0
sample_n = 40
n_experiments = 100

lowers, uppers, contains_true = [], [], []
for _ in range(n_experiments):
    sample = rng.normal(true_mu, true_sigma, sample_n)
    xbar = sample.mean()
    se_i = sample.std(ddof=1) / np.sqrt(sample_n)
    t_i = stats.t.ppf(0.975, df=sample_n - 1)
    lo, hi = xbar - t_i * se_i, xbar + t_i * se_i
    lowers.append(lo)
    uppers.append(hi)
    contains_true.append(lo <= true_mu <= hi)

contains_true = np.array(contains_true)
print("=== Part 2: 100 repeated 95% CIs from a KNOWN population ===")
print(f"True mean: {true_mu}")
print(f"Intervals that actually contained the true mean: {contains_true.sum()} / {n_experiments}")
print("(should be close to 95, though it varies experiment to experiment)")

fig, ax = plt.subplots(figsize=(8, 10))
for i, (lo, hi, ok) in enumerate(zip(lowers, uppers, contains_true)):
    color = "steelblue" if ok else "red"
    ax.plot([lo, hi], [i, i], color=color, linewidth=1.5)
ax.axvline(true_mu, color="black", linestyle="--", label="true mean")
ax.set_title(f"100 independent 95% confidence intervals\n"
             f"({contains_true.sum()} contain the true mean, "
             f"{n_experiments - contains_true.sum()} miss it, shown in red)")
ax.set_xlabel("value")
ax.set_ylabel("experiment #")
ax.legend()
plt.tight_layout()
plt.show()
