"""
Chapter 6 companion script: Continuous Distributions and the Normal Distribution

1) Plot Uniform, Exponential, and Normal PDFs.
2) Shade the 68-95-99.7 regions under a Normal curve and print the exact areas.
3) Convert a worked height example to a z-score.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 1) Plot the three PDFs
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x_u = np.linspace(-0.5, 1.5, 300)
axes[0].plot(x_u, stats.uniform(loc=0, scale=1).pdf(x_u), color="steelblue")
axes[0].set_title("Uniform(0, 1)")
axes[0].set_ylim(0, 1.3)

x_e = np.linspace(0, 5, 300)
axes[1].plot(x_e, stats.expon(scale=1).pdf(x_e), color="darkorange")
axes[1].set_title("Exponential(rate=1)")

x_n = np.linspace(-4, 4, 300)
axes[2].plot(x_n, stats.norm(0, 1).pdf(x_n), color="seagreen")
axes[2].set_title("Normal(0, 1)")

plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) 68-95-99.7 rule with exact areas
# ---------------------------------------------------------------------------
mu, sigma = 0, 1
dist = stats.norm(mu, sigma)

for k in [1, 2, 3]:
    area = dist.cdf(mu + k * sigma) - dist.cdf(mu - k * sigma)
    print(f"P(within {k} std dev of mean) = {area:.4f}  "
          f"(rule of thumb: {[0.68, 0.95, 0.997][k-1]})")

fig2, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-4, 4, 400)
y = dist.pdf(x)
ax.plot(x, y, color="black")
colors = ["#cfe2f3", "#9fc5e8", "#6fa8dc"]
for k, color in zip([3, 2, 1], colors):
    mask = (x >= -k) & (x <= k)
    ax.fill_between(x[mask], y[mask], color=color, label=f"within {k} std dev")
ax.set_title("68-95-99.7 rule for the Standard Normal distribution")
ax.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 3) Worked z-score example: height
# ---------------------------------------------------------------------------
height_mu, height_sigma = 175, 7
person_height = 189
z = (person_height - height_mu) / height_sigma
percentile = stats.norm(height_mu, height_sigma).cdf(person_height) * 100

print()
print(f"Height ~ N({height_mu}, {height_sigma}^2)")
print(f"Person's height: {person_height}cm -> z-score = {z:.2f}")
print(f"That means they're taller than about {percentile:.1f}% of the population")

plt.show()
