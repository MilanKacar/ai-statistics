"""
Chapter 14 companion script: Simple Linear Regression

1) Reproduce the hours-studied example by hand formula and cross-check
   against numpy.polyfit and scipy.stats.linregress.
2) Re-fit Anscombe's Quartet (from Chapter 2) with linear regression and
   plot BOTH the fitted line AND the residual plot for each dataset --
   showing how residual plots expose poor fits that R^2 alone hides.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# 1) Hours studied vs exam score
# ---------------------------------------------------------------------------
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([52, 60, 61, 68, 75], dtype=float)

x_bar, y_bar = x.mean(), y.mean()
beta1_hand = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
beta0_hand = y_bar - beta1_hand * x_bar

slope_np, intercept_np = np.polyfit(x, y, 1)
lr = stats.linregress(x, y)

print("=== Part 1: Simple linear regression ===")
print(f"Hand-computed:   beta1={beta1_hand:.4f}, beta0={beta0_hand:.4f}")
print(f"numpy.polyfit:   slope={slope_np:.4f}, intercept={intercept_np:.4f}")
print(f"scipy linregress: slope={lr.slope:.4f}, intercept={lr.intercept:.4f}, "
      f"R^2={lr.rvalue**2:.4f}")
print()

y_pred = beta0_hand + beta1_hand * x
sse = np.sum((y - y_pred) ** 2)
sst = np.sum((y - y_bar) ** 2)
r_squared = 1 - sse / sst
print(f"SSE={sse:.3f}, SST={sst:.3f}, R^2={r_squared:.4f} "
      f"(should match correlation^2 = {np.corrcoef(x,y)[0,1]**2:.4f})")

fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.scatter(x, y, color="steelblue", s=60)
xs = np.linspace(0, 6, 50)
ax1.plot(xs, beta0_hand + beta1_hand * xs, color="red")
ax1.set_title(f"Hours studied vs exam score\ny = {beta0_hand:.1f} + {beta1_hand:.1f}x, R^2={r_squared:.3f}")
ax1.set_xlabel("hours studied")
ax1.set_ylabel("exam score")
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Anscombe's Quartet: fitted line + residual plot
# ---------------------------------------------------------------------------
anscombe = {
    1: dict(x=[10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
             y=[8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    2: dict(x=[10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
             y=[9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    3: dict(x=[10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
             y=[7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    4: dict(x=[8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
             y=[6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}

fig2, axes = plt.subplots(2, 4, figsize=(18, 8))
for idx, data in anscombe.items():
    xi = np.array(data["x"], dtype=float)
    yi = np.array(data["y"], dtype=float)
    slope, intercept = np.polyfit(xi, yi, 1)
    pred = slope * xi + intercept
    resid = yi - pred
    r2 = np.corrcoef(xi, yi)[0, 1] ** 2

    ax_fit = axes[0, idx - 1]
    ax_fit.scatter(xi, yi, color="darkorange")
    xs = np.linspace(xi.min() - 1, xi.max() + 1, 50)
    ax_fit.plot(xs, slope * xs + intercept, color="steelblue")
    ax_fit.set_title(f"Dataset {idx}: R^2={r2:.3f}")

    ax_resid = axes[1, idx - 1]
    ax_resid.scatter(xi, resid, color="darkorange")
    ax_resid.axhline(0, color="steelblue", linestyle="--")
    ax_resid.set_title("Residuals")
    if idx == 1:
        ax_fit.set_ylabel("fitted line")
        ax_resid.set_ylabel("residual (y - prediction)")

plt.suptitle("Anscombe's Quartet: identical R^2, but residual plots instantly expose\n"
             "the curved relationship (set 2), the outlier (set 3), and the degenerate x (set 4)")
plt.tight_layout()
plt.show()
