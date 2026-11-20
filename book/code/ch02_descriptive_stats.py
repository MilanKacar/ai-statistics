"""
Chapter 2 companion script: Center, Spread, Shape, and Anscombe's Quartet

Part 1: a skewed "income-like" dataset, showing mean != median, and computing
         the standard toolkit of summary statistics by hand and with numpy.
Part 2: Anscombe's Quartet -- four datasets with identical summary statistics
         but wildly different shapes when plotted. The whole point: never
         trust summary numbers without plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Part 1: skewed data, mean vs median, variance, IQR
# ---------------------------------------------------------------------------
incomes = np.array([30, 32, 35, 38, 1000])  # in thousands of dollars

mean_income = incomes.mean()
median_income = np.median(incomes)
variance_population = incomes.var(ddof=0)   # divide by n
variance_sample = incomes.var(ddof=1)       # divide by n-1  (Bessel's correction)
std_sample = incomes.std(ddof=1)
q1, q3 = np.percentile(incomes, [25, 75])
iqr = q3 - q1

print("=== Part 1: Skewed income data ===")
print(f"data: {incomes}")
print(f"mean:               {mean_income:.2f}k")
print(f"median:             {median_income:.2f}k   <-- notice how far this is from the mean")
print(f"population variance:{variance_population:.2f}")
print(f"sample variance:    {variance_sample:.2f}  (uses n-1)")
print(f"sample std dev:     {std_sample:.2f}k")
print(f"Q1={q1}, Q3={q3}, IQR={iqr}")
print()

fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))
axes1[0].hist(incomes, bins=10, color="steelblue", edgecolor="white")
axes1[0].axvline(mean_income, color="red", label=f"mean={mean_income:.1f}")
axes1[0].axvline(median_income, color="green", label=f"median={median_income:.1f}")
axes1[0].set_title("Histogram: mean vs median under skew")
axes1[0].legend()

axes1[1].boxplot(incomes, vert=True)
axes1[1].set_title("Box plot (shows median, IQR, outliers)")
plt.tight_layout()

# ---------------------------------------------------------------------------
# Part 2: Anscombe's Quartet
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

print("=== Part 2: Anscombe's Quartet -- identical stats, different shapes ===")
fig2, axes2 = plt.subplots(2, 2, figsize=(9, 8))
for (idx, data), ax in zip(anscombe.items(), axes2.flat):
    x = np.array(data["x"], dtype=float)
    y = np.array(data["y"], dtype=float)
    corr = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)

    print(f"Dataset {idx}: mean(x)={x.mean():.2f} mean(y)={y.mean():.2f} "
          f"var(x)={x.var(ddof=1):.2f} var(y)={y.var(ddof=1):.2f} "
          f"corr={corr:.3f} best-fit line: y={slope:.2f}x+{intercept:.2f}")

    ax.scatter(x, y, color="darkorange")
    xs = np.linspace(x.min() - 1, x.max() + 1, 50)
    ax.plot(xs, slope * xs + intercept, color="steelblue")
    ax.set_title(f"Dataset {idx}")

plt.suptitle("Anscombe's Quartet: same mean/variance/correlation/regression line, totally different shapes")
plt.tight_layout()
plt.show()
