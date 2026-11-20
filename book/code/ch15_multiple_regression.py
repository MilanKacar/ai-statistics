"""
Chapter 15 companion script: Multiple Regression, R-squared, and Overfitting

1) Fit a multiple regression model on synthetic housing data (sqft, bedrooms,
   age -> price).
2) Add an increasing number of PURE RANDOM NOISE columns as extra predictors
   and show that raw R^2 climbs anyway, while adjusted R^2 stays flat/declines.
   This is overfitting, made mechanically visible.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(seed=99)

# ---------------------------------------------------------------------------
# 1) Realistic multiple regression
# ---------------------------------------------------------------------------
n = 200
sqft = rng.normal(1800, 500, n)
bedrooms = rng.integers(1, 6, n)
age = rng.uniform(0, 80, n)

true_price = (50_000 + 120 * sqft + 8_000 * bedrooms - 500 * age
              + rng.normal(0, 20_000, n))

X = np.column_stack([sqft, bedrooms, age])
model = LinearRegression().fit(X, true_price)
r2_real = model.score(X, true_price)

print("=== Part 1: Multiple regression on realistic predictors ===")
print(f"Coefficients: sqft={model.coef_[0]:.2f}, bedrooms={model.coef_[1]:.2f}, "
      f"age={model.coef_[2]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R^2 = {r2_real:.4f}")
print()

# ---------------------------------------------------------------------------
# 2) Add pure noise predictors and watch R^2 climb anyway
# ---------------------------------------------------------------------------
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

noise_counts = [0, 5, 10, 20, 40, 80, 120, 150]
raw_r2s, adj_r2s = [], []

for n_noise in noise_counts:
    noise_cols = rng.normal(0, 1, size=(n, n_noise)) if n_noise > 0 else np.empty((n, 0))
    X_aug = np.column_stack([X, noise_cols])
    m = LinearRegression().fit(X_aug, true_price)
    r2 = m.score(X_aug, true_price)
    p = X_aug.shape[1]
    raw_r2s.append(r2)
    adj_r2s.append(adjusted_r2(r2, n, p))
    print(f"noise columns added: {n_noise:>4} (total predictors={p:>4}) -> "
          f"raw R^2={r2:.4f}, adjusted R^2={adj_r2s[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(noise_counts, raw_r2s, "o-", label="raw R^2 (keeps climbing!)", color="red")
plt.plot(noise_counts, adj_r2s, "o-", label="adjusted R^2 (stays flat / declines)", color="steelblue")
plt.xlabel("number of USELESS random-noise predictors added")
plt.ylabel("R^2")
plt.title("Overfitting made visible: raw R^2 rewards adding noise, adjusted R^2 doesn't")
plt.legend()
plt.tight_layout()
plt.show()
