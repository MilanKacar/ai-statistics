"""
Chapter 16 companion script: Bias-Variance Tradeoff, Cross-Validation, Regularization

1) Fit polynomials of increasing degree; plot training error vs cross-validated
   error against complexity -- the classic U-shaped sweet-spot curve.
2) Apply Ridge regularization at several lambda values to an overly-complex
   polynomial fit and show how it tames the wild overfit curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

rng = np.random.default_rng(seed=17)

# ---------------------------------------------------------------------------
# Generate noisy data from a true nonlinear function
# ---------------------------------------------------------------------------
def true_function(x):
    return np.sin(x) * 2 + 0.3 * x

n = 40
x = np.sort(rng.uniform(0, 8, n))
y = true_function(x) + rng.normal(0, 0.6, n)
X = x.reshape(-1, 1)

# ---------------------------------------------------------------------------
# 1) Training error vs cross-validated error across polynomial degree
# ---------------------------------------------------------------------------
degrees = range(1, 16)
train_errors, cv_errors = [], []
kf = KFold(n_splits=5, shuffle=True, random_state=0)

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    train_pred = model.predict(X_poly)
    train_mse = np.mean((y - train_pred) ** 2)
    train_errors.append(train_mse)

    cv_scores = cross_val_score(LinearRegression(), X_poly, y,
                                 cv=kf, scoring="neg_mean_squared_error")
    cv_errors.append(-cv_scores.mean())

print("=== Part 1: Training error vs cross-validated error ===")
for d, tr, cv in zip(degrees, train_errors, cv_errors):
    print(f"degree={d:>2}: train MSE={tr:.4f}, cross-val MSE={cv:.4f}")

best_degree = degrees[int(np.argmin(cv_errors))]
print(f"\nBest degree by cross-validation: {best_degree}")

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(degrees, train_errors, "o-", label="training error (always improves)", color="red")
ax1.plot(degrees, cv_errors, "o-", label="cross-validated error (U-shaped)", color="steelblue")
ax1.axvline(best_degree, color="green", linestyle="--", label=f"best complexity (degree={best_degree})")
ax1.set_xlabel("polynomial degree (model complexity)")
ax1.set_ylabel("mean squared error")
ax1.set_title("Bias-variance tradeoff: training error vs honest cross-validated error")
ax1.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Regularization taming an overfit high-degree polynomial
# ---------------------------------------------------------------------------
high_degree = 14
poly = PolynomialFeatures(degree=high_degree)
X_poly = poly.fit_transform(X)
X_poly_scaled = X_poly / X_poly.std(axis=0, keepdims=True).clip(min=1e-8)

x_plot = np.linspace(0, 8, 300).reshape(-1, 1)
X_plot_poly = poly.transform(x_plot) / X_poly.std(axis=0, keepdims=True).clip(min=1e-8)

fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.scatter(x, y, color="black", zorder=5, label="training data")
ax2.plot(x_plot, true_function(x_plot), "k--", label="true underlying function", linewidth=1)

for lam, color in zip([0, 0.01, 1, 100], ["red", "orange", "steelblue", "purple"]):
    if lam == 0:
        model = LinearRegression().fit(X_poly_scaled, y)
        label = "lambda=0 (no regularization, wild overfit)"
    else:
        model = Ridge(alpha=lam).fit(X_poly_scaled, y)
        label = f"lambda={lam} (Ridge)"
    pred = model.predict(X_plot_poly)
    ax2.plot(x_plot, pred, color=color, label=label)
    ax2.set_ylim(-6, 6)

ax2.set_title(f"Degree-{high_degree} polynomial: regularization taming overfitting")
ax2.legend(fontsize=9)
plt.tight_layout()
plt.show()
