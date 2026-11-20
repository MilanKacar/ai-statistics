"""
Chapter 22 companion script: Statistical Learning Theory

Generate data that is PURE RANDOM NOISE (x and y are totally unrelated).
Fit increasingly flexible polynomial models and track:
  - training error (always improves with more flexibility, even on pure noise)
  - true held-out error (should NOT improve, since there's no real signal)
The widening gap between the two is overfitting, directly tied to model
"capacity" (here, polynomial degree, a stand-in for VC-dimension-style capacity).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(seed=43)

# ---------------------------------------------------------------------------
# Pure noise: y has NO real relationship to x
# ---------------------------------------------------------------------------
n_train = 30
x_train = rng.uniform(0, 10, n_train)
y_train = rng.normal(0, 1, n_train)  # completely independent of x_train!

n_test = 10_000  # large "population" stand-in for true generalization error
x_test = rng.uniform(0, 10, n_test)
y_test = rng.normal(0, 1, n_test)

degrees = range(1, 20)
train_errors, test_errors = [], []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(x_train.reshape(-1, 1))
    X_test_poly = poly.transform(x_test.reshape(-1, 1))

    model = LinearRegression().fit(X_train_poly, y_train)

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    train_mse = np.mean((y_train - train_pred) ** 2)
    test_mse = np.mean((y_test - test_pred) ** 2)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

print("=== Fitting pure noise with increasingly flexible models ===")
print("(y has NO real relationship to x -- true error should be ~1.0 always,")
print(" since that's just the variance of y_test itself)\n")
for d, tr, te in zip(degrees, train_errors, test_errors):
    print(f"degree={d:>2}: train MSE={tr:.4f}, TRUE held-out MSE={te:.4f}, "
          f"gap={te - tr:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(degrees, train_errors, "o-", color="red", label="training error (fools you into thinking it's improving)")
ax.plot(degrees, test_errors, "o-", color="steelblue", label="true generalization error (not actually improving)")
ax.axhline(1.0, color="gray", linestyle=":", label="true irreducible error (Var(y)=1)")
ax.set_xlabel("polynomial degree (model capacity)")
ax.set_ylabel("mean squared error")
ax.set_title("Fitting PURE NOISE: training error keeps dropping,\ntrue error does not -- the overfitting gap widens with capacity")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
