"""
Chapter 19 companion script: Linear Algebra Essentials

1) Visualize vector addition and the dot-product/cosine-angle relationship.
2) Show correlation = cosine of the angle between two centered data vectors.
3) Verify the (X^T X)^-1 X^T y normal-equation formula against sklearn.
4) Visualize eigenvectors of a covariance matrix as arrows on correlated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(seed=31)

# ---------------------------------------------------------------------------
# 1) Vector addition + dot product / cosine angle
# ---------------------------------------------------------------------------
a = np.array([3, 4])
b = np.array([1, -2])
a_plus_b = a + b

dot = np.dot(a, b)
cos_theta = dot / (np.linalg.norm(a) * np.linalg.norm(b))
angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

print("=== Part 1: Vectors ===")
print(f"a={a}, b={b}, a+b={a_plus_b}")
print(f"dot product a.b = {dot}")
print(f"angle between a and b: {angle_deg:.2f} degrees (cos={cos_theta:.4f})")

fig1, ax1 = plt.subplots(figsize=(6, 6))
origin = np.array([0, 0])
ax1.quiver(*origin, *a, angles="xy", scale_units="xy", scale=1, color="steelblue", label="a")
ax1.quiver(*origin, *b, angles="xy", scale_units="xy", scale=1, color="darkorange", label="b")
ax1.quiver(*origin, *a_plus_b, angles="xy", scale_units="xy", scale=1, color="green", label="a+b")
ax1.set_xlim(-3, 7)
ax1.set_ylim(-3, 7)
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.axvline(0, color="gray", linewidth=0.5)
ax1.set_title(f"Vector addition; angle(a,b)={angle_deg:.1f} deg")
ax1.legend()
ax1.set_aspect("equal")
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Correlation = cosine of angle between centered vectors
# ---------------------------------------------------------------------------
n = 50
x = rng.normal(0, 1, n)
y = 0.7 * x + rng.normal(0, 0.7, n)

x_centered = x - x.mean()
y_centered = y - y.mean()

corr_direct = np.corrcoef(x, y)[0, 1]
corr_as_cosine = np.dot(x_centered, y_centered) / (
    np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
)

print()
print("=== Part 2: Correlation as cosine of angle ===")
print(f"np.corrcoef:         {corr_direct:.6f}")
print(f"dot-product/cosine:  {corr_as_cosine:.6f}  (should match)")

# ---------------------------------------------------------------------------
# 3) Normal equations vs sklearn
# ---------------------------------------------------------------------------
X_design = np.column_stack([np.ones(n), x])  # add intercept column
beta_hand = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y

model = LinearRegression().fit(x.reshape(-1, 1), y)

print()
print("=== Part 3: Normal equations vs sklearn ===")
print(f"Hand (X^T X)^-1 X^T y: intercept={beta_hand[0]:.4f}, slope={beta_hand[1]:.4f}")
print(f"sklearn LinearRegression: intercept={model.intercept_:.4f}, slope={model.coef_[0]:.4f}")

# ---------------------------------------------------------------------------
# 4) Eigenvectors of a covariance matrix
# ---------------------------------------------------------------------------
cov_matrix = np.cov(x, y)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print()
print("=== Part 4: Eigenvectors of covariance matrix ===")
print(f"Covariance matrix:\n{cov_matrix}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.scatter(x, y, alpha=0.5, color="steelblue")
mean_point = np.array([x.mean(), y.mean()])
for val, vec in zip(eigenvalues, eigenvectors.T):
    scale = 2 * np.sqrt(val)
    ax2.quiver(*mean_point, *(vec * scale), angles="xy", scale_units="xy",
               scale=1, color="red", width=0.008)
ax2.set_title("Eigenvectors of the covariance matrix\n(point along the data's natural axes of spread)")
ax2.set_aspect("equal")
plt.tight_layout()
plt.show()
