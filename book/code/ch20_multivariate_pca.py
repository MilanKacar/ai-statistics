"""
Chapter 20 companion script: Multivariate Statistics and PCA

1) Visualize the Multivariate Normal's elliptical density contours.
2) Perform PCA "from scratch" via covariance matrix eigen-decomposition and
   confirm it matches sklearn's PCA exactly.
3) Plot principal component directions as arrows over the data.
4) Produce a scree plot on higher-dimensional data to illustrate the "elbow" rule.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

rng = np.random.default_rng(seed=37)

# ---------------------------------------------------------------------------
# 1) Multivariate Normal elliptical contours
# ---------------------------------------------------------------------------
mean = [0, 0]
cov = [[3, 2], [2, 2]]  # correlated covariance matrix
mvn = multivariate_normal(mean, cov)

x_grid, y_grid = np.meshgrid(np.linspace(-6, 6, 200), np.linspace(-6, 6, 200))
pos = np.dstack((x_grid, y_grid))
density = mvn.pdf(pos)

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.contour(x_grid, y_grid, density, levels=8, cmap="viridis")
ax1.set_title("Multivariate Normal density contours\n(ellipses, oriented by the covariance matrix)")
ax1.set_aspect("equal")
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) & 3) PCA from scratch vs sklearn, plus direction arrows
# ---------------------------------------------------------------------------
n = 300
data = rng.multivariate_normal(mean, cov, size=n)

data_centered = data - data.mean(axis=0)
cov_matrix = np.cov(data_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# eigh returns ascending order; reverse to descending (largest variance first)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

pca = PCA(n_components=2).fit(data)

print("=== Part 2: PCA from scratch vs sklearn ===")
print(f"From-scratch eigenvalues: {eigenvalues}")
print(f"sklearn explained_variance_: {pca.explained_variance_}")
print(f"From-scratch top eigenvector: {eigenvectors[:, 0]}")
print(f"sklearn first component:      {pca.components_[0]}")
print("(directions should match, possibly with a sign flip -- sign is arbitrary)")

fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.scatter(data[:, 0], data[:, 1], alpha=0.4, color="steelblue")
center = data.mean(axis=0)
for val, vec in zip(eigenvalues, eigenvectors.T):
    scale = 2 * np.sqrt(val)
    ax2.quiver(*center, *(vec * scale), angles="xy", scale_units="xy",
               scale=1, color="red", width=0.008)
ax2.set_title("Principal component directions (red arrows)\nlength proportional to variance explained")
ax2.set_aspect("equal")
plt.tight_layout()

# ---------------------------------------------------------------------------
# 4) Scree plot on higher-dimensional data
# ---------------------------------------------------------------------------
p = 10
# Construct data where only 3 underlying "true" factors drive most variance
true_factors = rng.normal(0, 1, size=(n, 3))
mixing_matrix = rng.normal(0, 1, size=(3, p))
high_dim_data = true_factors @ mixing_matrix + rng.normal(0, 0.3, size=(n, p))

pca_high = PCA(n_components=p).fit(high_dim_data)
explained_ratio = pca_high.explained_variance_ratio_
cumulative = np.cumsum(explained_ratio)

print()
print("=== Part 4: Scree plot data (10 features, 3 true underlying factors) ===")
for i, (r, c) in enumerate(zip(explained_ratio, cumulative), start=1):
    print(f"PC{i}: explains {r:.4f}, cumulative {c:.4f}")

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(range(1, p + 1), explained_ratio, color="steelblue", label="per-component")
ax3.plot(range(1, p + 1), cumulative, "o-", color="red", label="cumulative")
ax3.axhline(0.9, color="gray", linestyle="--", label="90% threshold")
ax3.set_xlabel("principal component")
ax3.set_ylabel("explained variance ratio")
ax3.set_title("Scree plot: 3 true factors show up as an 'elbow' after PC3")
ax3.legend()
plt.tight_layout()
plt.show()
