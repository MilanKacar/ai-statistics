"""
Chapter 23 companion script: Latent Variable Models (Gaussian Mixture + EM)

1) Generate bimodal data from two true hidden clusters.
2) Fit a single Normal via plain MLE -- show it badly misrepresents the shape.
3) Implement GMM's EM algorithm from scratch, watch parameters converge.
4) Confirm against sklearn.mixture.GaussianMixture.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(seed=47)

# ---------------------------------------------------------------------------
# Generate bimodal data (true labels hidden from the model)
# ---------------------------------------------------------------------------
n1, n2 = 300, 200
true_mu1, true_sigma1 = 165, 6
true_mu2, true_sigma2 = 178, 7

cluster1 = rng.normal(true_mu1, true_sigma1, n1)
cluster2 = rng.normal(true_mu2, true_sigma2, n2)
data = np.concatenate([cluster1, cluster2])
rng.shuffle(data)

# ---------------------------------------------------------------------------
# 1) Single Normal MLE fit (badly misrepresents bimodal shape)
# ---------------------------------------------------------------------------
single_mu, single_sigma = data.mean(), data.std()

print("=== Part 1: Single Normal MLE on bimodal data ===")
print(f"Single-Normal fit: mu={single_mu:.2f}, sigma={single_sigma:.2f}")
print("(this compromise curve will NOT capture the true two-hump shape)\n")

# ---------------------------------------------------------------------------
# 2) EM algorithm from scratch for a 2-component Gaussian mixture
# ---------------------------------------------------------------------------
def em_gmm(x, n_components=2, n_iters=50, seed=0):
    rng_local = np.random.default_rng(seed)
    n = len(x)
    # initialize randomly
    mus = rng_local.choice(x, size=n_components, replace=False).astype(float)
    sigmas = np.full(n_components, x.std())
    pis = np.full(n_components, 1 / n_components)

    for iteration in range(n_iters):
        # --- E-step: compute responsibilities ---
        responsibilities = np.zeros((n, n_components))
        for k in range(n_components):
            responsibilities[:, k] = pis[k] * norm.pdf(x, mus[k], sigmas[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # --- M-step: re-estimate parameters using weighted MLE ---
        Nk = responsibilities.sum(axis=0)
        for k in range(n_components):
            mus[k] = np.sum(responsibilities[:, k] * x) / Nk[k]
            sigmas[k] = np.sqrt(np.sum(responsibilities[:, k] * (x - mus[k]) ** 2) / Nk[k])
            pis[k] = Nk[k] / n

    return mus, sigmas, pis

mus_hand, sigmas_hand, pis_hand = em_gmm(data, n_components=2, n_iters=50)
order = np.argsort(mus_hand)

print("=== Part 2: EM from scratch ===")
print(f"True params:   mu=({true_mu1},{true_mu2}), sigma=({true_sigma1},{true_sigma2}), "
      f"pi=({n1/(n1+n2):.3f},{n2/(n1+n2):.3f})")
print(f"EM estimates:  mu=({mus_hand[order][0]:.2f},{mus_hand[order][1]:.2f}), "
      f"sigma=({sigmas_hand[order][0]:.2f},{sigmas_hand[order][1]:.2f}), "
      f"pi=({pis_hand[order][0]:.3f},{pis_hand[order][1]:.3f})")
print()

# ---------------------------------------------------------------------------
# 3) Confirm against sklearn
# ---------------------------------------------------------------------------
gmm_sklearn = GaussianMixture(n_components=2, random_state=0).fit(data.reshape(-1, 1))
sk_order = np.argsort(gmm_sklearn.means_.flatten())

print("=== Part 3: sklearn GaussianMixture ===")
print(f"sklearn means:  {gmm_sklearn.means_.flatten()[sk_order]}")
print(f"sklearn stds:   {np.sqrt(gmm_sklearn.covariances_.flatten())[sk_order]}")
print(f"sklearn weights:{gmm_sklearn.weights_[sk_order]}")

# ---------------------------------------------------------------------------
# Plot: data histogram, single-Normal fit, GMM fit
# ---------------------------------------------------------------------------
x_grid = np.linspace(data.min() - 5, data.max() + 5, 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(data, bins=40, density=True, color="lightsteelblue", edgecolor="white")
axes[0].plot(x_grid, norm.pdf(x_grid, single_mu, single_sigma), color="red", linewidth=2)
axes[0].set_title("Single Normal MLE fit\n(misses the true bimodal shape)")

axes[1].hist(data, bins=40, density=True, color="lightsteelblue", edgecolor="white")
mixture_density = sum(
    pis_hand[k] * norm.pdf(x_grid, mus_hand[k], sigmas_hand[k])
    for k in range(2)
)
axes[1].plot(x_grid, mixture_density, color="green", linewidth=2, label="fitted mixture")
for k in range(2):
    axes[1].plot(x_grid, pis_hand[k] * norm.pdf(x_grid, mus_hand[k], sigmas_hand[k]),
                 "--", label=f"component {k+1}")
axes[1].set_title("2-component Gaussian Mixture fit via EM\n(recovers the true two clusters)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
