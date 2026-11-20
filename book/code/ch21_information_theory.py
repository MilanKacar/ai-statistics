"""
Chapter 21 companion script: Information Theory

1) Plot entropy of a biased coin as a function of P(heads); confirm max at 0.5.
2) Compute cross-entropy and KL divergence between a true distribution and
   several candidate model distributions, showing KL -> 0 as model -> truth.
3) Demonstrate that binary cross-entropy loss on real predictions equals the
   negative log-likelihood formula from Chapter 10.
"""

import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """Entropy of a Bernoulli(p) distribution, in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

# ---------------------------------------------------------------------------
# 1) Entropy of a biased coin
# ---------------------------------------------------------------------------
p_values = np.linspace(0.001, 0.999, 500)
h_values = entropy(p_values)

print("=== Part 1: Entropy examples ===")
for p in [0.5, 0.99, 0.01, 0.9]:
    print(f"P(heads)={p}: entropy = {entropy(p):.4f} bits")
print(f"Max entropy occurs at p={p_values[np.argmax(h_values)]:.3f} (should be 0.5)")
print()

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(p_values, h_values, color="steelblue")
ax1.axvline(0.5, color="red", linestyle="--", label="max entropy at p=0.5")
ax1.set_xlabel("P(heads)")
ax1.set_ylabel("entropy (bits)")
ax1.set_title("Entropy of a biased coin: maximized exactly at p=0.5")
ax1.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Cross-entropy and KL divergence: true vs model distributions
# ---------------------------------------------------------------------------
# A 4-outcome true distribution (e.g., 4 classes)
true_dist = np.array([0.4, 0.3, 0.2, 0.1])

def cross_entropy(p_true, q_model):
    return -np.sum(p_true * np.log(q_model))

def kl_divergence(p_true, q_model):
    return np.sum(p_true * np.log(p_true / q_model))

true_entropy = cross_entropy(true_dist, true_dist)  # H(P,P) = H(P)

candidates = {
    "perfect model (Q=P)": true_dist.copy(),
    "close model": np.array([0.35, 0.3, 0.25, 0.1]),
    "mediocre model (uniform)": np.array([0.25, 0.25, 0.25, 0.25]),
    "bad model (reversed)": np.array([0.1, 0.2, 0.3, 0.4]),
}

print("=== Part 2: Cross-entropy and KL divergence ===")
print(f"True distribution: {true_dist}, H(P) = {true_entropy:.4f} nats\n")
for name, q in candidates.items():
    ce = cross_entropy(true_dist, q)
    kl = kl_divergence(true_dist, q)
    print(f"{name:<28}: cross-entropy={ce:.4f}, KL divergence={kl:.4f} "
          f"(cross-entropy - H(P) = {ce - true_entropy:.4f}, should equal KL)")
print()

# ---------------------------------------------------------------------------
# 3) Binary cross-entropy loss == negative log-likelihood (Chapter 10 link)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(seed=41)
n = 10
true_labels = rng.integers(0, 2, n)
predicted_probs = rng.uniform(0.05, 0.95, n)

bce_loss = -np.mean(
    true_labels * np.log(predicted_probs) + (1 - true_labels) * np.log(1 - predicted_probs)
)

# Negative log-likelihood, computed the Chapter 10 way:
nll = -np.mean([
    np.log(predicted_probs[i]) if true_labels[i] == 1 else np.log(1 - predicted_probs[i])
    for i in range(n)
])

print("=== Part 3: Binary cross-entropy == negative log-likelihood ===")
print(f"true labels:       {true_labels}")
print(f"predicted probs:   {np.round(predicted_probs, 3)}")
print(f"binary cross-entropy loss: {bce_loss:.6f}")
print(f"negative log-likelihood:   {nll:.6f}  (should match exactly)")

plt.show()
