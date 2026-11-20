"""
Chapter 25 companion script: Statistics Inside Neural Networks

1) Train a tiny neural network (pure numpy, fully visible) on a toy binary
   classification problem; confirm its loss equals the negative
   log-likelihood formula from Chapters 10/21.
2) Plot a calibration/reliability diagram of its softmax-style output.
3) Demonstrate why initialization variance matters: compare activation
   magnitudes through several layers under good vs poor init scaling.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=59)

# ---------------------------------------------------------------------------
# 1) Tiny neural network trained via gradient descent on binary cross-entropy
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

n = 300
X = rng.normal(0, 1, size=(n, 2))
true_w = np.array([2.0, -3.0])
logits_true = X @ true_w
y = (sigmoid(logits_true) > rng.random(n)).astype(float)

# simple 1-layer network: logistic regression (a "network" with no hidden layer,
# kept simple so every number is traceable by hand)
w = rng.normal(0, 0.1, size=2)
b = 0.0
lr = 0.1

losses = []
for epoch in range(500):
    z = X @ w + b
    p = sigmoid(z)

    # binary cross-entropy loss (clip for numerical stability)
    p_clipped = np.clip(p, 1e-12, 1 - 1e-12)
    bce = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    losses.append(bce)

    # gradient of BCE w.r.t. logits is simply (p - y) -- a clean, famous result
    grad_z = (p - y) / n
    grad_w = X.T @ grad_z
    grad_b = grad_z.sum()

    w -= lr * grad_w
    b -= lr * grad_b

# Manually recompute negative log-likelihood the Chapter 10 way, confirm match
nll_manual = -np.mean([
    np.log(p_clipped[i]) if y[i] == 1 else np.log(1 - p_clipped[i])
    for i in range(n)
])

print("=== Part 1: Training loss == negative log-likelihood ===")
print(f"Final training BCE loss:        {losses[-1]:.6f}")
print(f"Manually recomputed NLL:        {nll_manual:.6f}  (should match)")
print(f"Learned weights: {w}, learned bias: {b:.4f} (true weights were {true_w})")
print()

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(losses, color="steelblue")
ax1.set_xlabel("training epoch")
ax1.set_ylabel("binary cross-entropy loss")
ax1.set_title("Training loss curve (this IS negative log-likelihood, every step)")
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Calibration / reliability diagram
# ---------------------------------------------------------------------------
final_probs = sigmoid(X @ w + b)
bins = np.linspace(0, 1, 11)
bin_indices = np.digitize(final_probs, bins) - 1
bin_indices = np.clip(bin_indices, 0, 9)

bin_confidence, bin_accuracy = [], []
for b_idx in range(10):
    mask = bin_indices == b_idx
    if mask.sum() > 0:
        bin_confidence.append(final_probs[mask].mean())
        bin_accuracy.append(y[mask].mean())

print("=== Part 2: Calibration ===")
print("confidence vs actual accuracy per bin (perfectly calibrated = diagonal):")
for c, a in zip(bin_confidence, bin_accuracy):
    print(f"  predicted confidence ~{c:.2f} -> actual accuracy {a:.2f}")

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax2.plot(bin_confidence, bin_accuracy, "o-", color="darkorange", label="this model")
ax2.set_xlabel("predicted confidence")
ax2.set_ylabel("actual fraction positive")
ax2.set_title("Calibration / reliability diagram")
ax2.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 3) Why initialization variance matters: activation magnitude through layers
# ---------------------------------------------------------------------------
def propagate(init_std, n_layers=10, width=100, n_samples=500, rng=None):
    activations = rng.normal(0, 1, size=(n_samples, width))
    magnitudes = [activations.std()]
    for _ in range(n_layers):
        W = rng.normal(0, init_std, size=(width, width))
        activations = np.tanh(activations @ W)
        magnitudes.append(activations.std())
    return magnitudes

rng3 = np.random.default_rng(seed=61)
width = 100
good_std = 1 / np.sqrt(width)       # roughly "Xavier"-style scaling
bad_std_large = 1.0                 # too large
bad_std_small = 1 / (10 * np.sqrt(width))  # too small

mags_good = propagate(good_std, rng=rng3)
mags_large = propagate(bad_std_large, rng=rng3)
mags_small = propagate(bad_std_small, rng=rng3)

print()
print("=== Part 3: Activation magnitude through layers, by init std ===")
print(f"Good scaling (1/sqrt(width)):      {[f'{m:.3f}' for m in mags_good]}")
print(f"Too-large scaling (std=1.0):        {[f'{m:.3f}' for m in mags_large]}")
print(f"Too-small scaling (0.1/sqrt(width)):{[f'{m:.3f}' for m in mags_small]}")

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(mags_good, "o-", label="good init scaling")
ax3.plot(mags_large, "o-", label="too-large init (can saturate/explode)")
ax3.plot(mags_small, "o-", label="too-small init (vanishes)")
ax3.set_xlabel("layer depth")
ax3.set_ylabel("activation std dev")
ax3.set_title("Why initialization variance matters:\nsignal magnitude through a deep network")
ax3.legend()
plt.tight_layout()
plt.show()
