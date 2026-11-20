"""
Chapter 24 companion script: Markov Chains and Monte Carlo

1) Simulate the Sunny/Rainy weather Markov chain from different starting
   points and confirm convergence to the same stationary distribution (2/3, 1/3).
2) Implement Metropolis-Hastings from scratch to sample from a distribution
   known only up to its normalizing constant, and compare to the true shape.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(seed=53)

# ---------------------------------------------------------------------------
# 1) Markov chain: weather example
# ---------------------------------------------------------------------------
# states: 0 = Sunny, 1 = Rainy
transition = np.array([
    [0.8, 0.2],   # from Sunny
    [0.4, 0.6],   # from Rainy
])

def simulate_chain(start_state, n_steps, rng):
    states = [start_state]
    for _ in range(n_steps):
        probs = transition[states[-1]]
        next_state = rng.choice([0, 1], p=probs)
        states.append(next_state)
    return np.array(states)

n_steps = 20_000
chain_from_sunny = simulate_chain(0, n_steps, rng)
chain_from_rainy = simulate_chain(1, n_steps, rng)

print("=== Part 1: Markov chain stationary distribution ===")
print(f"Starting Sunny -> long-run P(Sunny) = {np.mean(chain_from_sunny == 0):.4f}")
print(f"Starting Rainy -> long-run P(Sunny) = {np.mean(chain_from_rainy == 0):.4f}")
print(f"Theoretical stationary P(Sunny) = 2/3 = {2/3:.4f}")
print()

running_sunny_from_sunny = np.cumsum(chain_from_sunny == 0) / np.arange(1, n_steps + 2)
running_sunny_from_rainy = np.cumsum(chain_from_rainy == 0) / np.arange(1, n_steps + 2)

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(running_sunny_from_sunny, label="started Sunny")
ax1.plot(running_sunny_from_rainy, label="started Rainy")
ax1.axhline(2 / 3, color="red", linestyle="--", label="theoretical stationary P(Sunny)=2/3")
ax1.set_xscale("log")
ax1.set_xlabel("step (log scale)")
ax1.set_ylabel("running fraction of Sunny days")
ax1.set_title("Markov chain forgets its starting point,\nconverges to the same stationary distribution")
ax1.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Metropolis-Hastings sampling from an unnormalized target
# ---------------------------------------------------------------------------
# Target: a mixture of two Normals, but we only give the sampler the
# UNNORMALIZED density (no need to know the true normalizing constant)
def unnormalized_target(x):
    return 0.3 * np.exp(-0.5 * ((x - (-2)) / 1.0) ** 2) + \
           0.7 * np.exp(-0.5 * ((x - 3) / 1.5) ** 2)

def metropolis_hastings(n_samples, proposal_std, rng):
    samples = np.zeros(n_samples)
    current = 0.0
    for i in range(n_samples):
        proposal = current + rng.normal(0, proposal_std)
        acceptance_ratio = unnormalized_target(proposal) / unnormalized_target(current)
        if rng.random() < min(1, acceptance_ratio):
            current = proposal
        samples[i] = current
    return samples

n_samples = 50_000
mcmc_samples = metropolis_hastings(n_samples, proposal_std=2.0, rng=rng)
burn_in = 2000  # discard early samples before the chain has "settled in"
mcmc_samples = mcmc_samples[burn_in:]

print("=== Part 2: Metropolis-Hastings sampling ===")
print(f"Collected {len(mcmc_samples)} samples (after burn-in) from an "
      f"unnormalized target density")

x_grid = np.linspace(-8, 8, 500)
true_density_unnormalized = unnormalized_target(x_grid)
# normalize numerically just for plotting comparison (MCMC itself never needed this)
normalizing_constant = np.trapz(true_density_unnormalized, x_grid)
true_density_normalized = true_density_unnormalized / normalizing_constant

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(mcmc_samples, bins=80, density=True, color="lightsteelblue",
         edgecolor="white", label="MCMC samples")
ax2.plot(x_grid, true_density_normalized, color="red", linewidth=2,
         label="true density (normalized, for comparison only)")
ax2.set_title("Metropolis-Hastings recovers the target distribution's shape\n"
              "without ever computing its normalizing constant")
ax2.legend()
plt.tight_layout()
plt.show()
