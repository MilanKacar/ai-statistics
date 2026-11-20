"""
Chapter 4 companion script: Bayes' Theorem

1) Recompute the medical test example by formula.
2) Simulate 200,000 patients to confirm the result empirically.
3) Sweep the prior (disease prevalence) and plot how the posterior changes --
   this is the single most important intuition in the whole chapter.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)

# ---------------------------------------------------------------------------
# 1) Formula calculation
# ---------------------------------------------------------------------------
prior_disease = 0.01
true_positive_rate = 0.99   # P(positive | disease)
false_positive_rate = 0.01  # P(positive | no disease)

p_evidence = (true_positive_rate * prior_disease
              + false_positive_rate * (1 - prior_disease))
posterior = (true_positive_rate * prior_disease) / p_evidence

print("=== Part 1: Formula ===")
print(f"P(disease) prior:        {prior_disease}")
print(f"P(positive | disease):   {true_positive_rate}")
print(f"P(positive | no disease):{false_positive_rate}")
print(f"P(positive) [evidence]:  {p_evidence:.4f}")
print(f"P(disease | positive):   {posterior:.4f}  <-- only ~50%, not 99%!")
print()

# ---------------------------------------------------------------------------
# 2) Simulate many patients to confirm empirically
# ---------------------------------------------------------------------------
n_patients = 200_000
has_disease = rng.random(n_patients) < prior_disease

test_positive = np.where(
    has_disease,
    rng.random(n_patients) < true_positive_rate,   # sick patients
    rng.random(n_patients) < false_positive_rate,  # healthy patients
)

among_positive = has_disease[test_positive]
empirical_posterior = among_positive.mean()

print("=== Part 2: Simulation ===")
print(f"Total patients: {n_patients}")
print(f"Tested positive: {test_positive.sum()}")
print(f"Of those positive, actually sick: {among_positive.sum()} "
      f"({empirical_posterior:.4f}) <-- matches the formula above")
print()

# ---------------------------------------------------------------------------
# 3) Sweep the prior and plot the posterior
# ---------------------------------------------------------------------------
priors = np.linspace(0.0001, 1.0, 500)
posteriors = (true_positive_rate * priors) / (
    true_positive_rate * priors + false_positive_rate * (1 - priors)
)

plt.figure(figsize=(8, 5))
plt.plot(priors, posteriors, color="steelblue")
plt.axvline(prior_disease, color="red", linestyle="--", label="our example: prior=0.01")
plt.axhline(posterior, color="red", linestyle=":")
plt.xlabel("Prior P(disease)  -- how rare the condition is")
plt.ylabel("Posterior P(disease | positive test)")
plt.title("How prevalence (prior) swamps a 'highly accurate' test")
plt.legend()
plt.tight_layout()
plt.show()
