"""
Chapter 27 companion script: Best-of-N, the Winner's Curse, and Self-Consistency

1) Simulate best-of-N selection using a NOISY reward model vs TRUE quality,
   and show the gap between "what the reward model thought" and "how good it
   actually was" -- the winner's curse / reward hacking, growing with N.
2) Simulate self-consistency majority voting on a toy correct-answer problem,
   showing accuracy climbing smoothly with N (no curse).
3) Confirm the order-statistics formula E[max of N draws] ~ sqrt(2 ln N).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=101)

# ---------------------------------------------------------------------------
# 1) Winner's curse in best-of-N reward-model selection
# ---------------------------------------------------------------------------
n_trials = 2000
reward_noise_std = 1.0  # how imperfect the reward model is relative to true quality

Ns = [1, 2, 5, 10, 20, 50, 100, 200]
mean_selected_true_quality = []
mean_selected_predicted_score = []

for N in Ns:
    true_q = rng.normal(0, 1, size=(n_trials, N))          # true quality of each candidate
    reward_score = true_q + rng.normal(0, reward_noise_std, size=(n_trials, N))  # noisy judge

    best_idx = np.argmax(reward_score, axis=1)
    selected_true_q = true_q[np.arange(n_trials), best_idx]
    selected_pred_score = reward_score[np.arange(n_trials), best_idx]

    mean_selected_true_quality.append(selected_true_q.mean())
    mean_selected_predicted_score.append(selected_pred_score.mean())

print("=== Part 1: Winner's curse in best-of-N selection ===")
print(f"{'N':>5} {'reward model THOUGHT it got':>28} {'ACTUAL true quality':>22} {'gap (the curse)':>18}")
for N, pred, true in zip(Ns, mean_selected_predicted_score, mean_selected_true_quality):
    print(f"{N:>5} {pred:>28.3f} {true:>22.3f} {pred-true:>18.3f}")

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(Ns, mean_selected_predicted_score, "o-", color="#E45756", label="reward model's score for its pick (inflated)")
ax1.plot(Ns, mean_selected_true_quality, "o-", color="#4C78A8", label="that pick's ACTUAL true quality")
ax1.set_xscale("log")
ax1.set_xlabel("N (number of candidates sampled)")
ax1.set_ylabel("quality score")
ax1.set_title("The winner's curse: picking the best-SCORING candidate\nis not the same as picking the best candidate")
ax1.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) Self-consistency majority voting -- no curse
# ---------------------------------------------------------------------------
true_answer_prob = 0.4  # model's per-sample chance of landing on the correct answer
n_wrong_answers = 5     # several plausible wrong answers splitting the remaining probability

def simulate_majority_vote(N, n_trials, rng):
    correct_count = 0
    for _ in range(n_trials):
        samples = rng.choice(
            np.arange(n_wrong_answers + 1),
            size=N,
            p=[true_answer_prob] + [(1 - true_answer_prob) / n_wrong_answers] * n_wrong_answers,
        )
        values, counts = np.unique(samples, return_counts=True)
        majority = values[np.argmax(counts)]
        if majority == 0:  # 0 = correct answer
            correct_count += 1
    return correct_count / n_trials

Ns_vote = [1, 3, 5, 11, 21, 41, 81]
accuracies = [simulate_majority_vote(N, 500, rng) for N in Ns_vote]

print()
print("=== Part 2: Self-consistency majority voting ===")
print(f"Per-sample accuracy: {true_answer_prob}")
for N, acc in zip(Ns_vote, accuracies):
    print(f"N={N:>3}: majority-vote accuracy = {acc:.3f}")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(Ns_vote, accuracies, "o-", color="#54A24B")
ax2.axhline(true_answer_prob, color="#888", linestyle="--", label="single-sample accuracy")
ax2.set_xlabel("N (number of independent reasoning paths)")
ax2.set_ylabel("majority-vote accuracy")
ax2.set_title("Self-consistency: accuracy climbs smoothly with N,\nno reward model to hack")
ax2.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 3) Order statistics: E[max of N standard Normal draws] ~ sqrt(2 ln N)
# ---------------------------------------------------------------------------
Ns_max = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
simulated_max = []
for N in Ns_max:
    draws = rng.normal(0, 1, size=(3000, N))
    simulated_max.append(draws.max(axis=1).mean())

def expected_max_approx(N):
    # Leading-order term sqrt(2 ln N), refined with the standard next-order
    # log-log correction for a closer match at moderate N (the leading term
    # alone is the headline intuition; this refinement is just for a tighter
    # numerical match in the plot below).
    if N <= 1:
        return 0.0
    leading = np.sqrt(2 * np.log(N))
    correction = (np.log(np.log(N)) + np.log(4 * np.pi)) / (2 * leading) if N > np.e else 0.0
    return leading - correction

theoretical = [expected_max_approx(N) for N in Ns_max]

print()
print("=== Part 3: Expected max of N standard Normal draws ===")
for N, sim, theo in zip(Ns_max, simulated_max, theoretical):
    print(f"N={N:>5}: simulated E[max]={sim:.3f}, refined approx={theo:.3f}")

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(Ns_max, simulated_max, "o-", color="#4C78A8", label="simulated E[max of N draws]")
ax3.plot(Ns_max, theoretical, "--", color="#E45756", label="approx sqrt(2 ln N)")
ax3.set_xscale("log")
ax3.set_xlabel("N (log scale)")
ax3.set_ylabel("expected maximum")
ax3.set_title("Diminishing returns of sampling more:\nthe best-of-N grows only like sqrt(log N)")
ax3.legend()
plt.tight_layout()
plt.show()
