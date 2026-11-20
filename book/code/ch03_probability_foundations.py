"""
Chapter 3 companion script: Probability Foundations

1) Law of Large Numbers: empirical frequency of a die roll converges to 1/6.
2) Verify the King / face-card conditional probability example by brute-force
   enumeration over an actual simulated deck of cards.
3) Monty Hall simulation: switching wins ~2/3 of the time, staying wins ~1/3.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)

# ---------------------------------------------------------------------------
# 1) Law of Large Numbers
# ---------------------------------------------------------------------------
n_rolls = 20_000
rolls = rng.integers(1, 7, size=n_rolls)  # fair die, values 1-6
running_freq_of_4 = np.cumsum(rolls == 4) / np.arange(1, n_rolls + 1)

print("=== Part 1: Law of Large Numbers ===")
print(f"Empirical P(roll=4) after {n_rolls} rolls: {running_freq_of_4[-1]:.4f} "
      f"(theoretical = {1/6:.4f})")

plt.figure(figsize=(8, 4))
plt.plot(running_freq_of_4, color="steelblue")
plt.axhline(1 / 6, color="red", linestyle="--", label="theoretical P = 1/6")
plt.xscale("log")
plt.xlabel("number of rolls (log scale)")
plt.ylabel("running empirical P(roll = 4)")
plt.title("Law of Large Numbers: empirical frequency converges to true probability")
plt.legend()
plt.tight_layout()

# ---------------------------------------------------------------------------
# 2) King / face-card conditional probability, verified by brute force
# ---------------------------------------------------------------------------
ranks = ["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
deck = [(r, s) for s in suits for r in ranks]
assert len(deck) == 52

is_king = lambda card: card[0] == "K"
is_face = lambda card: card[0] in ("J", "Q", "K")

A = [c for c in deck if is_king(c)]          # event A: King
B = [c for c in deck if is_face(c)]          # event B: face card
A_and_B = [c for c in deck if is_king(c) and is_face(c)]

P_A = len(A) / len(deck)
P_B = len(B) / len(deck)
P_A_and_B = len(A_and_B) / len(deck)

P_B_given_A = P_A_and_B / P_A   # should be 1.0
P_A_given_B = P_A_and_B / P_B   # should be 1/3

print()
print("=== Part 2: Conditional probability with a card deck ===")
print(f"P(King) = {P_A:.4f}")
print(f"P(face card) = {P_B:.4f}")
print(f"P(face card | King) = {P_B_given_A:.4f}  <- should be 1.0")
print(f"P(King | face card) = {P_A_given_B:.4f}  <- should be ~0.333")

# ---------------------------------------------------------------------------
# 3) Monty Hall simulation
# ---------------------------------------------------------------------------
def monty_hall_trial(switch: bool, rng) -> bool:
    doors = [0, 1, 2]
    car = rng.choice(doors)
    pick = rng.choice(doors)
    # host opens a door that is neither the car nor the player's pick
    openable = [d for d in doors if d != car and d != pick]
    host_opens = rng.choice(openable)
    if switch:
        final_pick = [d for d in doors if d != pick and d != host_opens][0]
    else:
        final_pick = pick
    return final_pick == car

n_trials = 10_000
wins_switch = sum(monty_hall_trial(switch=True, rng=rng) for _ in range(n_trials))
wins_stay = sum(monty_hall_trial(switch=False, rng=rng) for _ in range(n_trials))

print()
print("=== Part 3: Monty Hall, 10,000 simulated games each strategy ===")
print(f"Win rate if you ALWAYS SWITCH: {wins_switch / n_trials:.3f}  (theory: 0.667)")
print(f"Win rate if you ALWAYS STAY:   {wins_stay / n_trials:.3f}  (theory: 0.333)")

plt.show()
