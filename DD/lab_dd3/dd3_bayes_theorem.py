"""
LAB DD.3 — Bayes' Theorem: the Funnel, the Bomb Search & the Spam Filter
========================================================================
Companion notebook to "Bayes' Theorem: How to Update What You Believe".

pip install numpy        # that's it

You will verify the three set-pieces from the video with your own hands: the medical-test
funnel (why a positive result from a 99/99 test is a coin flip), the Navy's Bayesian
search algorithm (negative evidence reshaping a probability map - including WHY murky
water teaches you less than clear water), and a working naive-Bayes spam filter that
adds evidence in Turing's decibans. Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - THE FUNNEL: WHY A POSITIVE TEST IS A COIN FLIP ===
# 🎭 Analogy: the stadium scanner - one pickpocket, hundreds of false alarms.
# Define the test HONESTLY: prevalence, sensitivity, specificity (never "accuracy").
import numpy as np

N = 10_000
prevalence = 0.01        # 1 in 100 actually sick
sensitivity = 0.99       # of the sick, how many the test catches
specificity = 0.99       # of the healthy, how many the test correctly clears

sick = N * prevalence                                  # 100
true_pos = sick * sensitivity                          # 99
false_pos = (N - sick) * (1 - specificity)             # 9,900 * 0.01 = 99
p_sick_given_pos = true_pos / (true_pos + false_pos)
print(f"true positives : {true_pos:.0f}")
print(f"false positives: {false_pos:.0f}")
print(f"P(sick | positive) = {true_pos:.0f}/{true_pos + false_pos:.0f} "
      f"= {p_sick_given_pos:.3f}   <- a coin flip, not 99%")
assert abs(p_sick_given_pos - 0.5) < 0.01

# the same answer via Bayes' rule directly (the equation IS the funnel, sideways)
p_pos = prevalence * sensitivity + (1 - prevalence) * (1 - specificity)
bayes = prevalence * sensitivity / p_pos
assert abs(bayes - p_sick_given_pos) < 1e-12


# === SECTION 2 - PREVALENCE DOMINATES: SWEEP IT AND WATCH ===
# Same excellent test; only the base rate changes. Belief follows the base rate.
for prev in (0.10, 0.01, 0.001):
    p = prev * sensitivity / (prev * sensitivity + (1 - prev) * (1 - specificity))
    print(f"  prevalence {prev:>6.1%}  ->  P(sick | +) = {p:6.1%}")
# rare disease -> even a great test mostly finds false alarms


# === SECTION 3 - THE NAVY ALGORITHM: A PROBABILITY MAP THAT LEARNS ===
# The Palomares search: a prior grid, then every EMPTY search cell is evidence.
# grid *= (1 - P(find|there)) in the searched cell, renormalize, repeat.
rows, cols = 5, 8
grid = np.ones((rows, cols)) * 0.5
grid[1, 5], grid[1, 4], grid[2, 5], grid[0, 5] = 6.0, 3.0, 3.0, 2.0   # the sighting
grid /= grid.sum()
P_FIND = 0.85                                          # a good sensor in clear water

def search(grid, cell, p_find=P_FIND):
    """Bayes update after searching `cell` and finding NOTHING."""
    g = grid.copy()
    g[cell] *= (1 - p_find)                            # miss likelihood
    return g / g.sum()

plan = [(1, 4), (0, 5), (2, 5)]
for cell in plan:
    grid = search(grid, cell)
    print(f"searched {cell}: next-best cell -> {np.unravel_index(grid.argmax(), grid.shape)}"
          f"   P = {grid.max():.2f}")
assert np.unravel_index(grid.argmax(), grid.shape) == (1, 5)   # the map converges


# === SECTION 4 - NEGATIVE EVIDENCE NEEDS P(find | there) ===
# "Not finding something is also evidence - but ONLY if you know how likely you
# were to find it." Same empty search, two sensors: the update strengths differ.
fresh = np.ones((1, 2)); fresh[0, 0] = 3.0; fresh = fresh / fresh.sum()   # cell 0 favored
murky = search(fresh, (0, 0), p_find=0.20)             # murky water: weak evidence
clear = search(fresh, (0, 0), p_find=0.90)             # clear water: strong evidence
print(f"prior on cell 0        : {fresh[0,0]:.2f}")
print(f"after MURKY empty search: {murky[0,0]:.2f}   (barely moved)")
print(f"after CLEAR empty search: {clear[0,0]:.2f}   (collapsed)")
assert murky[0, 0] > 0.5 > clear[0, 0]


# === SECTION 5 - THE SPAM FILTER IN DECIBANS (naive Bayes = Turing's arithmetic) ===
# Each word's evidence = 10*log10( P(word|spam) / P(word|ham) ) decibans. ADD them.
# Toy per-word rates (real filters learn these from corpora; the MATH is exact).
rates = {                # word: (P(word|spam), P(word|ham))
    "free":     (0.40, 0.10),
    "offer":    (0.25, 0.10),
    "!!!":      (0.32, 0.10),
    "meeting":  (0.03, 0.10),
    "tomorrow": (0.06, 0.10),
}
def decibans(word):
    ps, ph = rates[word]
    return 10 * np.log10(ps / ph)

email = ["free", "offer", "!!!", "meeting", "tomorrow"]
total = 0.0
for w in email:
    db = decibans(w)
    total += db
    print(f"  {w:<9} {db:+6.1f} db   running total {total:+6.1f} db")
THRESHOLD = 5.0
verdict = "SPAM" if total > THRESHOLD else "HAM"
print(f"verdict: {verdict}  ({total:+.1f} db vs threshold {THRESHOLD} db)")
assert verdict == "SPAM"

# decibans back to probability (starting from 50/50): odds = 10^(db/10)
odds = 10 ** (total / 10)
print(f"P(spam | words) = {odds / (1 + odds):.2%}")


# === SECTION 6 - SOFTMAX IS BAYES, NUMERICALLY ===
# Class scores z_c = log P(x|c)P(c). Bayes' posterior == softmax(z). Verify.
rng = np.random.default_rng(3)
joint = rng.uniform(0.05, 1.0, 4)                      # P(x|c) P(c) for 4 classes
posterior_bayes = joint / joint.sum()
z = np.log(joint)                                      # the "logits"
posterior_softmax = np.exp(z) / np.exp(z).sum()
print("Bayes  :", posterior_bayes.round(4))
print("softmax:", posterior_softmax.round(4))
assert np.allclose(posterior_bayes, posterior_softmax)
print("\nLAB COMPLETE - the funnel, the Navy's map, negative evidence, decibans,")
print("and softmax revealed as Bayes' theorem, exponentiated.")
