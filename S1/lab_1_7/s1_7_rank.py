"""
LAB 1.7 — Rank: What a Matrix Keeps and What It Destroys
========================================================
Companion notebook to "Rank: What a Matrix Keeps and What It Destroys".

pip install numpy        # that's it

You will verify every claim from the episode: independence (a column that adds nothing),
the null space (many inputs -> ONE output, provably), the FIBERS (every output has a whole
line of ancestors, so the map cannot be inverted), the rank-nullity conservation law, the
dimension ladder (3 -> 2 -> 1 -> 0), and the payoff: real correlated feature tables are
low rank, which is the same idea LoRA bets on. Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - INDEPENDENCE: THE COLUMN THAT ADDS NOTHING ===
# 🎭 Analogy: the third hire who only repeats what the second already does. Headcount 3,
# real capability 2. That gap is rank.
import numpy as np

independent = np.array([[2.0, 1.0],
                        [1.0, 2.0]])          # two genuinely different directions
dependent = np.array([[2.0, 4.0],
                      [1.0, 2.0]])            # column 2 = 2 x column 1  (a copy in disguise)

print(f"rank(independent) = {np.linalg.matrix_rank(independent)}")
print(f"rank(dependent)   = {np.linalg.matrix_rank(dependent)}")
assert np.linalg.matrix_rank(independent) == 2
assert np.linalg.matrix_rank(dependent) == 1
# the 2D test from episode 1.4: a zero determinant means the columns collapsed
print(f"det(dependent) = {np.linalg.det(dependent):.1f}   (1.4's quick test agrees)")


# === SECTION 2 - THE NULL SPACE: MANY INPUTS, ONE OUTPUT ===
# Find the direction the matrix crushes, then prove several DIFFERENT inputs along it
# all land on exactly the same output.
def null_space(A, tol=1e-10):
    """An orthonormal basis for the null space, via SVD (the 1.9 preview)."""
    u, s, vh = np.linalg.svd(A)
    return vh[np.sum(s > tol):].T            # rows of vh past the nonzero singular values

ns = null_space(dependent)
print(f"null space basis:\n{ns.round(4)}")
n_dir = ns[:, 0]
for scale in (-2.0, -0.5, 1.0, 3.0):
    out = dependent @ (n_dir * scale)
    print(f"  input {np.round(n_dir * scale, 3)}  ->  output {np.round(out, 12)}")
    assert np.allclose(out, 0)               # every one of them is crushed to the origin


# === SECTION 3 - THE FIBERS: WHY YOU CANNOT RUN IT BACKWARDS ===
# The plane is sliced into PARALLEL lines; each whole line collapses to ONE point.
# So given an output, the input is not recoverable - there is a line of ancestors.
x0 = np.array([1.0, 0.5])                    # some particular input
target = dependent @ x0
ancestors = [x0 + n_dir * t for t in (-3, -1, 0, 1, 3)]
print(f"target output: {np.round(target, 6)}")
for a in ancestors:
    assert np.allclose(dependent @ a, target)     # ALL of them produce the same output
print(f"  {len(ancestors)} different inputs (a whole line) produce that same output")
try:
    np.linalg.inv(dependent)
    raise RuntimeError("unreachable")
except np.linalg.LinAlgError:
    print("  inverting it -> LinAlgError: you cannot un-shred the confetti")


# === SECTION 4 - RANK-NULLITY: THE CONSERVATION LAW ===
# rank + nullity = number of COLUMNS (input dimensions). Nothing is lost for free.
for name, A in [("independent", independent), ("dependent", dependent)]:
    r = np.linalg.matrix_rank(A)
    nullity = A.shape[1] - r
    print(f"  {name:<12} rank {r} + nullity {nullity} = {A.shape[1]} columns")
    assert r + nullity == A.shape[1]
    assert null_space(A).shape[1] == nullity      # the basis size MATCHES the nullity


# === SECTION 5 - THE DIMENSION LADDER (3 -> 2 -> 1 -> 0) ===
# Climb down one rung at a time and watch the budget trade kept-for-crushed.
rungs = {
    3: np.eye(3),                                        # keeps everything
    2: np.diag([1.0, 1.0, 0.0]),                         # squashes z -> a plane
    1: np.diag([1.0, 0.0, 0.0]),                         # -> a line
    0: np.zeros((3, 3)),                                 # -> a single point
}
for expected, A in rungs.items():
    r = np.linalg.matrix_rank(A)
    print(f"  rank {r} + nullity {3 - r} = 3")
    assert r == expected and r + (3 - r) == 3


# === SECTION 6 - THE PAYOFF: REAL FEATURE TABLES ARE LOW RANK ===
# Ten columns that LOOK independent: unit conversions and restatements are twins.
rng = np.random.default_rng(7)
n = 400
h_cm = rng.normal(170, 8, n)
weight = 0.5 * h_cm + rng.normal(0, 6, n)
age = rng.uniform(20, 60, n)
X = np.column_stack([
    h_cm,                     # 1  height (cm)
    h_cm / 2.54,              #    the SAME height, in inches
    h_cm * 10,                #    ...and in millimetres
    weight,                   # 2  weight
    weight * 2.20462,         #    the SAME weight, in pounds
    age,                      # 3  age
    age * 12,                 #    age in months
    age * 365.25,             #    age in days
    h_cm + weight,            #    a "sum" feature: just a mix of what we already have
    2 * age - 0.5 * weight,   #    another mix: still no new direction
])
r = np.linalg.matrix_rank(X)
print(f"\nX has {X.shape[1]} columns")
print(f"matrix_rank(X) = {r}   ->  only {r} genuinely independent directions")
print(f"rank {r} + nullity {X.shape[1] - r} = {X.shape[1]}")
assert X.shape[1] == 10 and r == 3          # ten columns, three real knobs (as in the video)

# BONUS - the honest caveat the video does not have room for: rank counts LINEAR
# redundancy only. BMI divides by height SQUARED, so it is not a linear combination -
# it genuinely adds a direction, even though it feels redundant to a human.
bmi = weight / (h_cm / 100) ** 2
X_bmi = np.column_stack([X, bmi])
print(f"add BMI (nonlinear) -> rank {np.linalg.matrix_rank(X_bmi)}: "
      f"'correlated' and 'linearly dependent' are NOT the same thing")
assert np.linalg.matrix_rank(X_bmi) == 4


# === SECTION 7 - THE LoRA IDEA: A LOW-RANK UPDATE ===
# LoRA's bet: the CHANGE you make when fine-tuning is well approximated by a low-rank
# matrix - so you train two skinny factors instead of the whole giant weight matrix.
d, k, rank_r = 512, 512, 8
W = rng.normal(0, 0.02, (d, k))                 # the frozen pretrained weights
B = rng.normal(0, 0.02, (d, rank_r))            # the two trainable slivers
A = rng.normal(0, 0.02, (rank_r, k))
delta = B @ A                                   # the low-rank update
print(f"\nfull update would train : {d * k:,} parameters")
print(f"LoRA trains             : {B.size + A.size:,} parameters "
      f"({100 * (B.size + A.size) / (d * k):.1f}%)")
print(f"rank(delta) = {np.linalg.matrix_rank(delta)}  (capped at r = {rank_r})")
assert np.linalg.matrix_rank(delta) == rank_r
assert B.size + A.size < 0.05 * d * k
print("\nLAB COMPLETE - independence, the null space, the fibers, rank-nullity,")
print("the ladder, low-rank real data, and the low-rank update behind LoRA.")
