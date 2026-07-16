"""
LAB 1.6 — Linear Systems: Solve It, Break It, Fit It
====================================================
Companion notebook to "The Equation Hiding Inside Every Trained Model".

pip install numpy        # that's it (matplotlib optional for the plot cell)

You will verify every claim from the episode: the solution is the point where every
constraint holds, elimination never moves the answer, the three endings (one / none /
infinite) and how the determinant predicts them, the COLUMN picture (x, y are a recipe),
and the climax - least squares via the normal equation, checked against np.linalg.lstsq,
plus the shadow: the residual is orthogonal to everything A can reach.
Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - THE SOLUTION IS WHERE EVERY CONSTRAINT HOLDS ===
# 🎭 Analogy: GPS - each satellite's distance is a constraint (a circle); you are
# the one point satisfying all of them. Here: two rules, one crossing.
#     x + 2y = 8          3x - y = 3
import numpy as np

A = np.array([[1.0, 2.0],
              [3.0, -1.0]])
b = np.array([8.0, 3.0])

sol = np.linalg.solve(A, b)
print(f"solution: x = {sol[0]:g}, y = {sol[1]:g}")
assert np.allclose(A @ sol, b)                       # BOTH rules hold at (2, 3)


# === SECTION 2 - ELIMINATION NEVER MOVES THE ANSWER (the bet) ===
# Row operations completely REPLACE the rules - and the crossing point stays put.
M = np.column_stack([A, b])                          # the augmented matrix [A | b]
M[1] = M[1] - 3 * M[0]                               # R2 -> R2 - 3 R1   (cancel x)
M[0] = M[0] + (2 / 7) * M[1]                         # R1 -> R1 + 2/7 R2 (cancel y)
M[1] = M[1] / M[1, 1]                                # normalize
print("after elimination:\n", M.round(10))
assert np.allclose(M[:, 2], [2.0, 3.0])              # the rules now just SAY the answer
assert np.allclose(np.linalg.solve(M[:, :2], M[:, 2]), sol)   # same point. You won the bet.


# === SECTION 3 - THREE ENDINGS, PREDICTED BY THE DETERMINANT (1.4 pays off) ===
cases = {
    "one solution ": np.array([[1.0, 2.0], [3.0, -1.0]]),   # crossing lines
    "none (parallel)": np.array([[1.0, 2.0], [1.0, 2.0]]),  # same left side, b differs
    "infinite (same)": np.array([[1.0, 2.0], [2.0, 4.0]]),  # a copy in disguise
}
for name, C_ in cases.items():
    d = np.linalg.det(C_)
    print(f"  {name}  det = {d:+.1f}   {'unique crossing' if abs(d) > 1e-12 else 'NO unique answer'}")
assert abs(np.linalg.det(cases["none (parallel)"])) < 1e-12
assert abs(np.linalg.det(cases["infinite (same)"])) < 1e-12


# === SECTION 4 - THE COLUMN PICTURE: x, y ARE A RECIPE ===
# Solving Ax = b asks: how much of EACH COLUMN do you mix to reach the target b?
c1, c2 = A[:, 0], A[:, 1]
mix = sol[0] * c1 + sol[1] * c2                      # 2 parts col-1 + 3 parts col-2
print(f"2*{c1} + 3*{c2} = {mix}  ->  reaches b = {b}")
assert np.allclose(mix, b)
# and WHY the singular case fails: collinear columns only reach one line
S = cases["infinite (same)"]
s1, s2 = S[:, 0], S[:, 1]
cross = s1[0] * s2[1] - s1[1] * s2[0]                # 2D cross product = 0 -> collinear
print(f"singular columns collinear? cross = {cross:g}  (the reachable world is one line)")
assert abs(cross) < 1e-12


# === SECTION 5 - THE CLIMAX: LEAST SQUARES (training IS this) ===
# 20 noisy points, 2 knobs: overdetermined - NO exact solution. The normal equation
# A^T A x = A^T b turns 20 impossible rules into 2 solvable ones.
rng = np.random.default_rng(11)
xs = rng.uniform(0.5, 8.5, 20)
ys = 0.7 * xs + 1.0 + rng.normal(0, 0.55, 20)
X = np.stack([xs, np.ones_like(xs)], axis=1)         # the design matrix

normal_sol = np.linalg.solve(X.T @ X, X.T @ ys)      # the normal equation, by hand
lstsq_sol = np.linalg.lstsq(X, ys, rcond=None)[0]    # numpy's answer
print(f"normal equation: slope = {normal_sol[0]:.4f}, intercept = {normal_sol[1]:.4f}")
print(f"np.linalg.lstsq: slope = {lstsq_sol[0]:.4f}, intercept = {lstsq_sol[1]:.4f}")
assert np.allclose(normal_sol, lstsq_sol)

# no other line beats it: nudge the knobs, the squared error only goes UP
def sse(w):
    return float(np.sum((ys - X @ w) ** 2))
best = sse(normal_sol)
for nudge in ([0.05, 0], [-0.05, 0], [0, 0.1], [0, -0.1]):
    assert sse(normal_sol + nudge) > best
print(f"minimum squared error = {best:.2f}  (every nudge makes it worse)")


# === SECTION 6 - THE SHADOW (the 1.5 callback, proven) ===
# The best fit X @ x_hat is b's PROJECTION onto everything X can reach - so the
# leftover error must be ORTHOGONAL to every column (dot product exactly zero).
residual = ys - X @ normal_sol
print(f"residual · column1 = {residual @ X[:, 0]:+.2e}")
print(f"residual · column2 = {residual @ X[:, 1]:+.2e}")
assert abs(residual @ X[:, 0]) < 1e-8 and abs(residual @ X[:, 1]) < 1e-8
print("\nLAB COMPLETE - solved, broken two ways, mixed as a recipe, fitted by the")
print("normal equation, and the fit proven to be the shadow (residual ⟂ columns).")
