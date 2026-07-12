"""
LAB 1.4 — The Determinant: Measure It, Break It, Meet It in ML
==============================================================
Companion notebook to "The Determinant: a Zoom Factor for Space".

pip install numpy        # that's it (matplotlib optional for the plot cell)

You will verify every claim from the episode with your own hands: the determinant IS
a measured area (not a formula), det = 0 really destroys information (two different
points collapse onto one), the sign is orientation, det(AB) = det(A)·det(B),
|det| = the product of the singular values, and the two places it lives in real ML -
the covariance "redundancy alarm" and the Gaussian log-likelihood's log-det.
Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - THE DETERMINANT IS A MEASURED AREA ===
# 🎭 Analogy: a cookie cutter pressed into rolled dough - the determinant is how
# much bigger the stamped shape's area becomes. Verify ad - bc by MEASURING:
# transform the unit square's corners and compute the parallelogram's area directly.
import numpy as np

A = np.array([[3.0, 1.0],
              [1.0, 2.0]])

formula = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]          # ad - bc = 5
i_hat, j_hat = A @ np.array([1.0, 0.0]), A @ np.array([0.0, 1.0])
measured_area = abs(np.cross(i_hat, j_hat))               # |i' x j'| = parallelogram area
print(f"ad - bc            = {formula:.4f}")
print(f"measured area      = {measured_area:.4f}")
print(f"np.linalg.det(A)   = {np.linalg.det(A):.4f}")
assert abs(formula - measured_area) < 1e-9                # the formula IS the area


# === SECTION 2 - det = 0 DESTROYS INFORMATION (no undo) ===
# 🎭 Analogy: flatten a box into a pancake - you can never rebuild which crumb
# came from where. Two DIFFERENT points map to the SAME point: unrecoverable.
S = np.array([[1.0, 2.0],
              [0.5, 1.0]])                                # det = 1 - 1 = 0 (columns aligned)
print(f"det(S) = {np.linalg.det(S):.4f}")

p1, p2 = np.array([2.0, 0.0]), np.array([0.0, 1.0])       # clearly different inputs
out1, out2 = S @ p1, S @ p2
print(f"S @ {p1} = {out1}   S @ {p2} = {out2}")
assert not np.allclose(p1, p2) and np.allclose(out1, out2)  # different in, SAME out
try:
    np.linalg.inv(S)
    raise RuntimeError("should never get here")
except np.linalg.LinAlgError as e:
    print(f"inverting S -> LinAlgError: {e}  (no undo button)")


# === SECTION 3 - THE SIGN IS ORIENTATION (a mirror flip) ===
# A reflection has det = -1: same areas, but space is flipped like a stamp
# pressed face-down. The sign of the cross product tracks the flip.
F = np.array([[-1.0, 0.0],
              [0.0, 1.0]])                                # mirror across the y-axis
print(f"det(mirror) = {np.linalg.det(F):.1f}")
orient_before = np.sign(np.cross([1, 0], [0, 1]))         # +1: counter-clockwise
orient_after = np.sign(np.cross(F @ [1, 0], F @ [0, 1]))  # -1: clockwise (flipped)
print(f"orientation before/after: {orient_before:+.0f} / {orient_after:+.0f}")
assert orient_before == 1 and orient_after == -1


# === SECTION 4 - COMPOSITION MULTIPLIES: det(AB) = det(A) · det(B) ===
# Stretch by 3, then by 2 -> space ends up 6x bigger (ties back to episode 1.3).
B = np.array([[2.0, 0.0],
              [0.4, 1.0]])                                # det = 2
lhs = np.linalg.det(A @ B)
rhs = np.linalg.det(A) * np.linalg.det(B)
print(f"det(AB) = {lhs:.4f}   det(A)·det(B) = {rhs:.4f}")
assert abs(lhs - rhs) < 1e-9


# === SECTION 5 - |det| = THE PRODUCT OF THE STRETCH FACTORS (SVD preview) ===
# Every matrix stretches space along two special directions by sigma_1, sigma_2
# (the singular values - episode 1.9). Their product is the area scaling: |det|.
M = np.array([[1.8, 0.6],
              [0.6, 1.2]])
sigmas = np.linalg.svd(M, compute_uv=False)
print(f"singular values = {sigmas.round(3)}   product = {np.prod(sigmas):.4f}"
      f"   |det| = {abs(np.linalg.det(M)):.4f}")
assert abs(np.prod(sigmas) - abs(np.linalg.det(M))) < 1e-9


# === SECTION 6 - ML #1: THE COVARIANCE DETERMINANT IS A REDUNDANCY ALARM ===
# Healthy features -> a fat data ellipse -> healthy det(Cov). Add a REDUNDANT
# feature (a copy in disguise: height in cm next to height in inches) and the
# cloud flattens onto a line: det(Cov) collapses toward zero. Models choke here.
rng = np.random.default_rng(7)
n = 500
height_cm = rng.normal(170, 8, n)
weight_kg = rng.normal(70, 6, n) + 0.4 * (height_cm - 170)   # genuinely new info
healthy = np.cov(np.stack([height_cm, weight_kg]))
height_in = height_cm / 2.54 + rng.normal(0, 0.01, n)        # SAME info, new costume
redundant = np.cov(np.stack([height_cm, height_in]))
print(f"det(Cov) healthy   = {np.linalg.det(healthy):10.4f}")
print(f"det(Cov) redundant = {np.linalg.det(redundant):10.4f}   <- the alarm rings")
assert np.linalg.det(redundant) < 0.01 * np.linalg.det(healthy)


# === SECTION 7 - ML #2: THE log-det INSIDE THE GAUSSIAN LOG-LIKELIHOOD ===
# The multivariate Gaussian's log-density carries a -1/2 log det(Sigma) term -
# the "volume correction" for how much the covariance stretches space. Build the
# log-likelihood from scratch and verify the log-det term does real work.
def gaussian_loglik(x, mu, Sigma):
    d = len(mu)
    diff = x - mu
    return float(-0.5 * (diff @ np.linalg.inv(Sigma) @ diff
                         + np.log(np.linalg.det(Sigma))          # <- the star of 1.4
                         + d * np.log(2 * np.pi)))

mu = np.zeros(2)
tight = np.eye(2) * 0.5                                   # small det: concentrated
wide = np.eye(2) * 4.0                                    # big det: spread out
x_at_center = np.zeros(2)
ll_tight = gaussian_loglik(x_at_center, mu, tight)
ll_wide = gaussian_loglik(x_at_center, mu, wide)
print(f"log-lik at center: tight = {ll_tight:.3f}   wide = {ll_wide:.3f}")
# same point, same distance (zero) - ONLY the log-det differs. Tight wins at the
# center because it wastes no probability elsewhere. That trade-off IS the log-det.
assert ll_tight > ll_wide
delta = ll_wide - ll_tight
expected = -0.5 * (np.log(np.linalg.det(wide)) - np.log(np.linalg.det(tight)))
assert abs(delta - expected) < 1e-9                       # the gap is EXACTLY the log-det term

print("\nLAB COMPLETE - the determinant: measured, broken (det=0), flipped, composed,")
print("factored into stretches, and found inside real ML (covariance + Gaussian log-lik).")
