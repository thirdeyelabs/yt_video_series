"""
LAB 1.9 — SVD: Every Matrix Is Rotate, Stretch, Rotate
=====================================================
Companion notebook to "SVD: Every Matrix Is Rotate, Stretch, Rotate".

pip install numpy        # that's it

You will verify every claim from the episode: A = U S V-transpose for ANY matrix (square or
not), that U and V are orthonormal (pure rotations/reflections), that the singular values are the
lengths of the ellipse the unit circle maps to, and that they come out sorted big-to-small — the
ranking that sets up compression in 1.9b. Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - ONE CALL, THREE FACTORS: A = U S V^T ===
# 🎭 Analogy: any reshaping of a photo is turn, scale, turn. U turns, Sigma scales along axes,
# V-transpose turns first. np.linalg.svd hands you all three at once.
import numpy as np

A = np.array([[2.0, 1.0],                     # the same non-symmetric matrix from the episode
              [0.0, 1.5]])
U, s, Vt = np.linalg.svd(A)                    # s is a 1D array of singular values (sorted desc)
print(f"singular values: {np.round(s, 3)}")    # -> [2.379, 1.261]
Sigma = np.diag(s)
recon = U @ Sigma @ Vt
print(f"U @ diag(s) @ Vt =\n{np.round(recon, 6)}")
assert np.allclose(recon, A)                   # the three factors rebuild A exactly


# === SECTION 2 - U AND V ARE ROTATIONS (ORTHONORMAL) ===
# "Pure rotation" means the transpose is the inverse: U^T U = I. No stretching hidden in U or V —
# all the stretch lives in Sigma alone.
assert np.allclose(U.T @ U, np.eye(2))         # columns of U are orthonormal
assert np.allclose(Vt @ Vt.T, np.eye(2))       # rows of Vt (columns of V) are orthonormal
print("U orthonormal:", np.allclose(U.T @ U, np.eye(2)))
print("V orthonormal:", np.allclose(Vt.T @ Vt, np.eye(2)))
print(f"det(U) = {np.linalg.det(U):+.2f},  det(V) = {np.linalg.det(Vt.T):+.2f}")  # +1 = rotation


# === SECTION 3 - THE UNIT CIRCLE MAPS TO AN ELLIPSE WHOSE AXES ARE THE SIGMAS ===
# Map every point of the unit circle by A. The result is an ellipse; its semi-axis lengths are
# exactly the singular values, and its axis directions are the columns of U.
theta = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(theta), np.sin(theta)])   # 2 x 400 unit circle
ellipse = A @ circle
radii = np.linalg.norm(ellipse, axis=0)
print(f"ellipse longest reach: {radii.max():.3f}   (== sigma_1 = {s[0]:.3f})")
print(f"ellipse shortest reach: {radii.min():.3f}  (== sigma_2 = {s[1]:.3f})")
assert np.isclose(radii.max(), s[0], atol=1e-2)     # long axis  == biggest singular value
assert np.isclose(radii.min(), s[1], atol=1e-2)     # short axis == smallest singular value


# === SECTION 4 - EVERY MATRIX HAS ONE, EVEN NON-SQUARE ===
# Eigenvectors need a square matrix; SVD does not. A 2x3 matrix still factors cleanly — the
# block shapes just change (U is 2x2, Sigma is 2x3, Vt is 3x3).
B = np.array([[3.0, 1.0, 0.0],
              [0.0, 2.0, 2.0]])                # a 2x3 rectangle: no eigenvectors possible
Ub, sb, Vtb = np.linalg.svd(B)                 # full_matrices=True by default
print(f"B is {B.shape};  U is {Ub.shape},  s has {sb.shape[0]} values,  Vt is {Vtb.shape}")
Sig = np.zeros_like(B)                          # build the 2x3 Sigma block
Sig[:len(sb), :len(sb)] = np.diag(sb)
assert np.allclose(Ub @ Sig @ Vtb, B)          # the rectangle rebuilds exactly
print("non-square SVD reconstructs B:", np.allclose(Ub @ Sig @ Vtb, B))


# === SECTION 5 - THE SINGULAR VALUES ARE A RANKING (energy in the first few) ===
# 🎭 Analogy: a top-hits chart for directions — the first carries the most, the tail almost
# nothing. This ordering is what compression (1.9b) exploits: drop the small ones.
rng = np.random.default_rng(0)
M = rng.standard_normal((20, 20)) @ rng.standard_normal((20, 5)) @ rng.standard_normal((5, 20))
sm = np.linalg.svd(M, compute_uv=False)        # just the singular values
energy = sm**2
cum = np.cumsum(energy) / energy.sum()
print("singular values (desc):", np.round(sm[:8], 2), "...")
print(f"top 5 directions hold {cum[4]*100:.1f}% of the energy")
assert sm[0] >= sm[-1]                          # always sorted big -> small
assert cum[4] > 0.99                            # this matrix is really rank ~5: tail is ~nothing

print("\nLAB COMPLETE - A = U S V^T for any matrix (even non-square), U and V orthonormal,")
print("the unit circle maps to an ellipse with sigma-length axes, and the sigmas rank directions.")
