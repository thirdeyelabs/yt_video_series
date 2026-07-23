"""
LAB 1.7b — Change of Basis: the Same Vector, Different Coordinates
=================================================================
Companion notebook to "Change of Basis: the Same Vector, Different Coordinates".

pip install numpy        # that's it

You will verify every claim from the episode: a vector's COORDINATES depend on the basis
(the arrow never moves, the numbers do), the change-of-basis matrix B and its inverse, the
similarity transform B^-1 A B, and the punchline that sets up 1.8 - a matrix's OWN
eigenvectors are the basis in which it becomes diagonal (pure stretches, no rotation).
Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - THE SAME ARROW, TWO SETS OF NUMBERS ===
# 🎭 Analogy: one street corner, two addresses (GPS vs nearest-subway-stop). Neither
# label is the "real" place - the corner just IS. Coordinates only exist once you pick axes.
import numpy as np

v_std = np.array([4.0, 2.0])                  # the arrow, in the standard basis
B = np.array([[2.0, -1.0],                    # a new basis: its columns are the new axes
              [1.0,  1.0]])
v_new = np.linalg.inv(B) @ v_std              # the SAME arrow, read off the new axes
print(f"standard coords: {v_std}")
print(f"new-basis coords: {v_new}")           # -> [2, 0]
# the arrow is unchanged: rebuild it from the new coords and you get the same point
assert np.allclose(B @ v_new, v_std)


# === SECTION 2 - B GOES ONE WAY, B-INVERSE THE OTHER ===
# B: new coords -> standard coords.   B^-1: standard coords -> new coords.
assert np.allclose(B @ v_new, v_std)                  # new -> std
assert np.allclose(np.linalg.inv(B) @ v_std, v_new)   # std -> new
# a coordinate is just a recipe: v_new = [2, 0] means "2 of column-1, 0 of column-2"
assert np.allclose(2 * B[:, 0] + 0 * B[:, 1], v_std)
print("B and B^-1 are exact inverses:", np.allclose(B @ np.linalg.inv(B), np.eye(2)))


# === SECTION 3 - THE SIMILARITY TRANSFORM: TRANSLATE, ACT, TRANSLATE BACK ===
# To apply transform A "as seen in the new basis": B^-1 A B. It does the same geometric
# job as A, just expressed in the other coordinate system.
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
A_in_new = np.linalg.inv(B) @ A @ B
x_std = np.array([3.0, -1.0])
x_new = np.linalg.inv(B) @ x_std
# route 1: act in standard coords, then translate the answer to new coords
lhs = np.linalg.inv(B) @ (A @ x_std)
# route 2: translate first, then act with the new-basis version
rhs = A_in_new @ x_new
print(f"act-then-translate: {np.round(lhs, 6)}")
print(f"translate-then-act: {np.round(rhs, 6)}")
assert np.allclose(lhs, rhs)                  # same destination, either order


# === SECTION 4 - THE PUNCHLINE: EIGENVECTORS DIAGONALIZE (sets up 1.8) ===
# 🎭 Analogy: Celsius vs Fahrenheit - same warmth, different numbers; B/B^-1 is the
# conversion. The BEST basis for a matrix is its own eigenvectors: there, the messy warp
# becomes pure axis stretches - a diagonal matrix.
vals, V = np.linalg.eig(A)                     # V's columns are the eigenvectors
D = np.linalg.inv(V) @ A @ V                   # A, expressed in its own eigenbasis
print(f"eigenvalues: {np.round(vals, 3)}")
print(f"V^-1 A V =\n{np.round(D, 6)}")          # -> diag(3, 1): no rotation, just stretches
assert np.allclose(D, np.diag(vals))           # off-diagonal entries are exactly zero
# each eigenvector stays on its own line, only scaled by its eigenvalue (no rotation):
for i in range(2):
    assert np.allclose(A @ V[:, i], vals[i] * V[:, i])
print("every eigenvector: A v = lambda v  (stays on its line, only stretches)")


# === SECTION 5 - WHY AI CARES: PCA IS A LEARNED CHANGE OF BASIS ===
# PCA finds the basis (eigenvectors of the covariance) where the data's spread is
# axis-aligned - the coordinates in which the problem is simplest. Same machinery as above.
rng = np.random.default_rng(0)
cloud = rng.multivariate_normal([0, 0], [[3.0, 1.6], [1.6, 1.0]], size=500)
cov = np.cov(cloud.T)
pca_vals, pca_V = np.linalg.eigh(cov)          # the principal axes = an eigenbasis
cov_in_pca = pca_V.T @ cov @ pca_V             # covariance, viewed in the PCA basis
print(f"\ncovariance in PCA basis =\n{np.round(cov_in_pca, 4)}")   # diagonal: axes decorrelated
assert abs(cov_in_pca[0, 1]) < 1e-9            # off-diagonal ~ 0: the features are now independent
print("PCA = the change of basis that makes the data's covariance diagonal")

print("\nLAB COMPLETE - same arrow different numbers, B and B^-1, the B^-1 A B sandwich,")
print("eigenvectors as the diagonalizing basis, and PCA as a learned change of basis.")
