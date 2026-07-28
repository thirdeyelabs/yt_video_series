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


# === SECTION 6 - INSIDE AI #1: RECOMMENDERS (20 ratings are really 2 hidden tastes) ===
# 🎭 Analogy: nobody labelled "action" and "romance" - the factorization discovers them.
# A ratings table with structure is LOW RANK: SVD finds the few hidden factors behind it.
users = ["Ana", "Ben", "Cara", "Dev"]
taste = np.array([[1.0, 0.2],      # how much each viewer likes [action, romance]
                  [0.9, 0.3],
                  [0.2, 1.0],
                  [0.3, 0.9]])
film = np.array([[5.0, 4.6, 1.2, 1.0, 4.8],    # how much each film HAS of [action, romance]
                 [1.0, 1.4, 4.9, 4.7, 1.2]])
R = taste @ film                                # the full 4x5 ratings table
print("ratings table:\n", np.round(R, 1))
sr = np.linalg.svd(R, compute_uv=False)
print("singular values:", np.round(sr, 2))      # -> [16.04, 5.64, 0, 0]: only TWO matter
assert np.linalg.matrix_rank(R) == 2            # 20 numbers, 2 real degrees of freedom
# reconstruct from the top 2 singular directions alone -> exact
Ur, sr2, Vtr = np.linalg.svd(R)
R2 = Ur[:, :2] @ np.diag(sr2[:2]) @ Vtr[:2, :]
assert np.allclose(R2, R)                       # rank-2 rebuild is perfect
print("rank-2 reconstruction is exact:", np.allclose(R2, R))
# NOTE (honest caveat): real recommenders can't run plain SVD - the table has MISSING entries.
# The Netflix-prize systems used the same factorization IDEA fit by gradient descent on the
# observed cells only ("FunkSVD"), then predicted the gaps from the learned factors.


# === SECTION 7 - INSIDE AI #2: MEANING FROM TEXT (truncated SVD = LSA) ===
# Keep the top-k singular directions of a word x document count matrix and synonyms collapse
# together — the direct ancestor of modern word embeddings.
vocab = ["car", "automobile", "banana", "mango"]
#            doc1 doc2 doc3 doc4  (docs 1-2 are about vehicles, 3-4 about fruit)
counts = np.array([[4, 6, 0, 1],     # car          (real counts are lopsided, not symmetric)
                   [5, 3, 1, 0],     # automobile
                   [0, 1, 5, 4],     # banana
                   [1, 0, 3, 6]], float)
Uw, sw, Vtw = np.linalg.svd(counts)
emb = Uw[:, :2] * sw[:2]             # each word's coordinates in the top-2 "concept" space


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


sim_syn = cos(emb[0], emb[1])        # car vs automobile
sim_unrel = cos(emb[0], emb[2])      # car vs banana
print(f"cos(car, automobile) = {sim_syn:+.6f}")   # ~0.99999: they land almost exactly together
print(f"cos(car, banana)     = {sim_unrel:+.6f}")
assert sim_syn > 0.9                 # synonyms land almost on top of each other
assert sim_syn > sim_unrel           # and far from the unrelated word
print("truncated SVD learned that car ~ automobile, with no dictionary")


# === SECTION 8 - INSIDE AI #3: THE SPECTRAL NORM DECIDES IF TRAINING SURVIVES ===
# sigma_1 is the MOST a layer can amplify any input (the spectral norm / Lipschitz bound).
# Stack layers and it compounds: >1 explodes, <1 vanishes. Spectral normalization divides
# W by sigma_1 to force it to exactly 1 — and finds sigma_1 by POWER ITERATION (episode 1.8).
rng = np.random.default_rng(0)
W = rng.standard_normal((64, 64))
sigma1 = np.linalg.svd(W, compute_uv=False)[0]
# sigma_1 really is the worst-case amplification over all inputs:
worst = max(np.linalg.norm(W @ (v := rng.standard_normal(64)) ) / np.linalg.norm(v)
            for _ in range(2000))
print(f"sigma_1 = {sigma1:.3f},  worst amplification sampled = {worst:.3f}")
assert worst <= sigma1 + 1e-9        # nothing can be amplified more than sigma_1

for s in (1.4, 0.7):                 # what compounding does over depth
    print(f"  sigma_1={s}: after 3 layers x{s**3:.2f},  after 100 layers x{s**100:.2e}")

# power iteration finds sigma_1 without a full SVD (what spectral norm does every step)
v = rng.standard_normal(64)
for _ in range(100):
    v = W.T @ (W @ v)
    v /= np.linalg.norm(v)
sigma1_power = np.linalg.norm(W @ v)
print(f"power iteration sigma_1 = {sigma1_power:.3f}  (exact {sigma1:.3f})")
assert np.isclose(sigma1_power, sigma1, rtol=1e-4)
W_sn = W / sigma1                     # spectral normalization
assert np.isclose(np.linalg.svd(W_sn, compute_uv=False)[0], 1.0)
print("after spectral normalization, sigma_1 = 1.0 exactly")

print("\nLAB COMPLETE - A = U S V^T for any matrix (even non-square), U and V orthonormal,")
print("the unit circle maps to an ellipse with sigma-length axes, the sigmas rank directions,")
print("and SVD is running inside recommenders, word embeddings, and training stability.")
