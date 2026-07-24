"""
LAB 1.8 — Eigenvectors: the Axes That Don't Rotate
==================================================
Companion notebook to "Eigenvectors: the Axes That Don't Rotate".

pip install numpy        # that's it

You will verify every claim from the episode: a matrix only STRETCHES its eigenvectors
(A v = lambda v, no rotation), those eigenvectors are the basis that diagonalizes the matrix
(the eigenbasis from 1.7b), and the AI payoff — PageRank as the dominant eigenvector of a link
matrix, found from scratch by power iteration. Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - A MATRIX ONLY STRETCHES ITS EIGENVECTORS ===
# 🎭 Analogy: the grain of wood — a few natural directions the matrix stretches cleanly along,
# everything else it twists. A v = lambda v means "acting with A = just scaling", for those dirs.
import numpy as np

A = np.array([[2.0, 1.0],                     # the same matrix from 1.7b (eigenvalues 3 and 1)
              [1.0, 2.0]])
vals, V = np.linalg.eig(A)                     # V's columns are the eigenvectors (unit length)
print(f"eigenvalues: {np.round(vals, 3)}")     # -> [3, 1]
for i in range(2):
    v = V[:, i]
    Av = A @ v
    print(f"  A v = {np.round(Av, 3)},   lambda v = {np.round(vals[i] * v, 3)}")
    assert np.allclose(A @ v, vals[i] * v)     # same direction, only scaled — never rotated


# === SECTION 2 - A NON-EIGENVECTOR DOES ROTATE ===
# Contrast: pick an ordinary direction and watch it turn (its output is NOT a scalar multiple).
def angle(u):
    return np.degrees(np.arctan2(u[1], u[0]))

v_gen = np.array([np.cos(np.radians(100)), np.sin(np.radians(100))])   # an ordinary direction
out = A @ v_gen
print(f"ordinary vector: {angle(v_gen):.1f} deg  ->  {angle(out):.1f} deg   (it ROTATED)")
assert abs(angle(v_gen) - angle(out)) > 1.0                # direction changed
v_eig = V[:, 0]
print(f"eigenvector:     {angle(v_eig):.1f} deg  ->  {angle(A @ v_eig):.1f} deg   (unchanged)")
assert np.isclose(angle(v_eig) % 180, angle(A @ v_eig) % 180)   # stayed on its own line


# === SECTION 3 - EIGENVECTORS ARE THE BASIS THAT DIAGONALIZES (callback to 1.7b) ===
# Stack the eigenvectors as columns -> that's the change-of-basis B. Then B^-1 A B is diagonal:
# in the eigenbasis the messy matrix becomes pure stretch factors, no rotation.
D = np.linalg.inv(V) @ A @ V
print(f"V^-1 A V =\n{np.round(D, 6)}")          # -> diag(3, 1)
assert np.allclose(D, np.diag(vals))            # off-diagonal is exactly zero
assert np.allclose(np.diag(D), vals)            # the diagonal IS the eigenvalues


# === SECTION 4 - PAGERANK FROM SCRATCH: POWER ITERATION ===
# 🎭 Analogy: shake a tray of marbles and the biggest one settles to the bottom — repeated
# multiplication drags any start vector toward the DOMINANT eigenvector. That is PageRank.
# Link graph (column-stochastic): A->C,B  B->C  C->A,B,D,E  D->C,A  E->C,A   (C = the authority)
names = ["A", "B", "C", "D", "E"]
M = np.array([
    [0.0, 0.0, 0.25, 0.5, 0.5],   # -> A
    [0.5, 0.0, 0.25, 0.0, 0.0],   # -> B
    [0.5, 1.0, 0.0,  0.5, 0.5],   # -> C
    [0.0, 0.0, 0.25, 0.0, 0.0],   # -> D
    [0.0, 0.0, 0.25, 0.0, 0.0],   # -> E
])
assert np.allclose(M.sum(axis=0), 1.0)          # every column sums to 1 (a probability of clicks)

d = 0.85                                         # damping (the 15% random-jump Google added)
n = len(names)
G = d * M + (1 - d) / n * np.ones((n, n))        # the Google matrix
r = np.ones(n) / n                               # start: every page equally important
for step in range(60):
    r = G @ r
    r = r / r.sum()                              # power iteration + renormalize
ranking = sorted(zip(names, r), key=lambda kv: -kv[1])
print("PageRank (power iteration):")
for name, score in ranking:
    print(f"  {name}: {score:.3f}   " + "#" * int(score * 60))
assert ranking[0][0] == "C"                      # the authority wins


# === SECTION 5 - VERIFY: PAGERANK IS THE DOMINANT EIGENVECTOR (lambda = 1) ===
# The vector power iteration converged to is exactly G's eigenvector for eigenvalue 1.
w, U = np.linalg.eig(G)
k = int(np.argmax(w.real))                       # index of the largest eigenvalue
print(f"\ndominant eigenvalue: {w[k].real:.4f}")   # -> 1.0 (a stochastic matrix's top eigenvalue)
assert np.isclose(w[k].real, 1.0)
dom = np.abs(U[:, k].real)
dom = dom / dom.sum()                             # normalize to a probability vector
print("eigenvector match:", np.allclose(dom, r, atol=1e-3))
assert np.allclose(dom, r, atol=1e-3)            # power iteration == the eigenvector, exactly

print("\nLAB COMPLETE - A v = lambda v (stretch, no rotation), eigenvectors diagonalize A,")
print("and PageRank is the dominant eigenvector of the link matrix, found by power iteration.")
