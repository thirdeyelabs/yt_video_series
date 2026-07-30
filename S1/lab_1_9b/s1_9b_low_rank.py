"""
LAB 1.9b — Low-Rank Approximation: How SVD Compresses Everything
===============================================================
Companion notebook to "Low-Rank: How SVD Compresses Everything".

pip install numpy pillow        # pillow only for the image demo

You will verify every number shown in the episode: that a matrix is a SUM of rank-1 layers, that
truncating the sum gives A_k, that Eckart-Young holds EXACTLY (the error is the singular values you
threw away), that a real photograph is recognisable at ~9% of its storage, and that LoRA is the same
arithmetic applied to a weight matrix. Deterministic, runnable top-to-bottom.
"""

# === SECTION 1 - A MATRIX IS A SUM OF RANK-1 LAYERS ===
# 🎭 Analogy: a film summarised by its five key scenes. Each layer is one "scene" of the matrix,
# and the singular value in front of it says how loud that scene is.
import numpy as np

rng = np.random.default_rng(0)
A = rng.standard_normal((60, 40)) @ rng.standard_normal((40, 40))
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# rebuild A by adding one rank-1 layer at a time: sigma_i * u_i * v_i^T
rebuilt = np.zeros_like(A)
for i in range(len(s)):
    rebuilt += s[i] * np.outer(U[:, i], Vt[i])
assert np.allclose(rebuilt, A)                    # the sum of ALL layers is exactly A
print("A is exactly the sum of its rank-1 layers:", np.allclose(rebuilt, A))

layer1 = s[0] * np.outer(U[:, 0], Vt[0])
assert np.linalg.matrix_rank(layer1) == 1         # each layer really is rank 1
print(f"each layer is rank {np.linalg.matrix_rank(layer1)}; sigma1={s[0]:.1f} vs sigma2={s[1]:.1f}")


# === SECTION 2 - TRUNCATE: A_k, AND WHAT IT COSTS TO STORE ===
def rank_k(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k]


m, n = A.shape
print(f"\n{'k':>4}{'stored':>10}{'% of full':>11}{'rel error':>11}")
for k in (1, 2, 5, 10, 20, 40):
    Ak = rank_k(A, k)
    stored = k * (m + n + 1)                       # k columns + k rows + k singular values
    err = np.linalg.norm(A - Ak) / np.linalg.norm(A)
    print(f"{k:>4}{stored:>10,}{stored/(m*n)*100:>10.1f}%{err*100:>10.1f}%")
assert np.linalg.matrix_rank(rank_k(A, 5)) == 5


# === SECTION 3 - ECKART-YOUNG: A_k IS THE *BEST* RANK-k APPROXIMATION ===
# Two claims: (a) the error equals exactly the dropped singular values, and (b) no other rank-k
# matrix gets closer. We check (a) exactly and (b) empirically against random rank-k competitors.
k = 8
Ak = rank_k(A, k)
lhs = np.linalg.norm(A - Ak)                       # Frobenius norm of the error
rhs = np.sqrt((s[k:] ** 2).sum())                  # sqrt of the sum of dropped sigma^2
print(f"\n||A - A_k||      = {lhs:.6f}")
print(f"sqrt(sum s[k:]^2) = {rhs:.6f}")
assert np.isclose(lhs, rhs)                        # (a) holds EXACTLY

best_random = min(                                  # (b) 2000 random rank-k matrices, none closer
    np.linalg.norm(A - (rng.standard_normal((m, k)) @ rng.standard_normal((k, n))))
    for _ in range(2000))
print(f"best of 2000 random rank-{k} matrices: {best_random:.2f}  (vs A_k's {lhs:.2f})")
assert lhs < best_random
print("nothing of rank k got closer than A_k  ->  Eckart-Young")


# === SECTION 4 - A REAL PHOTOGRAPH (the episode's climax, reproduced) ===
# 🎭 Analogy: an equaliser — almost all the sound sits in a few bands; mute the rest and it is
# still the song. Image: NASA Blue Marble (Apollo 17), public domain.
try:
    from PIL import Image
    im = Image.open("assets/images/earth.jpg").convert("RGB").resize((440, 440))
    P = np.asarray(im, float) / 255.0
    mm, nn, _ = P.shape

    def rank_k_rgb(P, k):
        out = np.zeros_like(P)
        for c in range(3):
            U, s, Vt = np.linalg.svd(P[:, :, c], full_matrices=False)
            out[:, :, c] = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
        return np.clip(out, 0, 1)

    print(f"\n{'k':>5}{'storage':>10}{'rel error':>11}   what you see")
    seen = {1: "unrecognisable", 8: "recognisably Earth", 20: "clearly Earth",
            50: "looks fine", 120: "indistinguishable"}
    for k in (1, 3, 8, 20, 50, 120):
        Pk = rank_k_rgb(P, k)
        stored = 3 * k * (mm + nn + 1)
        pct = stored / (mm * nn * 3) * 100
        err = np.linalg.norm(P - Pk) / np.linalg.norm(P) * 100
        print(f"{k:>5}{pct:>9.1f}%{err:>10.1f}%   {seen.get(k,'')}")
        Image.fromarray((Pk * 255).astype("uint8")).save(
            f"assets/images/svd_rank/earth_k{k:03d}.png")
    # the headline claim from the video:
    P20 = rank_k_rgb(P, 20)
    pct20 = 3 * 20 * (mm + nn + 1) / (mm * nn * 3) * 100
    assert pct20 < 10.0                            # under 10% of the storage
    print(f"\nHEADLINE: k=20 keeps the picture at {pct20:.1f}% of the storage")
except (ImportError, FileNotFoundError) as e:
    print(f"\n[image demo skipped: {e}]")


# === SECTION 5 - THE SAME ARITHMETIC IS LoRA ===
# A weight matrix W is d x d. Fine-tuning normally retrains all d^2 of it. LoRA freezes W and
# learns a LOW-RANK correction B @ A instead: d x r and r x d. Same "keep only r layers" idea.
d = 4096
print(f"\nW is {d}x{d} = {d*d:,} weights")
print(f"{'r':>4}{'LoRA params (2dr)':>20}{'% of W':>10}")
for r in (1, 2, 4, 8, 16, 64):
    lora = 2 * d * r
    print(f"{r:>4}{lora:>20,}{lora/(d*d)*100:>9.2f}%")
r = 8
assert 2 * d * r / (d * d) < 0.005                 # under half a percent at r=8
print(f"\nat r={r}: {2*d*r:,} trainable vs {d*d:,} frozen = {2*d*r/(d*d)*100:.2f}%")

# and the shapes really do compose back to a d x d update:
B = rng.standard_normal((d // 64, r))              # scaled down so the demo runs instantly
Aa = rng.standard_normal((r, d // 64))
dW = B @ Aa
assert dW.shape == (d // 64, d // 64)
assert np.linalg.matrix_rank(dW) <= r              # the update is rank <= r, by construction
print(f"B @ A has shape {dW.shape} and rank {np.linalg.matrix_rank(dW)} (<= r = {r})")

print("\nLAB COMPLETE — a matrix is a sum of rank-1 layers; keeping the top k is provably the")
print("closest rank-k copy (Eckart-Young); that is image compression, and that is LoRA.")
