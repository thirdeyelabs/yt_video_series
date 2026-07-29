"""
LAB — Attention, Part 1 of 5: a weighted average of values
==========================================================
Companion notebook to "Attention: How AI Knows What 'it' Means"
(part 1 of the series on *Attention Is All You Need*, Vaswani et al., NeurIPS 2017).

pip install numpy        # that's it

THE ONE SENTENCE: attention is a weighted average of values, and the weights are dot products.

This is FILE 1 OF 5. Each part of the series adds one piece, and by part 5 this folder is a
working transformer in pure numpy that generates text — no PyTorch:
    1. scaled dot-product attention   <-- you are here (unscaled for now; part 2 adds /sqrt(d_k))
    2. + multi-head
    3. + positional encoding
    4. + residual, LayerNorm, FFN  -> a full block
    5. + causal mask + generation loop  -> it writes

NOTE ON THE EXAMPLE: the sentence "The animal didn't cross the street because it was too tired"
is a well-known teaching example from the community's illustrated explanations, NOT from the
paper. The toy embeddings below are hand-designed so you can SEE why the answer comes out the way
it does — a real model learns these from data.
"""

# === SECTION 1 - THE SCORING RULE IS A DOT PRODUCT ===
# 🎭 Analogy: two arrows pointing the same way agree; at 90 degrees they don't; opposite is
# disagreement. One number for "how much do these two match?" (this is episode 1.5's tool).
import numpy as np

q = np.array([2.2, 1.1])
for name, k in [("same direction", np.array([1.9, 1.5])),
                ("perpendicular  ", np.array([-1.1, 2.2])),
                ("opposite       ", np.array([-2.2, -1.1]))]:
    print(f"  {name}  q . k = {q @ k:+.2f}")
assert q @ np.array([1.9, 1.5]) > 0            # agreement is positive
assert abs(q @ np.array([-1.1, 2.2])) < 1e-9   # perpendicular is exactly zero
assert q @ np.array([-2.2, -1.1]) < 0          # opposite is negative


# === SECTION 2 - THE TOY WORLD (hand-designed so the answer is inspectable) ===
# Six interpretable feature axes. A real model learns its own; these are readable on purpose.
FEATURES = ["animate", "road_like", "size", "tiredness", "width", "function_word"]
E = {
    "the":     [.05, .05, 0, .05, .05, 1],
    "animal":  [1, .05, .5, .1, .05, 0],    # strongly animate
    "didn't":  [.05, .05, 0, .05, .05, 1],
    "cross":   [.1, .4, 0, .05, .25, .3],   # road-ish (you cross a road)
    "street":  [.05, 1, .6, .05, .15, 0],   # strongly road-like
    "because": [.05, .05, 0, .05, .05, 1],
    "it":      [.3, .3, 0, 0, 0, .4],       # the ambiguous pronoun: between both candidates
    "was":     [.05, .05, 0, .05, .05, 1],
    "too":     [.05, .05, 0, .05, .05, 1],
    "tired":   [0, 0, 0, 1, 0, 0],          # tiredness is an ANIMATE property
    "wide":    [0, 0, 0, 0, 1, 0],          # width is a ROAD/object property
}
SENT = "the animal didn't cross the street because it was too".split()
IT = SENT.index("it")

# The learned projections. W_K: a word advertises WHICH property it can take.
# W_Q: the asker looks for whoever fits the property it is carrying.
W_K = np.zeros((6, 6)); W_K[0, 3] = 1.0; W_K[1, 4] = 1.0; W_K[2, 4] = 0.3
W_Q = np.zeros((6, 6)); W_Q[3, 3] = 1.0; W_Q[4, 4] = 1.0
W_V = np.eye(6)                                  # values = the words themselves, for readability
GAIN = 3.0                                       # real Q/K have bigger magnitudes than this toy


# === SECTION 3 - ATTENTION IN EIGHT LINES ===
# This is the whole mechanism. Everything after this in the series is a refinement of it.
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(X, Wq, Wk, Wv, gain=1.0):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    A = softmax(Q @ K.T * gain)        # every word vs every word  (part 2 adds / sqrt(d_k))
    return A @ V, A                    # the weighted average, and the weights


# === SECTION 4 - RESOLVING THE PRONOUN (the episode's payoff, reproduced) ===
# 🎭 Analogy: a noisy dinner party — every voice reaches you, you just weight one far more.
# NOTE: "it" carries the adjective because an EARLIER LAYER already blended it in. That is what
# makes coreference work in a real transformer: layer 1 mixes context, layer 2 resolves the
# reference. We simulate that one mixing step here so a single layer can show the effect.
def attend_from_it(adjective):
    toks = SENT + [adjective]
    X = np.array([E[t] for t in toks], float)
    X = X.copy()
    X[IT] = X[IT] + X[toks.index(adjective)]      # <- the "previous layer" mixing step
    _, A = attention(X, W_Q, W_K, W_V, gain=GAIN)
    w = A[IT].copy()
    w[IT] = 0.0                                    # ignore self-attention for this demo
    w /= w.sum()
    return toks, w


for adj in ("tired", "wide"):
    toks, w = attend_from_it(adj)
    top = int(np.argmax(w))
    print(f"\n'...too {adj}'  ->  'it' attends to '{toks[top]}' ({w[top]:.3f})")
    for t, x in sorted(zip(toks, w), key=lambda kv: -kv[1])[:4]:
        print(f"    {t:9} {x:.3f}  " + "#" * int(x * 50))

# the claim from the video, asserted:
toks_t, w_t = attend_from_it("tired")
toks_w, w_w = attend_from_it("wide")
A_i, S_i = SENT.index("animal"), SENT.index("street")
assert toks_t[int(np.argmax(w_t))] == "animal"     # tired -> the ANIMAL was tired
assert toks_w[int(np.argmax(w_w))] == "street"     # wide  -> the STREET was wide
assert w_t[A_i] > 0.6 and w_w[S_i] > 0.6           # and decisively so
print(f"\nFLIP CONFIRMED: animal {w_t[A_i]:.3f} -> {w_w[A_i]:.3f}, "
      f"street {w_t[S_i]:.3f} -> {w_w[S_i]:.3f}")


# === SECTION 5 - IT REALLY IS A WEIGHTED AVERAGE (the one sentence, proven) ===
# The output for "it" is literally sum_j (weight_j * value_j). Two things must hold:
# the weights sum to 1, and the output lands inside the cloud of value vectors.
toks, w = attend_from_it("tired")
X = np.array([E[t] for t in toks], float)
V = X @ W_V
out = w @ V                                        # the weighted average

assert np.isclose(w.sum(), 1.0)                    # softmax guarantees this
manual = sum(w[j] * V[j] for j in range(len(toks)))
assert np.allclose(out, manual)                    # it is exactly the weighted sum

print(f"\noutput for 'it'      = {np.round(out, 3)}")
print(f"the 'animal' vector  = {np.round(V[A_i], 3)}")
print(f"cosine(out, animal)  = {out @ V[A_i] / (np.linalg.norm(out) * np.linalg.norm(V[A_i])):.4f}")
# the new "it" should sit much closer to animal than to street:
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
assert cos(out, V[A_i]) > cos(out, V[S_i])
print("the new 'it' vector is closer to ANIMAL than to STREET  ->  the pronoun is resolved")


# === SECTION 6 - THE FULL ATTENTION MATRIX (every word vs every word) ===
# Run it for all words at once. A is n x n: row i is "who does word i pay attention to?"
X_all = np.array([E[t] for t in SENT + ["tired"]], float)
X_all[IT] = X_all[IT] + X_all[-1]
out_all, A = attention(X_all, W_Q, W_K, W_V, gain=GAIN)
print(f"\nattention matrix shape: {A.shape}   (rows sum to 1: {np.allclose(A.sum(1), 1)})")
assert A.shape == (len(SENT) + 1, len(SENT) + 1)
assert np.allclose(A.sum(axis=1), 1.0)             # EVERY row is a probability distribution
print("row for 'it':", " ".join(f"{t}={v:.2f}" for t, v in zip(SENT + ["tired"], A[IT]) if v > 0.03))


# === SECTION 7 - THE SAME FORMULA ON OTHER SENTENCES (nothing new, just reuse) ===
# 🎭 Analogy: one tool, many jobs. Change the sentence, change nothing else, and the same
# equation resolves a different pronoun. NOTE the design: NOUN axes say what a thing IS, QUERY
# axes are what an adjective ASKS FOR, and W_K maps noun-property -> query-axis (CROSS-axis).
# That matters: with an identity mapping the adjective advertises its own property, so "sweet"
# scores as high as "flower" and the demo points at the wrong word.
NOUN = ["animate", "road", "size", "plant", "container", "massive", "fn"]
QRY = ["q_tired", "q_wide", "q_sweet", "q_hollow", "q_heavy"]
AX = NOUN + QRY
N2 = len(AX)


def vec(**kw):
    a = [0.0] * N2
    for k, val in kw.items():
        a[AX.index(k)] = val
    for i, nm in enumerate(AX):                      # dense, like real embeddings
        if a[i] == 0 and nm in ("animate", "road", "size", "plant", "container", "massive"):
            a[i] = .05
    return a


E2 = {"the": vec(fn=1), "i": vec(animate=1, fn=.2), "on": vec(fn=1), "was": vec(fn=1),
      "too": vec(fn=1), "because": vec(fn=1), "didn't": vec(fn=1),
      "animal": vec(animate=1, size=.5), "street": vec(road=1, size=.6),
      "cross": vec(animate=.1, road=.4, fn=.3),
      "bee": vec(animate=1, size=.1), "flower": vec(plant=1, size=.2),
      "landed": vec(animate=.1, fn=.3),
      "book": vec(massive=1, size=.3), "shelf": vec(container=1, size=.5),
      "put": vec(animate=.1, fn=.3),
      "it": vec(animate=.3, road=.3, fn=.4),
      "tired": vec(q_tired=1), "wide": vec(q_wide=1), "sweet": vec(q_sweet=1),
      "empty": vec(q_hollow=1), "heavy": vec(q_heavy=1)}
WK2 = np.zeros((N2, N2))
for src, dst in [("animate", "q_tired"), ("road", "q_wide"), ("plant", "q_sweet"),
                 ("container", "q_hollow"), ("massive", "q_heavy")]:
    WK2[AX.index(src), AX.index(dst)] = 1.0
WK2[AX.index("size"), AX.index("q_wide")] = 0.3
WQ2 = np.zeros((N2, N2))
for qa in QRY:
    WQ2[AX.index(qa), AX.index(qa)] = 1.0


def attend(sentence, adjective, pronoun="it"):
    toks = sentence.split() + [adjective]
    X = np.array([E2[t] for t in toks], float)
    i = toks.index(pronoun)
    q = (X[i] + X[-1]) @ WQ2                          # the earlier layer's mixing step again
    sc = (X @ WK2) @ q * GAIN
    sc[i] = -1e9
    w = np.exp(sc - sc.max()); w /= w.sum()
    return toks, w


CASES = [("the animal didn't cross the street because it was too", "tired", "animal"),
         ("the animal didn't cross the street because it was too", "wide", "street"),
         ("the bee landed on the flower because it was", "sweet", "flower"),
         ("the bee landed on the flower because it was", "tired", "bee"),
         ("i put the book on the shelf because it was", "empty", "shelf"),
         ("i put the book on the shelf because it was", "heavy", "book")]
print("\nsame formula, six sentences:")
for sent, adj, expected in CASES:
    toks, w = attend(sent, adj)
    j = int(np.argmax(w))
    print(f"  ...too {adj:6} -> {toks[j]:7} ({w[j]:.3f})")
    assert toks[j] == expected, f"{adj} should resolve to {expected}"
    # and only ONE candidate should be strong enough to draw (no ambiguous second arrow)
    assert sum(1 for x in w if x >= 0.15) == 1

print("\nLAB COMPLETE — attention is a weighted average of values, and the weights are dot products.")
print("NEXT (part 2): run this at real size and the softmax collapses to a single spike.")
print("The fix is one division — / sqrt(d_k) — and then we run eight of these at once.")
