"""
LAB 1.5 — Dot Products, Cosine Similarity & a Tiny Attention Head
=================================================================
Companion notebook to "The Dot Product: How AI Measures Similarity".

pip install numpy        # that's it

You will build every idea from the episode: both dot-product recipes (proven equal),
the sign as alignment, projection (the shadow), cosine similarity, a working
similarity search, a neuron's weighted sum, and the episode's climax - a tiny
attention head that resolves "it" in a real sentence, with softmax weights.
Deterministic, runnable top-to-bottom, no API keys.
"""

# === SECTION 1 - TWO RECIPES, ONE NUMBER ===
# 🎭 Analogy: two people pulling a cart - same direction and the pulls add up.
# Recipe 1: multiply matching parts and add. Recipe 2: |a| |b| cos(theta).
# They ALWAYS agree - one is arithmetic, one is geometry.
import numpy as np

a = np.array([3.0, 1.0])
b = np.array([2.0, 2.0])

recipe1 = float(a @ b)                                        # (3)(2) + (1)(2) = 8
cos_theta = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
recipe2 = float(np.linalg.norm(a) * np.linalg.norm(b) * cos_theta)
print(f"recipe 1 (components): {recipe1:.4f}")
print(f"recipe 2 (|a||b|cos) : {recipe2:.4f}")
assert abs(recipe1 - recipe2) < 1e-9


# === SECTION 2 - THE SIGN IS THE STORY (aligned / orthogonal / opposed) ===
# Rotate one vector and watch only the sign: + agrees, 0 ignores, - opposes.
def rot(v, deg):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ v

for deg in (0, 45, 90, 135, 180):
    d = float(a @ rot(a, deg))
    tag = "aligned +" if d > 1e-9 else ("ORTHOGONAL 0" if abs(d) < 1e-9 else "opposed -")
    print(f"  angle {deg:>3d}°  a·b = {d:+7.3f}   {tag}")
assert abs(a @ rot(a, 90)) < 1e-9        # exactly zero at 90° - not small, ZERO


# === SECTION 3 - PROJECTION: THE SHADOW OF ONE VECTOR ON ANOTHER ===
# 🎭 Analogy: a stick over the ground at noon - the shadow is the part of the
# stick that lies ALONG the ground.  proj_b(a) = (a·b / |b|²) b
def project(a, b):
    return (a @ b) / (b @ b) * b

p = project(a, b)
perp = a - p                              # what's left is orthogonal to b...
print(f"shadow of a on b: {p}   leftover · b = {perp @ b:.1e}")
assert abs(perp @ b) < 1e-9               # ...provably


# === SECTION 4 - COSINE SIMILARITY: DIRECTION, NOT LENGTH ===
# A long document isn't automatically "more similar" - divide the lengths out.
def cos_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

short_doc = np.array([1.2, 0.9])
long_doc = short_doc * 2.7                # same direction, much longer
print(f"raw dot: {short_doc @ long_doc:.2f}   cosine: {cos_sim(short_doc, long_doc):.4f}")
assert abs(cos_sim(short_doc, long_doc) - 1.0) < 1e-9


# === SECTION 5 - A WORKING SIMILARITY SEARCH (the build_it, for real) ===
# Toy word vectors (hand-made, deterministic). Real systems learn these
# (episode 1.1) - the SEARCH mechanism is identical: best cosine wins.
vocab = {
    "cat":     np.array([0.9, 0.2, 0.8]),
    "dog":     np.array([0.8, 0.3, 0.7]),
    "milk":    np.array([0.1, 0.9, 0.2]),
    "tea":     np.array([0.2, 0.8, 0.1]),
    "thirsty": np.array([0.6, 0.3, 0.5]),
}
query = np.array([0.85, 0.25, 0.75])      # "something furry and animal-like"
scores = {w: cos_sim(query, v) for w, v in vocab.items()}
best = max(scores, key=scores.get)
for w, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"  {w:<8} cos = {s:.3f}")
print(f"best match: {best}")
assert best == "cat"


# === SECTION 6 - EVERY NEURON COMPUTES A DOT PRODUCT ===
# A neuron's pre-activation is w·x + bias: "how much does this input AGREE
# with my weights?" (the Σ wᵢxᵢ from episode 0.2, now as geometry).
w = np.array([0.5, 0.8, 0.3])
x = np.array([0.7, 0.2, 0.9])
pre_activation = w @ x
print(f"w·x = {pre_activation:.2f}")
assert abs(pre_activation - 0.78) < 1e-9


# === SECTION 7 - THE CLIMAX: A TINY ATTENTION HEAD RESOLVES "it" ===
# "The cat drank the milk because IT was thirsty."  Who is "it"?
# The query for "it" is dotted against every word's key; softmax turns the
# scores into weights. Toy embeddings, REAL mechanism (this is Q·K attention).
q_it = np.array([0.8, 0.1, 0.7])
keys = {
    "cat":     np.array([0.9, 0.2, 0.8]),
    "drank":   np.array([0.4, 0.5, 0.2]),
    "milk":    np.array([0.1, 0.9, 0.2]),
    "thirsty": np.array([0.6, 0.3, 0.5]),
}
raw = {w: float(q_it @ k) for w, k in keys.items()}

def softmax(d):
    e = {w: np.exp(v) for w, v in d.items()}
    z = sum(e.values())
    return {w: v / z for w, v in e.items()}

attn = softmax(raw)
for w in keys:
    bar = "#" * int(attn[w] * 40)
    print(f"  it -> {w:<8} q·k = {raw[w]:.2f}   attention = {attn[w]:.2f}  {bar}")
winner = max(attn, key=attn.get)
print(f'"it" attends to: {winner}')
assert winner == "cat"                    # the pronoun is resolved - by dot products

print("\nLAB COMPLETE - one tiny operation: similarity, search, neurons, attention.")
