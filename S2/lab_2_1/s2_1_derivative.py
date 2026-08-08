"""
LAB 2.1 - The Derivative, from scratch.

Folded into episode 2.1 (no standalone upload - CLAUDE.md golden rule 0c) and shipped as a
downloadable notebook. Everything here is what the episode showed, reproducible on any laptop.

    pip install numpy torch
    python labs/s2_1_derivative/s2_1_derivative.py

Deterministic: fixed seed, no downloads, runs in under a second.
"""
import numpy as np

# === SECTION 1 - THE SMALLEST HONEST MODEL ===
# One weight. Sixty-four real points. A score that says how wrong the weight is.
# Real-life analogy: a shower tap you cannot read - nudge it, feel what happens.
rng = np.random.default_rng(0)
x = rng.normal(1.0, 0.4, 64)
y = 3.0 * x + rng.normal(0, 0.25, 64)          # the data was really made with w = 3.0

def loss(w):
    """Mean squared error: square every miss so both directions count, then average."""
    return float(np.mean((w * x - y) ** 2))

w0 = 2.0
print(f"[1] score at w = {w0}: {loss(w0):.6f}")

# === SECTION 2 - NUDGE IT AND DIVIDE ===
# The whole idea: change the knob a little, see how much the score moved, divide.
h = 0.1
rise = loss(w0 + h) - loss(w0)
print(f"[2] nudge {h}  ->  score moves {rise:+.6f}  ->  ratio {rise / h:+.6f}")

# === SECTION 3 - SHRINK THE NUDGE (watch a limit happen) ===
# The digits stop changing one at a time. That settling number IS the derivative.
print("[3] shrinking the nudge:")
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    print(f"      h = {h:<8.0e}  ratio = {(loss(w0 + h) - loss(w0)) / h:+.8f}")

# === SECTION 4 - VALIDATE AGAINST AUTOGRAD ===
# A number that appears out of a shrinking nudge could be rounding error. Check it.
import torch
tw = torch.tensor(w0, requires_grad=True, dtype=torch.float64)
L = torch.mean((tw * torch.tensor(x) - torch.tensor(y)) ** 2)
L.backward()
exact = float(tw.grad)
print(f"[4] PyTorch autograd says: {exact:+.8f}")
assert abs(exact - (-2.372898)) < 1e-5, "the measured derivative changed - re-check the data"

# === SECTION 5 - BREAK IT: MAKE THE NUDGE TOO SMALL ===
# The counter-intuitive part. Smaller is better... until floating point runs out of digits
# and the subtraction cancels away every significant figure. THIS is why autograd exists.
print("[5] break it - keep shrinking past what float64 can hold:")
for h in [1e-8, 1e-10, 1e-12, 1e-14, 1e-16]:
    r = (loss(w0 + h) - loss(w0)) / h
    print(f"      h = {h:<8.0e}  ratio = {r:+.8f}   error = {abs(r - exact):.2e}")

# === SECTION 6 - USE IT: WALK DOWNHILL ===
# The sign says which way, the size says how far. Repeat until it stops moving.
w, lr = 1.5, 0.35
for step in range(5):
    grad = float(np.mean(2 * x * (w * x - y)))
    w -= lr * grad
print(f"[6] after 5 steps: w = {w:.4f}   (the data was made with 3.0)")
