"""
LAB 2.2 - The Gradient, from scratch, on a real photograph.

Folded into episode 2.2 (no standalone upload - CLAUDE.md golden rule 0c) and shipped as a
downloadable notebook. Every number the episode puts on screen is rebuilt here, so you can
check the claims instead of trusting them.

    pip install numpy torch opencv-python matplotlib
    python labs/s2_2_gradient/s2_2_gradient.py

THE SETUP, in one line: a washed-out photo, two dials (contrast and brightness), and one
score that says how far the result is from the correct picture.

    loss(w) = mean( (w0 * A + w1 - y)^2 )

A = the washed-out photo's pixels, y = the correct photo, w0 = contrast (multiply every
pixel), w1 = brightness (add to every pixel). That is ordinary two-parameter least squares
- the same maths as fitting a line - except here you can SEE the answer arrive.

Dataset: frog.jpg, shipped beside this file. Public domain (CC0), Wikimedia Commons,
"Portrait of a Bull Frog, mid summer, Cedar Mill Wetlands.jpg".

Deterministic (fixed seed), no downloads, runs in a few seconds.
"""
import pathlib

import numpy as np

# `__file__` does not exist inside a notebook, and this lab ships AS a notebook - so
# resolve the folder defensively and look for the image in the obvious places.
try:
    HERE = pathlib.Path(__file__).resolve().parent
except NameError:                                    # running in Jupyter
    HERE = pathlib.Path.cwd()


def _find(name):
    for base in (HERE, pathlib.Path.cwd(), pathlib.Path.cwd() / "labs" / "s2_2_gradient"):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(f"{name} must sit next to this notebook")

# === SECTION 1 - A PHOTO THAT IS WRONG ===
# Real-life analogy: a picture taken through a dirty window - the colours are all still
# there, just squashed into a narrow band of grey. Two numbers stretch them back out.
# We MAKE the washed-out version by inverting the fix, so we know the exact right answer
# and can check whether gradient descent actually finds it.
import cv2

C_TRUE, B_TRUE = 1.8, -0.35          # the fix this photo needs
NOISE = 0.004                        # so the bottom of the bowl is real, not exactly zero

img = cv2.imread(str(_find("frog.jpg")), cv2.IMREAD_COLOR)
assert img is not None, "frog.jpg could not be read"
h = int(round(240 * img.shape[0] / img.shape[1]))
y = cv2.resize(img, (240, h), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
rng = np.random.default_rng(0)
A = (y - B_TRUE) / C_TRUE + rng.normal(0, NOISE, y.shape)

print(f"[1] photo: {A.shape[1]}x{A.shape[0]} pixels x 3 colours = {A.size:,} numbers")
print(f"    washed out: mean {A.mean():.3f} std {A.std():.3f}")
print(f"    correct   : mean {y.mean():.3f} std {y.std():.3f}")


# === SECTION 2 - ONE NUMBER FOR HOW WRONG IT IS ===
# Square every miss (so too-bright and too-dark both count as wrong), then average.
def loss(w):
    return float(np.mean((w[0] * A + w[1] - y) ** 2))


def shades(score):
    """The score in units a human can picture: average error in shades out of 255."""
    return np.sqrt(max(score, 0.0)) * 255.0


w0 = np.array([1.0, 0.0])            # the photo exactly as it is: no stretch, no shift
print(f"[2] score of the untouched photo: {loss(w0):.5f}"
      f"   (every pixel off by ~{shades(loss(w0)):.0f} shades out of 255)")

# === SECTION 3 - WIGGLE EACH DIAL SEPARATELY ===
# The whole idea, and it is just episode 2.1's question asked once per dial.
h_ = 1e-5


def finite_diff_grad(w):
    out = []
    for i in range(len(w)):
        wp = w.copy()
        wp[i] += h_
        out.append((loss(wp) - loss(w)) / h_)
    return np.array(out)


g_fd = finite_diff_grad(w0)
print(f"[3] turn CONTRAST a hair   -> {g_fd[0]:+.5f}")
print(f"    turn BRIGHTNESS a hair -> {g_fd[1]:+.5f}")
print(f"    stack them and you have the gradient: [{g_fd[0]:+.5f}, {g_fd[1]:+.5f}]")
print(f"    its length (how steep it is here): {np.linalg.norm(g_fd):.5f}")


# === SECTION 4 - BUILD THAT NUMBER ONE REAL PIXEL AT A TIME ===
# Nothing is asserted here. The first component is an average over every pixel, and this
# is what each pixel actually contributes.
def exact_grad(w, A=A, y=y):
    r = w[0] * A + w[1] - y
    return np.array([float(np.mean(2 * A * r)), float(np.mean(2 * r))])


print("[4] the contrast number, computed pixel by pixel:")
pick = np.random.default_rng(3).choice(A.size, size=4, replace=False)
flatA, flaty = A.ravel(), y.ravel()
run = 0.0
for n, i in enumerate(pick, 1):
    now = w0[0] * flatA[i] + w0[1]
    err = now - flaty[i]
    term = 2 * flatA[i] * err
    run += term
    print(f"      pixel {n}: is {now:.3f}  should be {flaty[i]:.3f}  off {err:+.3f}"
          f"  ->  2 x {flatA[i]:.3f} x {err:+.3f} = {term:+.4f}   running {run/n:+.4f}")
g = exact_grad(w0)
print(f"      ...all {A.size:,} numbers average to {g[0]:+.5f}")

# === SECTION 5 - VALIDATE AGAINST AUTOGRAD ===
# Two methods that agree could still be wrong the same way. Ask a third, independent one.
import torch

tw = torch.tensor(w0, requires_grad=True, dtype=torch.float64)
tA, ty = torch.tensor(A), torch.tensor(y)
torch.mean((tw[0] * tA + tw[1] - ty) ** 2).backward()
g_auto = tw.grad.numpy().copy()
print(f"[5] finite differences {g_fd}\n    exact formula      {g}\n    autograd           {g_auto}")
assert np.allclose(g, g_auto, atol=1e-9), "autograd disagrees"
assert np.allclose(g_fd, g, atol=1e-4), "finite differences disagree"
print("    all three agree.")

# === SECTION 6 - THE RACE: 5,000 RANDOM DIRECTIONS ===
# The episode's headline claim. Note the TWO scorings - the difference is the honest part,
# and it is the reason episode 2.3 (step size) has to exist.
n_random, step = 5000, 0.05
rng2 = np.random.default_rng(1)
gu = g / np.linalg.norm(g)
grad_slope = float(g @ gu)
grad_rise = loss(w0 + step * gu) - loss(w0)
beat_slope = beat_rise = 0
for _ in range(n_random):
    d = rng2.normal(size=2)
    d /= np.linalg.norm(d)
    if float(g @ d) > grad_slope + 1e-12:
        beat_slope += 1
    if loss(w0 + step * d) - loss(w0) > grad_rise:
        beat_rise += 1
print(f"[6] steepest INSTANTANEOUSLY: {beat_slope} of {n_random} random directions beat it")
print(f"    after a real step of {step}: {beat_rise} of {n_random} beat it "
      f"({beat_rise/n_random*100:.2f}%)  <- curvature, i.e. episode 2.3")
assert beat_slope == 0, "a random direction beat the gradient instantaneously - impossible"

# === SECTION 7 - WHY NONE OF THEM *CAN* WIN ===
# The race shows it is true. One line of arithmetic shows it must be: any unit direction d
# scores g . d = |g| cos(angle). Cosine never exceeds 1, and equals 1 in exactly one place.
glen = float(np.linalg.norm(g))
print(f"[7] |g| = {glen:.5f}; every direction scores |g| x cos(angle):")
for a in (0, 30, 60, 90):
    th = np.radians(a)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    print(f"      {a:>3} deg off: g.d = {float(g @ (R @ gu)):+.5f}"
          f"   = {glen:.5f} x cos = {glen*np.cos(th):+.5f}")

# === SECTION 8 - PERPENDICULAR TO THE CONTOUR, EXACTLY ===
# Walk along a contour and the score does not change, so the direction of fastest change
# has to be at a right angle to it. Not approximately - exactly, even in floating point.
tangent = np.array([-g[1], g[0]])
dot = float(g @ tangent)
print(f"[8] gradient . contour tangent = {dot:.2e}   angle = "
      f"{np.degrees(np.arccos(np.clip(dot/(glen*np.linalg.norm(tangent)), -1, 1))):.4f} degrees")

# === SECTION 9 - BREAK IT: THE GRADIENT NOBODY ACTUALLY COMPUTES ===
# Everything above used all ~97,000 numbers. No real model does that - GPT-2 never sees its
# whole training set before taking a step. Use a random handful of pixels instead.
# Real-life analogy: asking a hundred people in a city which way is north. You do not need
# to ask everyone to get a good answer.
print(f"[9] gradients from only SOME of the {A.size:,} numbers (median over 300 draws):")
rng3 = np.random.default_rng(7)
for size in (1, 10, 100, 1000):
    angs = []
    for _ in range(300):
        idx = rng3.choice(A.size, size=size, replace=False)
        gb = exact_grad(w0, flatA[idx], flaty[idx])
        nb = np.linalg.norm(gb)
        angs.append(90.0 if nb < 1e-12 else
                    np.degrees(np.arccos(np.clip(float(gb @ gu) / nb, -1, 1))))
    angs = np.array(angs)
    name = "stochastic" if size == 1 else "mini-batch"
    print(f"      {size:>5} numbers -> {np.median(angs):5.1f} deg off, "
          f"{np.mean(angs < 90)*100:5.1f}% still downhill, "
          f"{A.size/size:8.0f}x cheaper   [{name}]")

# === SECTION 10 - USE IT: WALK DOWNHILL AND WATCH THE PHOTO COME RIGHT ===
# Turn the arrow around and take small steps. The step size is COMPUTED, not guessed: the
# loss gradient is 2M(w - w*), so descent only converges while lr < 1/lambda_max.
M = np.array([[float(np.mean(A * A)), float(np.mean(A))], [float(np.mean(A)), 1.0]])
lr = 0.9 / float(np.linalg.eigvalsh(M).max())
w = w0.copy()
print(f"[10] descending with the computed step size lr = {lr:.3f}")
for i in range(1, 401):
    w -= lr * exact_grad(w)
    if i in (5, 25, 100, 400):
        print(f"      step {i:>3}: contrast {w[0]:.3f} brightness {w[1]:+.3f}"
              f"  score {loss(w):.6f}  (~{shades(loss(w)):.0f} shades off)")
print(f"     the photo really needed contrast {C_TRUE}, brightness {B_TRUE:+.2f}."
      f"  We found {w[0]:.3f}, {w[1]:+.3f}.")

out = pathlib.Path.cwd() / "lab_2.2_gradient.png"
out.parent.mkdir(parents=True, exist_ok=True)
strip = np.hstack([np.clip(A, 0, 1), np.clip(w[0] * A + w[1], 0, 1), y])
cv2.imwrite(str(out), np.rint(strip * 255).astype(np.uint8))
print(f"     before / fixed / target  ->  {out}")
