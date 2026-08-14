"""
LAB 2.3 - The Learning Rate, from scratch, on real handwritten digits.

Companion to episode 2.3. Folded into the episode (no standalone upload) and shipped as a
downloadable notebook.

    pip install numpy scikit-learn matplotlib
    python labs/s2_3_learning_rate/s2_3_learning_rate.py

THE WHOLE IDEA IN ONE LINE: the gradient says WHICH WAY to move; the learning rate decides
HOW FAR. That second number is the difference between a network that learns, one that
never gets going, and one that destroys itself.

The network is written here from scratch in numpy - no framework, ~40 lines - so you can
read every step. Same seeds as the episode, so you get the same figures it shows:
97.2%, 28.2%, 8.3%.

Dataset: sklearn's `load_digits` - 1,797 real handwritten digits, 8x8 pixels, 10 classes.
Deterministic, no downloads, runs in a few seconds.
"""
import numpy as np

# === SECTION 1 - REAL DATA ===
# Real-life analogy: a postcode reader on an envelope. 64 pixels in, one of ten digits out.
from sklearn.datasets import load_digits

SEED = 0
d = load_digits()
X, y = d.data / 16.0, d.target.astype(int)          # 0..16 greyscale -> 0..1
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
Xtr, ytr, Xte, yte = X[:1400], y[:1400], X[1400:], y[1400:]
print(f"[1] {len(Xtr)} training digits, {len(Xte)} held out (never seen while learning)")
print(f"    each digit is {X.shape[1]} pixels, and there are 10 possible answers")


# === SECTION 2 - THE NETWORK, BY HAND ===
# 64 inputs -> 32 hidden (ReLU) -> 10 outputs (softmax). 2,410 weights in total.
def init_params(n_in=64, n_hid=32, n_out=10, seed=SEED):
    r = np.random.default_rng(seed)
    return dict(W1=r.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hid)), b1=np.zeros(n_hid),
                W2=r.normal(0, np.sqrt(2.0 / n_hid), (n_hid, n_out)), b2=np.zeros(n_out))


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"]
    a1 = np.maximum(z1, 0.0)                        # ReLU
    z2 = a1 @ p["W2"] + p["b2"]
    z2 = z2 - z2.max(axis=1, keepdims=True)         # subtract the max: keeps exp() stable
    e = np.exp(z2)
    return z1, a1, e / e.sum(axis=1, keepdims=True)


def loss_and_grads(p, X, y):
    """Cross-entropy and its gradients, derived by hand rather than by autograd."""
    n = len(X)
    z1, a1, probs = forward(p, X)
    loss = float(-np.mean(np.log(probs[np.arange(n), y] + 1e-12)))
    d2 = probs.copy()
    d2[np.arange(n), y] -= 1.0                      # dL/dz2 for softmax + cross-entropy
    d2 /= n
    g = dict(W2=a1.T @ d2, b2=d2.sum(0))
    d1 = (d2 @ p["W2"].T) * (z1 > 0)                # ReLU passes gradient only where active
    g["W1"] = X.T @ d1
    g["b1"] = d1.sum(0)
    return loss, g


n_params = 64 * 32 + 32 + 32 * 10 + 10
print(f"[2] network built: 64 -> 32 -> 10 = {n_params:,} weights")


# === SECTION 3 - THE LOOP (this is all training is) ===
# measure the slope -> step against it -> repeat. The learning rate is the ONE new number.
def accuracy(p, X, y):
    return float(np.mean(forward(p, X)[2].argmax(1) == y))


def train(lr, steps=300, batch=64, seed=SEED):
    p = init_params(seed=seed)
    r = np.random.default_rng(seed + 1)
    losses, accs = [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            i = r.choice(len(Xtr), size=batch, replace=False)
            L, g = loss_and_grads(p, Xtr[i], ytr[i])
            losses.append(L); accs.append(accuracy(p, Xte, yte))
            for k in p:
                p[k] = p[k] - lr * g[k]              # <- the update rule, in full
    return np.array(losses), np.array(accs), p


# === SECTION 4 - THE EXPERIMENT: change ONE number ===
print("[4] same network, same data, same seed - only the learning rate differs:")
results = {}
for name, lr in (("just right", 0.5), ("too small", 0.002), ("too large", 60.0)):
    losses, accs, _ = train(lr)
    results[name] = (losses, accs)
    print(f"    lr={lr:<7} loss {losses[0]:.4f} -> {losses[-1]:8.4f}"
          f"   accuracy {accs[0]*100:5.1f}% -> {accs[-1]*100:5.1f}%   [{name}]")
assert results["just right"][1][-1] > 0.95, "the good run should reach 95%+"
assert results["too large"][1][-1] < 0.10, "the diverged run should be below chance"

# === SECTION 5 - EPOCH vs ITERATION vs BATCH (the interview question) ===
batch = 64
iters = int(np.ceil(len(Xtr) / batch))
print(f"[5] {len(Xtr):,} examples / batch {batch} = {iters} iterations = 1 EPOCH")
print(f"    300 steps = {300/iters:.1f} epochs;  10 epochs would be {10*iters:,} updates")
print("    an epoch is not a unit of TIME - it is one pass over the data")

# === SECTION 6 - BREAK IT: can the loss actually reach NaN? ===
# Everyone says a blown-up model gives you NaN. Test it rather than believe it.
print("[6] pushing the learning rate absurdly high and looking for NaN:")
for lr in (60, 2000, 200000):
    losses, _, _ = train(lr, steps=40)
    print(f"    lr={lr:<8} peak loss {np.nanmax(losses):7.2f}   any NaN? "
          f"{bool(np.isnan(losses).any())}")
print("    it never NaNs: a stable softmax bounds cross-entropy at about -log(1e-12) = 27.6,")
print("    so the loss just pins at that ceiling and accuracy collapses to chance.")

# === SECTION 7 - WHERE IT REALLY BREAKS: an UNBOUNDED loss ===
# Squared error has no ceiling, so it can grow without limit - and it overflows to inf.
# NOTE it becomes `inf`, NOT `nan`. Real NaN comes from arithmetic ON infinities.
print("[7] the same mistake on a squared-error loss (no ceiling):")
rng2 = np.random.default_rng(1)
A = rng2.normal(1.0, 0.5, 400)
target = 3.0 * A + rng2.normal(0, 0.1, 400)
w, lr = 0.0, 1.4                                    # past what this surface tolerates
with np.errstate(over="ignore", invalid="ignore"):
    for step in range(600):   # it needs a few hundred steps to actually overflow
        err = w * A - target
        L = float(np.mean(err ** 2))
        if step < 6 or not np.isfinite(L):
            print(f"    step {step:>3}   loss {L:.4g}")
        if not np.isfinite(L):
            break
        w = w - lr * float(np.mean(2 * A * err))

# === SECTION 8 - SEE IT ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (name, (losses, accs)), c in zip(results.items(), ("#3fb950", "#f0b429", "#ff8a80")):
    ax[0].plot(np.clip(losses, 0, 30), color=c, label=f"{name}")
    ax[1].plot(np.array(accs) * 100, color=c, label=f"{name}")
ax[1].axhline(10, ls="--", c="grey", lw=1, label="guessing (10%)")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("step"); ax[1].set_ylabel("test accuracy (%)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("lab_2.3_learning_rate.png", dpi=140)
print("[8] plot saved -> lab_2.3_learning_rate.png")
