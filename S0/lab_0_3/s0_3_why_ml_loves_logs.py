"""
LAB 0.3 - Why ML Loves Logs: log-scale loss, overflow-safe softmax, and confidence.

Pairs with concept episode 0.3 ("Exponentials & Logarithms"). The episode showed
WHY logs matter; this lab puts them to work on a real classifier and walks three
use cases you hit in real machine learning:

  USE CASE 1  Log-scale loss reveal - a falling loss curve looks like a cliff on a
              linear axis but a near-straight line on a LOG axis (hidden structure
              appears; "every ML loss is secretly exponential").
  USE CASE 2  Overflow-safe softmax - naive exp() overflows to inf -> nan on big
              scores; the log-sum-exp / subtract-the-max trick keeps it finite.
  USE CASE 3  Confidence / out-of-distribution - the model's top softmax probability
              is a confidence score; feed it noise and that confidence collapses.

To get there we first build the tools the episode named - softmax, log-softmax
(log-sum-exp), and cross-entropy (the log-likelihood) - by hand in NumPy, then
validate them against scipy / scikit-learn.

DATASET: the built-in scikit-learn DIGITS set (1797 handwritten 8x8 digits, 10
classes) - no download, runs on any laptop.

REAL-LIFE ANALOGIES (mirrored in the narration):
  - softmax        = turning raw scores into a "share of the vote" (they sum to 1)
  - cross-entropy  = adding up the surprise of each prediction (log-likelihood)
  - subtract-the-max = measuring everyone's height relative to the tallest person,
                       so the numbers never blow up
  - log axis       = a zoomed-out map that fits a huge range on one page
  - softmax confidence = how sure a spam filter is, not just its yes/no answer

Install:  pip install numpy scipy scikit-learn matplotlib pyyaml
Run:      python labs/s0_3_why_ml_loves_logs.py
"""
import numpy as np
import matplotlib.pyplot as plt

# House palette for plots (kept in sync with the channel) -------------------
import yaml, pathlib
_cfg = yaml.safe_load(open(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config" / "channel.yaml").exists()) / "config" / "channel.yaml"))
P = _cfg["palette"]
plt.rcParams.update({
    "figure.facecolor": P["bg"], "axes.facecolor": P["bg"],
    "axes.edgecolor": P["text"], "axes.labelcolor": P["text"], "text.color": P["text"],
    "xtick.color": P["text"], "ytick.color": P["text"],
    "axes.grid": True, "grid.color": "#243042", "font.size": 12,
})
np.random.seed(7)   # deterministic so the video matches what viewers reproduce

OUT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "config" / "channel.yaml").exists()) / "outputs" / "thumbnails"  # plots dir
OUT.mkdir(parents=True, exist_ok=True)


# === SECTION 1 - RECAP ======================================================
# The episode's three facts we will lean on, checked in code:
#   1. exponential growth outruns linear growth (it is proportional to its own size)
#   2. the log is the inverse: log_b(b^x) = x  ("to what power?")
#   3. the log's superpower: log(a*b) = log(a) + log(b)  (turns x into +)
steps = np.arange(0, 11)
linear = 5 * steps                       # +5 each step
expo = 2.0 ** steps                      # doubles each step
print(f"[recap] after 10 steps: linear 5x -> {linear[-1]},  exponential 2^x -> {expo[-1]}")
print(f"[recap] log2(1024) = {np.log2(1024):.0f}   <- 'how many doublings?' (the inverse)")
a, b = 8.0, 32.0
print(f"[recap] log(a*b)={np.log(a*b):.4f}  vs  log(a)+log(b)={np.log(a)+np.log(b):.4f}"
      f"   <- multiply becomes add")
assert np.isclose(np.log(a * b), np.log(a) + np.log(b))


# === SECTION 2 - BUILD THE TOOLS (SOFTMAX, LOG-SUM-EXP, CROSS-ENTROPY) =======
def log_sum_exp(Z, axis=1):
    """Stable log(sum(exp(Z))) - subtract the row max first, add it back after.

    ANALOGY: measure everyone's height relative to the TALLEST person; the ranking
    is unchanged but no number ever blows up. (This is the whole trick.)
    """
    m = Z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(Z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def log_softmax(Z):
    """log of the softmax, computed the stable way: Z - logsumexp(Z)."""
    return Z - log_sum_exp(Z, axis=1)[:, None]


def softmax(Z):
    """Turn raw scores (logits) into probabilities that sum to one.

    ANALOGY: softmax is a 'share of the vote' - bigger score, bigger slice, total 100%.
    """
    return np.exp(log_softmax(Z))


def cross_entropy(Z, Y_onehot):
    """Mean cross-entropy = mean negative log-likelihood of the true classes.

    ANALOGY: add up the SURPRISE of each prediction; being confidently wrong hurts most.
    """
    return -(Y_onehot * log_softmax(Z)).sum(axis=1).mean()


# === SECTION 3 - TRAIN A SOFTMAX CLASSIFIER ON DIGITS =======================
# One real dataset, one tiny model: multinomial logistic (softmax) regression,
# trained by plain gradient descent. We record the loss at every step.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)                       # (1797, 64), labels 0..9
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
scaler = StandardScaler().fit(Xtr)
Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
N, D, K = Xtr.shape[0], Xtr.shape[1], 10
Ytr = np.eye(K)[ytr]                                      # one-hot targets

W, bvec = np.zeros((D, K)), np.zeros(K)
lr, n_steps = 0.5, 300
losses = []
for t in range(n_steps):
    Z = Xtr @ W + bvec
    losses.append(cross_entropy(Z, Ytr))
    gZ = (softmax(Z) - Ytr) / N                           # softmax+CE gradient
    W -= lr * (Xtr.T @ gZ)
    bvec -= lr * gZ.sum(axis=0)
losses = np.array(losses)
test_acc = (softmax(Xte @ W + bvec).argmax(1) == yte).mean()
print(f"[train] start loss {losses[0]:.3f} (= ln 10, a random guess)  ->  "
      f"end loss {losses[-1]:.3f}")
print(f"[train] test accuracy = {test_acc*100:.1f}%   over {n_steps} gradient steps")


# === SECTION 4 - USE CASE 1: A LOG SCALE REVEALS HIDDEN STRUCTURE ===========
# On a LINEAR axis the loss looks like a cliff then a flat line - you cannot read
# the late progress. Put BOTH axes on a log scale and the hidden structure appears:
# a near-straight line. The loss is following a POWER LAW (loss ~ step^p) - the same
# shape as the "scaling laws" that govern how big models improve with more compute.
# (An exponential drop would instead be straight on a log-Y axis; either way a log
#  scale exposes what a linear axis hides.) ANALOGY: a log scale is a zoomed-out map.
t = np.arange(1, n_steps)
p_exp = np.polyfit(np.log(t), np.log(losses[1:]), 1)[0]
r2 = np.corrcoef(np.log(t), np.log(losses[1:]))[0, 1] ** 2
print(f"[reveal] loss fell {losses[0]/losses[-1]:.0f}x; on a log-log axis it is a "
      f"straight line: loss ~ step^{p_exp:.2f}  (R^2 = {r2:.3f}, a power law)")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))
axL.plot(losses, color=P["red"], lw=2.5)
axL.set(title="Linear axis: a cliff, then 'nothing'", xlabel="gradient step",
        ylabel="cross-entropy loss")
axR.loglog(t, losses[1:], color=P["green"], lw=2.5)
axR.set(title=f"Log-log axis: a straight line  (loss ~ step^{p_exp:.2f})",
        xlabel="gradient step (log)", ylabel="cross-entropy loss (log)")
fig.suptitle("Lab 0.3 - USE CASE 1: a log scale reveals the loss's hidden structure",
             color=P["text"])
fig.tight_layout()
loss_path = OUT / "lab_0.3_logloss.png"
fig.savefig(loss_path, dpi=140)
print(f"[reveal] saved plot -> {loss_path}")


# === SECTION 5 - VALIDATE AGAINST THE LIBRARIES =============================
# Our hand-built pieces must match the trusted implementations exactly.
from scipy.special import softmax as sp_softmax, log_softmax as sp_logsoftmax
from sklearn.metrics import log_loss

Zte = Xte @ W + bvec
assert np.allclose(softmax(Zte), sp_softmax(Zte, axis=1), atol=1e-12), "softmax != scipy"
assert np.allclose(log_softmax(Zte), sp_logsoftmax(Zte, axis=1), atol=1e-12), "log_softmax != scipy"
our_ce = cross_entropy(Zte, np.eye(K)[yte])
lib_ce = log_loss(yte, sp_softmax(Zte, axis=1), labels=list(range(K)))
assert np.isclose(our_ce, lib_ce, atol=1e-8), "cross-entropy != sklearn log_loss"
print(f"[validate] our softmax / log-softmax match scipy, and our cross-entropy "
      f"{our_ce:.4f} == sklearn log_loss {lib_ce:.4f}")


# === SECTION 6 - USE CASE 2: BREAK IT (OVERFLOW), THEN FIX IT ===============
# Big logits are normal once a network is confident. The naive softmax does
# exp() FIRST, and exp(1000) overflows to +inf, so inf/inf = nan and training dies.
big = np.array([[1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big) / np.exp(big).sum()
print(f"[break] naive exp-first softmax on logits ~1000 -> {naive}  <- nan, dead")
safe = softmax(big)
print(f"[fix]   our subtract-the-max softmax        -> {np.round(safe, 4)}  "
      f"(sums to {safe.sum():.4f})")
assert np.isfinite(safe).all() and np.isclose(safe.sum(), 1.0)
print("[fix]   same idea as log-sum-exp: shift by the max, math unchanged, no overflow.")


# === SECTION 7 - USE CASE 3: CONFIDENCE / OUT-OF-DISTRIBUTION ===============
# The top softmax probability is a CONFIDENCE score, not just a label. On real
# digits it is high; on pure noise (something the model has never seen) it
# collapses - a simple "I am not sure" / out-of-distribution detector.
# ANALOGY: like a spam filter's certainty, not just its yes/no verdict.
from sklearn.metrics import roc_auc_score

Pte = softmax(Xte @ W + bvec)
conf_real = Pte.max(axis=1)                               # confidence on real digits

rng = np.random.default_rng(0)
noise = rng.normal(size=(len(Xte), D))                    # already standardized scale
conf_noise = softmax(noise @ W + bvec).max(axis=1)        # confidence on noise

# confidence as an OOD score: how well does it separate digits from noise?
auroc = roc_auc_score(np.r_[np.ones(len(conf_real)), np.zeros(len(conf_noise))],
                      np.r_[conf_real, conf_noise])
thresh = np.percentile(conf_real, 5)                     # reject below 5th pct of real digits
caught = (conf_noise < thresh).mean()
false_reject = (conf_real < thresh).mean()
print(f"[confid] median confidence: real digits {np.median(conf_real):.2f}  vs  "
      f"pure noise {np.median(conf_noise):.2f}")
print(f"[confid] confidence separates digits from noise with AUROC = {auroc:.3f}")
print(f"[confid] reject below {thresh:.2f}: catches {caught*100:.0f}% of noise, "
      f"wrongly rejects only {false_reject*100:.0f}% of real digits")

# Visualize: one confident digit's vote, plus the confidence split (digits vs noise)
i = int(conf_real.argmax())                               # a very confident test digit
img = scaler.inverse_transform(Xte[i:i+1]).reshape(8, 8)
fig2, (axImg, axBar, axHist) = plt.subplots(1, 3, figsize=(12.5, 4.0))
axImg.imshow(img, cmap="gray"); axImg.set_title(f"a test digit (true={yte[i]})")
axImg.grid(False); axImg.set_xticks([]); axImg.set_yticks([])
axBar.bar(range(K), Pte[i], color=P["blue"])
axBar.set(title=f"its softmax vote -> {Pte[i].argmax()} ({conf_real[i]*100:.0f}% sure)",
          xlabel="digit class", ylabel="probability", xticks=range(K))
axHist.hist(conf_real, bins=20, color=P["green"], alpha=0.7, label="real digits")
axHist.hist(conf_noise, bins=20, color=P["red"], alpha=0.6, label="pure noise")
axHist.axvline(thresh, color=P["yellow"], lw=2, ls="--", label=f"reject < {thresh:.2f}")
axHist.set(title="confidence = top softmax prob", xlabel="max probability", ylabel="count")
axHist.legend(loc="upper center", fontsize=9)
fig2.suptitle("Lab 0.3 - USE CASE 3: softmax confidence flags the unknown", color=P["text"])
fig2.tight_layout()
conf_path = OUT / "lab_0.3_confidence.png"
fig2.savefig(conf_path, dpi=140)
print(f"[confid] saved plot -> {conf_path}")


if __name__ == "__main__":
    print("\nLab 0.3 complete: built softmax / log-sum-exp / cross-entropy by hand, "
          "trained a digit classifier, revealed the loss as a power-law straight line on "
          "a log-log axis, made softmax overflow-safe, and used its confidence to flag noise.")
