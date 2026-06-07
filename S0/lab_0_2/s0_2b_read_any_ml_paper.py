"""
LAB 0.2b - Read Any ML Paper: decode a real training objective, then BUILD it.

Pairs with concept episode 0.2b ("Advanced Notation, Part 2: Read Any ML Paper").

The dense line that scares people in papers:

        L(w) = E_{(x,y)~D} [ - sum_k  y_k * log softmax(f_w(x))_k ]   +  lambda * ||w||_2^2

reads, symbol by symbol, as a sentence:
    "the AVERAGE (E) over the data of the NEGATIVE LOG-PROBABILITY the model
     assigns to the correct class (cross-entropy), plus a penalty on the SIZE
     (L2 norm) of the weights."

We build each PIECE on its own (softmax -> one-hot -> cross-entropy -> L2 penalty),
explain it, then RUN THE FULL BUILD on the built-in scikit-learn IRIS dataset
(150 flowers, 4 features, 3 classes; runs on any laptop, no download), validate it
against scikit-learn and PyTorch, break it (log(0) -> nan), fix it, and visualize.

REAL-LIFE ANALOGIES (same as the concept episode):
  - E[.] expectation  = a weighted class average (averaging grades across a class)
  - ||.|| norm        = a tape measure for a vector (how long the weight arrow is)

Install:  pip install numpy scikit-learn torch matplotlib pyyaml
Run:      python labs/s0_2b_read_any_ml_paper.py
"""
import numpy as np
import matplotlib.pyplot as plt

# House palette for plots (kept in sync with the channel) -------------------
import yaml, pathlib
_cfg = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "config" / "channel.yaml"))
P = _cfg["palette"]
plt.rcParams.update({
    "figure.facecolor": P["bg"], "axes.facecolor": P["bg"],
    "axes.edgecolor": P["text"], "axes.labelcolor": P["text"], "text.color": P["text"],
    "xtick.color": P["text"], "ytick.color": P["text"],
    "axes.grid": True, "grid.color": "#243042", "font.size": 12,
})
np.random.seed(7)   # deterministic so the video matches what viewers reproduce

OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "thumbnails"  # plots dir
OUT.mkdir(parents=True, exist_ok=True)


# === SECTION 1 - RECAP ======================================================
# In the concept episode (0.2b) we learned to READ the scary symbols. Part 1 gave
# us the alphabet (sum, product, indices); this lab reads a full SENTENCE -- a real
# loss function -- and proves a paper formula is just a handful of simple operations.
# We build each piece on its own, then run the whole thing on real data.


# === SECTION 2 - SOFTMAX ====================================================
def softmax(z):
    """softmax(z)_k = e^{z_k} / sum_j e^{z_j}, computed stably.

    Turns raw scores into a probability distribution: the exponentials make every
    score POSITIVE, and dividing by their sum makes them add up to one (a vote share).
    Subtracting the row max is the standard trick to avoid overflow in e^z.
    """
    z = z - z.max(axis=1, keepdims=True)             # shift: does not change the result
    e = np.exp(z)                                    # e^{z_k}
    return e / e.sum(axis=1, keepdims=True)          # divide by sum_j e^{z_j}


# === SECTION 3 - ONE-HOT LABELS =============================================
def one_hot(idx, n_classes):
    """y_k in {0,1}: a 1 in the true-class slot, 0 elsewhere.

    This is the y_k in the formula - it simply PICKS OUT the true class so the sum
    keeps only the term we care about.
    """
    oh = np.zeros((len(idx), n_classes))
    oh[np.arange(len(idx)), idx] = 1.0
    return oh


# === SECTION 4 - CROSS-ENTROPY ==============================================
def cross_entropy_from_scratch(logits, y_idx):
    """Decode  L = E[ - sum_k y_k log p_k ]  one symbol at a time."""
    p = softmax(logits)                              # softmax(f_w(x)) -> probabilities p_k
    y = one_hot(y_idx, logits.shape[1])              # y_k : one-hot true labels
    log_p = np.log(p)                                # log p_k  (the surprise of each guess)
    per_sample = -np.sum(y * log_p, axis=1)          # - sum_k y_k log p_k  (one number / sample)
    # ANALOGY: E[.] is just a WEIGHTED CLASS AVERAGE - like averaging grades across a class.
    return per_sample.mean()                         # E[...] : average over the dataset


# === SECTION 5 - L2 PENALTY =================================================
def l2_penalty(w, lam):
    """lambda * ||w||_2^2 : the squared L2 norm measures the SIZE of the weights.

    ANALOGY: ||.|| is a TAPE MEASURE for a vector - how long the weight arrow is.
    """
    return lam * np.sum(w ** 2)                      # ||w||_2^2 = sum_i w_i^2


# === SECTION 6 - PUT IT ALL TOGETHER ========================================
# Now run the FULL build on one small REAL dataset: scikit-learn's built-in IRIS.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y_true = load_iris(return_X_y=True)               # 150 samples, 4 features, 3 classes
X = StandardScaler().fit_transform(X)                # standardize so training converges cleanly
clf = LogisticRegression(max_iter=1000).fit(X, y_true)
logits = clf.decision_function(X)                    # f_w(x): the model's raw scores (N, K)

my_ce = cross_entropy_from_scratch(logits, y_true)   # our hand-built loss
reg = l2_penalty(clf.coef_, lam=0.01)                # measured on the REAL model weights
print(f"[build]    cross-entropy (from scratch) = {my_ce:.6f}")
print(f"[build]    L2 penalty  lambda*||w||^2   = {reg:.6f}  (lambda=0.01)")


# === SECTION 7 - VALIDATE ===================================================
# Same number, two trusted libraries. If our decoding is right, they agree.
from sklearn.metrics import log_loss
import torch
import torch.nn.functional as F

probs = softmax(logits)
sk_ce = log_loss(y_true, probs)                                          # scikit-learn
pt_ce = F.cross_entropy(torch.tensor(logits),
                        torch.tensor(y_true)).item()                     # PyTorch (logits in)

print(f"[validate] scikit-learn log_loss        = {sk_ce:.6f}")
print(f"[validate] torch F.cross_entropy        = {pt_ce:.6f}")
assert np.isclose(my_ce, sk_ce, atol=1e-6), "scratch != sklearn"
assert np.isclose(my_ce, pt_ce, atol=1e-6), "scratch != torch"
print("[validate] all three match -> our symbol-by-symbol decoding is correct.")


# === SECTION 8 - BREAK IT ON PURPOSE ========================================
# The trap everyone hits: a predicted probability of exactly 0 for the true class
# makes log(0) = -inf, so the loss becomes nan. This is WHY real code never feeds
# raw probabilities into log -- it works in log-space or clips with a tiny epsilon.
confident_wrong = np.array([[0.0, 1.0]])     # model is 100% sure of class 1...
y_is_class0 = np.array([0])                  # ...but the truth is class 0
with np.errstate(divide="ignore"):           # the log(0) below is the whole point
    broken = -np.log(confident_wrong[np.arange(1), y_is_class0])
print(f"\n[break]    naive -log(p_true) with p=0  = {broken[0]}   <- nan/inf, training dies")

eps = 1e-12
fixed = -np.log(np.clip(confident_wrong[np.arange(1), y_is_class0], eps, 1.0))
print(f"[fix]      clip to eps then -log         = {fixed[0]:.4f}   <- large but finite")
print("[fix]      lesson: dense notation hid a simple op; the bug was numerical, not conceptual.")


# === SECTION 9 - VISUALIZE ==================================================
# Cross-entropy for a single sample as a function of the probability the model
# gave to the CORRECT class. As p_true -> 0 the penalty explodes; at p_true = 1
# it is zero. This curve is the whole reason "be confident AND wrong" is costly.
p_true = np.linspace(1e-3, 1.0, 500)
loss = -np.log(p_true)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(p_true, loss, color=P["red"], lw=3, label=r"$-\log(p_{\mathrm{true}})$")
ax.axhline(0, color=P["green"], lw=1.2, ls="--")
ax.scatter([1.0], [0.0], color=P["green"], zorder=5, s=60, label="confident & right (loss 0)")
ax.annotate("confident & WRONG\n(loss explodes)",
            xy=(0.02, -np.log(0.02)), xytext=(0.30, 5.2),
            color=P["yellow"], arrowprops=dict(color=P["yellow"], arrowstyle="->"))
ax.set_xlabel("probability assigned to the TRUE class")
ax.set_ylabel("cross-entropy loss")
ax.set_title("Read Any ML Paper 0.2b - what cross-entropy actually penalizes (iris)")
ax.legend(loc="upper right")
fig.tight_layout()
plot_path = OUT / "lab_0.2b_cross_entropy.png"
fig.savefig(plot_path, dpi=140)
print(f"\n[viz]      saved plot -> {plot_path}")


if __name__ == "__main__":
    print("\nLab 0.2b complete: built each piece, ran the full loss on iris, "
          "validated it, broke it, fixed it.")
