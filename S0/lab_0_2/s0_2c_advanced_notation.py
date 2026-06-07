"""
LAB 0.2c - Read Any ML Paper, Part 3: BUILD the symbols inside the VAE's loss.

Pairs with concept episode 0.2c ("Advanced Notation, Part 3: The Deepest Symbols").

Part 1 was the alphabet, Part 2 full sentences; Part 3 is the deepest symbols -
the Gaussian N, variance, normalization layers, and the ELBO that trains VAEs:

    ELBO(x) = E_q[ log p(x|z) ]  -  KL( q(z|x) || p(z) )
              \----- reconstruction -----/   \--- regularizer ---/

We build each piece by hand in NumPy (Gaussian pdf -> log-likelihood -> LayerNorm ->
KL to the standard normal), then ASSEMBLE the ELBO as one untrained VAE forward pass
on a real dataset, validate every piece against scipy / PyTorch, break it (sigma -> 0),
fix it, and visualize the KL term pulling a posterior toward the prior.

DATASET: the built-in scikit-learn IRIS set (150 flowers, 4 features) - no download,
runs on any laptop.

REAL-LIFE ANALOGIES (same as the concept episode):
  - N  Gaussian   = the bell curve of natural variation (heights in a crowd)
  - E[.] expectation = a weighted average (averaging grades across a class)
  - LayerNorm     = re-leveling audio so every track sits at the same volume
  - KL divergence = the extra surprise from using the wrong map (callback 3.8 / 0.2b)

Install:  pip install numpy scipy scikit-learn torch matplotlib pyyaml
Run:      python labs/s0_2c_advanced_notation.py
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
TWO_PI = 2.0 * np.pi


# === SECTION 1 - RECAP ======================================================
# In the concept episode (0.2c) we learned to READ the deepest symbols: the
# Gaussian N, variance, normalization layers, and the ELBO that trains VAEs.
# This lab BUILDS those exact pieces by hand and assembles them into the ELBO.


# === SECTION 2 - THE GAUSSIAN (N) ===========================================
def gaussian_pdf(x, mu, sigma):
    """N(x; mu, sigma^2) = exp(-1/2 ((x-mu)/sigma)^2) / (sigma * sqrt(2*pi)).

    ANALOGY: the Gaussian is the bell curve of natural variation - heights in a crowd.
    """
    z = (x - mu) / sigma
    return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(TWO_PI))


# === SECTION 3 - GAUSSIAN LOG-LIKELIHOOD ====================================
def gaussian_log_likelihood(x, mu, sigma):
    """log N(x; mu, sigma^2), summed over the last axis (the reconstruction term).

    Working in log-space turns a tiny product of probabilities into a stable sum.
    """
    z = (x - mu) / sigma
    log_p = -0.5 * np.log(TWO_PI) - np.log(sigma) - 0.5 * z ** 2
    return log_p.sum(axis=-1)                          # one number per sample


# === SECTION 4 - LAYER NORM =================================================
def layer_norm(x, eps=1e-5):
    """Standardize each ROW across its features: (x - mean) / sqrt(var + eps).

    ANALOGY: LayerNorm re-levels the audio so every track sits at the same volume.
    The eps in the denominator is the guard that keeps a flat row from dividing by zero.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# === SECTION 5 - KL TO THE STANDARD NORMAL ==================================
def kl_standard_normal(mu, log_var):
    """KL( N(mu, sigma^2) || N(0, 1) ) in closed form (the ELBO's regularizer).

    = 1/2 * sum( exp(log_var) + mu^2 - 1 - log_var ).  ANALOGY: the extra surprise
    you pay for using the wrong map - it is zero only when q already equals the prior.
    """
    return 0.5 * np.sum(np.exp(log_var) + mu ** 2 - 1.0 - log_var, axis=-1)


# === SECTION 6 - PUT IT ALL TOGETHER: THE ELBO ==============================
# Assemble one untrained VAE forward pass on the real IRIS data.
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, _ = load_iris(return_X_y=True)                      # 150 samples, 4 features
X = StandardScaler().fit_transform(X)                  # standardize the inputs
N, D, H = X.shape[0], X.shape[1], 2                    # latent dim H = 2

W_enc = 0.3 * np.random.randn(D, H)                    # tiny fixed "encoder" x -> mu
V_dec = 0.3 * np.random.randn(H, D)                    # tiny fixed "decoder" z -> x_hat

mu = X @ W_enc                                         # q(z|x) mean, shape (N, H)
log_var = np.zeros((N, H))                             # q variance = 1 (no sampling, deterministic)
z = mu                                                 # the latent code
x_hat = z @ V_dec                                      # decoder reconstruction

recon = gaussian_log_likelihood(X, x_hat, sigma=1.0).mean()   # E_q[ log p(x|z) ]
kl = kl_standard_normal(mu, log_var).mean()                   # KL( q || p )
elbo = recon - kl                                             # the ELBO
print(f"[build]    reconstruction  E[log p(x|z)] = {recon:.4f}")
print(f"[build]    regularizer     KL(q || p)    = {kl:.4f}")
print(f"[build]    ELBO = recon - KL             = {elbo:.4f}  (untrained, so low)")


# === SECTION 7 - VALIDATE ===================================================
# Each hand-built piece must match the trusted library.
from scipy.stats import norm
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

xs = np.linspace(-3, 3, 11)
assert np.allclose(gaussian_pdf(xs, 0.3, 1.2), norm.pdf(xs, 0.3, 1.2)), "pdf != scipy"

xt = torch.tensor(X, dtype=torch.float64)
ours_ln = layer_norm(X)
lib_ln = F.layer_norm(xt, (D,)).numpy()
assert np.allclose(ours_ln, lib_ln, atol=1e-6), "layer_norm != torch"

mt = torch.tensor(mu); st = torch.ones_like(mt)
lib_kl = kl_divergence(Normal(mt, st), Normal(0.0, 1.0)).sum(-1).mean().item()
assert np.isclose(kl, lib_kl, atol=1e-6), "KL != torch.distributions"
print("[validate] Gaussian, LayerNorm, and KL all match scipy / PyTorch.")


# === SECTION 8 - BREAK IT ON PURPOSE ========================================
# The trap: a Gaussian with sigma = 0, or LayerNorm on a flat row, divides by zero.
flat = np.array([[2.0, 2.0, 2.0, 2.0]])               # every feature identical -> var = 0
with np.errstate(divide="ignore", invalid="ignore"):
    broken = layer_norm(flat, eps=0.0)
print(f"\n[break]    LayerNorm with eps=0 on a flat row -> {broken[0, 0]}   <- nan, training dies")
fixed = layer_norm(flat, eps=1e-5)
print(f"[fix]      same row with eps=1e-5            -> {fixed[0, 0]:.4f}   <- finite and safe")
print("[fix]      lesson: the scary symbols hid one simple guard, the epsilon in the denominator.")


# === SECTION 9 - VISUALIZE ==================================================
# The KL term pulls the encoder's Gaussian (posterior) toward the standard-normal prior.
feat = X @ W_enc[:, 0]                                 # one latent coordinate over the data
m, s = feat.mean(), feat.std()
grid = np.linspace(-4, 4, 400)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(feat, bins=24, density=True, color=P["blue"], alpha=0.35, label="latent values (data)")
ax.plot(grid, gaussian_pdf(grid, m, s), color=P["yellow"], lw=3, label=f"posterior q  N({m:.2f}, {s:.2f}^2)")
ax.plot(grid, gaussian_pdf(grid, 0.0, 1.0), color=P["green"], lw=3, ls="--", label="prior  N(0, 1)")
ax.annotate("KL pulls q toward the prior", xy=(m, gaussian_pdf(m, m, s)), xytext=(1.4, 0.5),
            color=P["red"], arrowprops=dict(color=P["red"], arrowstyle="->"))
ax.set_xlabel("latent z (one dimension)")
ax.set_ylabel("density")
ax.set_title("Read Any ML Paper 0.2c - the ELBO's KL term, on iris")
ax.legend(loc="upper right")
fig.tight_layout()
plot_path = OUT / "lab_0.2c_elbo.png"
fig.savefig(plot_path, dpi=140)
print(f"\n[viz]      saved plot -> {plot_path}")


# === SECTION 10 - REAL-WORLD USE: ANOMALY DETECTION =========================
# A direct payoff of the Gaussian log-likelihood we built: a "surprise score".
# Fit a Gaussian to the data, then score each sample by its NEGATIVE log-likelihood.
# Points the model finds unlikely score HIGH -> likely anomalies.
#
# REAL-WORLD BENEFIT: this exact idea is how Gaussian / VAE models flag credit-card
# FRAUD, factory DEFECTS, network intrusions, and abnormal MEDICAL scans. You learn
# what "normal" looks like once, then ask of every new point: is this surprising?
# No labels of the rare event are needed - you only model the normal data.
mu_d, sd_d = X.mean(axis=0), X.std(axis=0)
surprise = -gaussian_log_likelihood(X, mu_d, sd_d)          # NLL per flower = surprise
outlier = np.array([[6.0, 6.0, -6.0, 6.0]])                 # a clearly off measurement
outlier_surprise = -gaussian_log_likelihood(outlier, mu_d, sd_d)[0]
thresh = surprise.mean() + 3 * surprise.std()              # simple 3-sigma flag
print(f"\n[use-case] typical flower surprise   ~ {np.median(surprise):.2f}")
print(f"[use-case] injected outlier surprise = {outlier_surprise:.2f}  "
      f"-> {'FLAGGED' if outlier_surprise > thresh else 'missed'} (threshold {thresh:.2f})")
print("[use-case] benefit: one Gaussian log-likelihood = a fraud / defect / health-screen detector.")

fig2, ax2 = plt.subplots(figsize=(8, 4.0))
ax2.hist(surprise, bins=24, color=P["blue"], alpha=0.55, label="iris flowers (normal)")
ax2.axvline(outlier_surprise, color=P["red"], lw=3, label="injected outlier (anomaly)")
ax2.axvline(thresh, color=P["yellow"], lw=2, ls="--", label="3-sigma threshold")
ax2.set_xlabel("surprise score = negative log-likelihood")
ax2.set_ylabel("count")
ax2.set_title("Read Any ML Paper 0.2c - anomaly detection with the Gaussian we built")
ax2.legend(loc="upper right")
fig2.tight_layout()
anom_path = OUT / "lab_0.2c_anomaly.png"
fig2.savefig(anom_path, dpi=140)
print(f"[use-case] saved plot -> {anom_path}")


if __name__ == "__main__":
    print("\nLab 0.2c complete: built the Gaussian, LayerNorm, and KL by hand, "
          "assembled the ELBO on iris, validated it, broke it, fixed it, "
          "and used it for anomaly detection.")
