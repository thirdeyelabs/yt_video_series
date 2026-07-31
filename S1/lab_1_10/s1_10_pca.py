"""
LAB 1.10 — PCA: How AI Sees a Thousand Dimensions   ⭐ SEASON 1 FINALE
=====================================================================
Companion notebook to "PCA: How AI Sees a Thousand Dimensions".

pip install numpy scikit-learn        # sklearn only for the datasets

Every number shown in the video is reproduced and asserted here, including the honest limit
most PCA tutorials skip: PCA maximises SPREAD, not class separation, and on iris two raw
features beat it. This is also where S1 closes — PCA is eigenvectors (1.8) + SVD (1.9) +
change of basis (1.7b), turned on DATA instead of one matrix.
"""

# === SECTION 1 - REAL DATA, UNREADABLE (the wall of numbers) ===
# 🎭 Analogy: photograph a chair straight down the back and it is a stick. The object never
# changed - you just found the angle that showed the most.
import numpy as np
from sklearn.datasets import load_iris

d = load_iris()
X, y = d.data, d.target
print("iris:", X.shape, "= 150 flowers, 4 measurements each")
print("first 4 rows:")
for r in X[:4]:
    print("   ", "  ".join(f"{v:.1f}" for v in r))
print("three species hide in here; the human eye cannot see them in 4 columns")
assert X.shape == (150, 4)


# === SECTION 2 - STEP ONE: SLIDE IT TO THE MIDDLE (centre the data) ===
means = X.mean(0)
Xc = X - means
print(f"\nsubtract the average of each column: {np.round(means, 2)}")
assert np.allclose(Xc.mean(0), 0)          # the cloud is now centred on the origin


# === SECTION 3 - THE DIRECTION OF MOST SPREAD (a principal component) ===
# In plain words: try every direction, measure how spread out the dots are along it, keep the
# winner. That maximiser IS the top eigenvector of the covariance - no search needed.
C = Xc.T @ Xc / len(Xc)                    # how the columns move together
vals, vecs = np.linalg.eigh(C)
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]

def spread_along(direction):
    return np.std(Xc @ (direction / np.linalg.norm(direction)))

best = spread_along(vecs[:, 0])
rng = np.random.default_rng(0)
random_dirs = [spread_along(rng.standard_normal(4)) for _ in range(5000)]
print(f"\nspread along PC1            = {best:.4f}")
print(f"best of 5000 random directions = {max(random_dirs):.4f}")
assert best >= max(random_dirs)            # nothing beats the top eigenvector


# === SECTION 4 - HOW MUCH DID WE KEEP? (4 numbers -> 2) ===
ev = vals / vals.sum() * 100
print(f"\nvariance explained: {np.round(ev, 1)}")
print(f"top two together:   {ev[:2].sum():.1f}%")
assert np.isclose(ev[0], 92.46, atol=0.1)
assert np.isclose(ev[:2].sum(), 97.77, atol=0.1)

P = Xc @ vecs[:, :2]                       # the 2D picture
for i, name in enumerate(d.target_names):
    print(f"  {name:11} centre = ({P[y==i][:,0].mean():+.2f}, {P[y==i][:,1].mean():+.2f})")


# === SECTION 5 - IT IS EIGENVECTORS (1.8) + SVD (1.9) + CHANGE OF BASIS (1.7b) ===
# The season closes here: the SVD of the centred data gives the SAME directions and the same
# variances. PCA is not a new idea - it is three old ones pointed at data.
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
assert np.allclose(vals, s**2 / len(Xc))                    # eigenvalues == s^2 / n
assert np.allclose(np.abs(Vt[:2]), np.abs(vecs[:, :2].T))   # same directions
print("\nSVD of the centred data gives the SAME answer:")
print(f"  eig(C) values  = {np.round(vals, 4)}")
print(f"  s^2 / n        = {np.round(s**2/len(Xc), 4)}   -> identical")


# === SECTION 6 - NOW WITH SIXTY-FOUR (earning "a thousand dimensions") ===
# 🎭 Analogy: a pancake is a 3D object that is almost flat - press it down and you lose the
# thickness, not the pancake. High-dimensional data is nearly always a pancake in disguise.
from sklearn.datasets import load_digits

dg = load_digits()
Xd, yd = dg.data, dg.target
print(f"\ndigits: {Xd.shape}  = 8x8 images -> {Xd.shape[1]} numbers per digit")
Xdc = Xd - Xd.mean(0)
dvals, dvecs = np.linalg.eigh(Xdc.T @ Xdc / len(Xdc))
o = np.argsort(-dvals); dvals, dvecs = dvals[o], dvecs[:, o]
dev = dvals / dvals.sum() * 100
Pd = Xdc @ dvecs[:, :2]
print(f"top-2 variance explained: {dev[0]:.1f}% + {dev[1]:.1f}% = {dev[:2].sum():.1f}%")
assert np.isclose(dev[:2].sum(), 28.5, atol=0.5)

# do the digits still separate on only 28.5%?
centres = np.array([Pd[yd == i].mean(0) for i in range(10)])
spread = np.mean([np.linalg.norm(Pd[yd == i] - centres[i], axis=1).mean() for i in range(10)])
sep = np.mean([np.linalg.norm(centres[i] - centres[j])
               for i in range(10) for j in range(i + 1, 10)])
print(f"separation / spread = {sep/spread:.2f}   (> 1 means the clusters are distinguishable)")
assert sep / spread > 2.0
print("=> you can throw away 71% of the variance and the structure survives")


# === SECTION 7 - ⚠️ THE HONEST LIMIT (what most tutorials skip) ===
# PCA maximises SPREAD. It never sees the labels. Spread and usefulness are NOT the same.
import itertools

def separation(P2, labels, k):
    cs = [P2[labels == i].mean(0) for i in range(k)]
    sp = np.mean([np.linalg.norm(P2[labels == i] - cs[i], axis=1).mean() for i in range(k)])
    se = np.mean([np.linalg.norm(cs[i] - cs[j]) for i in range(k) for j in range(i + 1, k)])
    return se / sp

pca_score = separation(P, y, 3)
print(f"\nPCA's own 2D view:              {pca_score:.2f}")
best_raw, best_pair = -1, None
for a, b in itertools.combinations(range(4), 2):
    sc = separation(Xc[:, [a, b]], y, 3)
    if sc > best_raw:
        best_raw, best_pair = sc, (a, b)
print(f"best raw feature pair ({d.feature_names[best_pair[0]][:12].strip()} + "
      f"{d.feature_names[best_pair[1]][:12].strip()}): {best_raw:.2f}")
assert best_raw > pca_score          # <- two raw features BEAT PCA on this dataset
print("PCA is NOT told which flower is which - it optimises spread, not separation.")
print("Use PCA to SEE your data. Do not use it to decide what is important.")

print("\nLAB COMPLETE - and that closes Season 1.")
print("PCA = the directions of most spread (1.8) + a change of basis (1.7b) + the singular")
print("values (1.9), pointed at data instead of one matrix.")
