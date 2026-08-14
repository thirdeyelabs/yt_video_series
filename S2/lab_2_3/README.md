# Lab 2.3 — The Learning Rate, from scratch

Companion notebook for **ThirdEye Labs 2.3**.

A real neural network — **64 → 32 → 10, 2,410 weights, written by hand in numpy** — learning
to read **1,400 real handwritten digits**. Then one controlled experiment: same network, same
data, same seed, change **one number**.

| learning rate | accuracy | verdict |
|---|---|---|
| 0.5 | 12.6% → **97.2%** | learns |
| 0.002 | 12.6% → **28.2%** | crawls, still useless after 300 steps |
| 60 | 12.6% → **8.3%** | **worse than guessing (10%)** |

It also settles two things people repeat and rarely test:

- **Your classifier's loss cannot reach `NaN`.** Pushed to `lr = 200,000` it just pins at
  `−log(1e-12) ≈ 27.6` and accuracy collapses to chance.
- **Where it really breaks** is an *unbounded* loss, which overflows to **`inf`** (step 457
  here) — not `NaN`. Real `NaN` comes from arithmetic *on* infinities.

Plus **epoch vs iteration vs batch**, counted out: `1,400 ÷ 64 = 22 iterations = 1 epoch`.

```
pip install numpy scikit-learn matplotlib
python s2_3_learning_rate.py
```

Deterministic (fixed seed), no downloads, runs in a few seconds. Same seeds as the episode,
so you get the same figures it shows.
