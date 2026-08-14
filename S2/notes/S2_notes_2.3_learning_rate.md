# 2.3 — The Learning Rate
*Study notes · ThirdEye Labs · Season 2, episode 3*

---

## The one sentence

> The gradient says **WHICH WAY**. The learning rate says **HOW FAR**.
> That second number decides whether the model learns, barely learns, or destroys itself.

---

## The loop (this is all training is)

```
     ┌──────────────────────────┐
     │  measure the slope       │
     │  step against it         │
     │  repeat                  │
     └──────────────────────────┘
```

## The update, on one real weight

```
W1[7,11]  =  −0.2089
gradient  =  −0.0057
lr        =   0.5

−0.2089  −  0.5 × (−0.0057)  =  −0.2061
```

2,410 of these, every single step.

$$\theta \leftarrow \theta - \eta \nabla L$$

θ = the weight · η = the learning rate (HOW FAR) · ∇L = the arrow (WHICH WAY)

---

## The experiment — one variable

Real network **64 → 32 → 10** (2,410 weights), **1,400 real handwritten digits**,
300 steps, same seed. Only η changes.

| η | loss | accuracy | verdict |
|---|---|---|---|
| **0.5** | 2.3162 → **0.0718** | 12.6% → **97.2%** | learns |
| 0.002 | 2.3162 → 2.1622 | 12.6% → **28.2%** | crawls — still useless |
| 60 | pinned at ceiling | 12.6% → **8.3%** | **worse than guessing (10%)** |

⚠️ **"A smaller learning rate is safer" is the myth.** It is how you wait all day and
arrive nowhere.

---

## Watch it actually learn

A real test digit — one the model never trained on:

```
a real 9:  called  3 → 7 → 3 → 9 → 9      (99% confident)
a real 1:  called  7 → 2 → 1 → 1 → 1      (60% confident)
```

It learned. **It did not become infallible** — and knowing the difference is most of
what using these models well means.

---

## Epoch vs iteration vs batch

```
1,400 examples  ÷  batch 64   =  22 iterations  =  1 EPOCH
300 steps                     =  13.6 epochs
10 epochs                     =  220 updates
```

An epoch is **not** a unit of time. It is one pass over the data.

> 2.2 asked how **wrong** a small batch makes your arrow (direction).
> 2.3 asks how **many** steps it buys you (counting).

---

## ⚠️ Your loss will never reach NaN (if it's a classifier)

Tested to **lr = 200,000**:

| lr | peak loss | NaN? |
|---|---|---|
| 60 | 25.90 | **No** |
| 2,000 | 27.63 | **No** |
| 200,000 | 27.63 | **No** |

A numerically stable softmax bounds cross-entropy at **−log(1e-12) ≈ 27.6**. The loss
pins at that ceiling and accuracy collapses to chance. It cannot NaN.

**Where it really breaks:** a loss with *no ceiling* (squared error). Each step multiplies
the error by ~29×:

```
0.0218 → 0.204 → 5.63 → 165 → 4,856 → 142,700 → …  →  inf   at step 209
```

⚠️ It becomes **`inf`**, not `NaN`. Real `NaN` comes from arithmetic *on* infinities
(`inf − inf`, `0 × inf`, `log 0`) — one step further down the same road.

---

## The boundary is exact

For that surface the largest stable step is **0.701**.
Below it every run converges. Above it every run explodes. Not usually — always.
It comes from the shape of the surface itself, which is why nobody can hand you one
learning rate that works for every model.

---

## So: `lr = 3e-4`

```python
optimizer = Adam(model.parameters(), lr=3e-4)
```

Three ten-thousandths of the arrow, every step. Small enough not to explode, big enough
to actually move — and **nobody derived it.** Somebody ran the experiment above on a
bigger model and that number worked.

**Next:** get the step size perfect and the front of a deep network can *still* learn
nothing → **2.4, vanishing gradients.**
