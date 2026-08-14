# 2.2 — The Gradient
*Study notes · ThirdEye Labs · Season 2, episode 2*

---

## The one sentence

> Ask 2.1's question **once per knob**, stack the answers into **one arrow** — and that
> arrow points straight uphill.

---

## The setup: two dials on a real photo

```
A = a washed-out photograph (97,200 numbers: 240 × 135 × 3 colours)
y = the correct photograph
w₀ = contrast (multiply)      w₁ = brightness (add)

loss(w) = mean( (w₀·A + w₁ − y)² )
```

Start at **w = (1.0, 0.0)** — the photo untouched.
Score **0.02181** → every pixel off by about **38 shades out of 255**.

---

## Wiggle each dial separately

```
turn CONTRAST a hair    →  −0.11762
turn BRIGHTNESS a hair  →  −0.14436
                           ─────────
stack them:  g = [−0.11762, −0.14436]      ← THE GRADIENT
length |g| = 0.18621                        ← how steep it is here
```

Three methods agree: finite differences, the exact formula, autograd.

---

## Build −0.11762 one real pixel at a time

| pixel | is | should be | off | 2 × A × miss | running avg |
|---|---|---|---|---|---|
| 1 | 0.700 | 0.922 | −0.221 | −0.3101 | −0.3101 |
| 2 | 0.634 | 0.780 | −0.147 | −0.1860 | −0.2480 |
| 3 | 0.730 | 0.973 | −0.243 | −0.3545 | −0.2835 |
| 4 | 0.639 | 0.808 | −0.168 | −0.2154 | −0.2665 |

…all 97,200 numbers average to **−0.11762**

---

## The race — and why nothing can win

**5,000 random directions vs the gradient.**
Instantaneously: **0 of 5,000** beat it.

Why none *can*:

$$g \cdot \hat{d} = |g|\cos\theta$$

| angle off | score |
|---|---|
| 0° | +0.18621 ← the maximum |
| 30° | +0.16127 |
| 60° | +0.09311 |
| 90° | 0 |

Cosine never exceeds 1, and equals 1 in exactly one direction. **That's the proof.**

> ⚠️ Over a *finite step* a few random directions can win — the surface curves away.
> Steepest means steepest **instantaneously**. That gap is episode 2.3.

---

## Perpendicular to the contour

```
g · tangent = 0.00e+00        angle = 90.0000°
```

Exactly, not approximately. Walk along a contour and the score doesn't change, so the
direction of fastest change must be at a right angle to it.

---

## Nobody computes the real one

| pixels used | angle off | still downhill | cheaper |
|---|---|---|---|
| 1 | 11.5° | 65.7% | 97,200× |
| 10 | 3.8° | 97.3% | 9,720× |
| **100** | **1.1°** | **100%** | **972×** |
| 1000 | 0.4° | 100% | 97× |

**Throw away 99.9% of your data and you still know which way to go.** That is what
*stochastic* / *mini-batch* / *batch* mean.

---

## What one gradient costs

```
GPT-2:  124,439,808 numbers × 4 bytes  =  498 MB        per step
        × 300,000 steps                =  ~149 TB       computed and thrown away
```

---

## The symbol, last

$$\theta \leftarrow \theta - \eta \nabla L$$

∇L = the arrow. It points **uphill** — use its negative to go down.

**Next:** you know which way. How far? → **2.3, the learning rate.**
