# 2.1 — Sensitivity (The Derivative)
*Study notes · ThirdEye Labs · Season 2, episode 1*

---

## The one question

> **"If I nudge this number, does the score get better or worse — and by how much?"**

Asked once per weight, per step, for the entire history of machine learning.

---

## The setup

```
one weight  w
64 real data points
score(w) = mean( (w·x − y)² )        ← how wrong we are
```

Start at **w = 2.0**, score **0.5062**

---

## Nudge it and divide

```
h = 0.1
score(2.1) − score(2.0)  =  −0.02254
                  ÷ 0.1  =  −0.2254     ← the ratio
```

Now shrink the nudge and watch the digits stop changing:

| h | ratio |
|---|---|
| 0.1 | −0.22540 |
| 0.01 | −0.24754 |
| 0.001 | −0.24975 |
| 0.0001 | −0.24998 |
| 0.00001 | −0.25000 |

**It settles on −0.25.** That settling number *is* the derivative.
This is a limit actually happening — not a definition, a measurement.

---

## Check it

| method | value |
|---|---|
| shrinking nudge | −0.25000 |
| autograd (PyTorch) | −0.2500000 |

Agreement to 6 decimals. The number is real.

---

## What the sign means

- ratio **negative** → score falls as w rises → **increase w**
- ratio **positive** → score rises as w rises → **decrease w**

The sign is the instruction. Walking downhill from w = 2.0 lands at **w = 2.9993**
(the data was really made with w = 3.0).

---

## The symbol, last

$$\frac{dL}{dw}$$

⚠️ **Not a fraction.** It is the *name* of the number you just watched settle.
If the notation means nothing to you, ignore it — you already have the idea.

---

## Where this lives in an LLM

Every weight in a transformer gets this same question asked about it, on every
training step. A 124-million-weight model asks it 124 million times per step.

**Next:** with one weight the answer is a *sign*. With two it has to become a
*direction* → **2.2, the gradient.**
