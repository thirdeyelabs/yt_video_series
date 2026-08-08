# Lab 2.1 — The Derivative, from scratch

Companion to **ThirdEye Labs 2.1 · The One Question AI Asks a Trillion Times**. One weight, 64
real points, and the number that tells you which way to turn it — including the moment it breaks:
at `h = 1e-16` the finite difference collapses to exactly `0.00000000`, because the subtraction
cancels every digit float64 has. That failure is why autograd exists.

```bash
pip install numpy torch
python s2_1_derivative.py          # or open s2_1_derivative.ipynb
```

Deterministic (fixed seed), no downloads, runs in under a second.
