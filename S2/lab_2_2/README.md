# Lab 2.2 — The Gradient, from scratch, on a real photograph

Companion notebook for **ThirdEye Labs 2.2 — "How AI Finds Downhill in a Billion Dimensions."**

A washed-out photo, two dials — **contrast** and **brightness** — and one score saying how far
the result is from the correct picture. That is ordinary two-parameter least squares, the same
maths as fitting a line, except you can watch the answer arrive as an image.

Every number the episode shows is rebuilt here so you can check it rather than trust it:

- the gradient `[-0.11762, -0.14435]` built **one real pixel at a time**, then verified three
  ways (finite differences, the exact formula, autograd)
- the race — **0 of 5,000** random directions beat it instantaneously, and the handful that beat
  it *over a real step*, which is episode 2.3's problem rather than a hole in the claim
- why none of them **can** win: every direction scores `|g| × cos(angle)`
- exactly **90.0000°** to the contour, dot product `0.00e+00`
- mini-batch on pixels: **100 numbers out of 97,200** give an arrow 1.1° off that still points
  downhill every time — throw away 99.9% of your data and you still know which way to go
- descent recovering **contrast 1.799, brightness −0.349** against the true 1.8 / −0.35, taking
  the photo from ~38 shades off to ~2

```
pip install numpy torch opencv-python matplotlib
python s2_2_gradient.py
```

Deterministic (fixed seed), no downloads, runs in a few seconds. Saves a
before / fixed / target strip so you can see the result.

**Image credit:** `frog.jpg` — public domain (CC0), Wikimedia Commons, *"Portrait of a Bull Frog,
mid summer, Cedar Mill Wetlands.jpg"*. Shipped here so the notebook reproduces the episode's
numbers exactly; the script resizes it to 240px wide, and a different source file would give
slightly different pixels and therefore slightly different numbers.
