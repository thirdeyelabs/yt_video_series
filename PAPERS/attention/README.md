# The Paper That Built ChatGPT — *Attention Is All You Need*, animated

Companion notebooks for the 5-part series on Vaswani et al. (NeurIPS 2017).
Each part adds one piece. By part 5 this folder is **a working transformer in pure numpy that
generates text — no PyTorch.**

| Lab | Part | Adds |
|---|---|---|
| [`lab_att_1/`](lab_att_1/) | 1 — Attention | scaled dot-product attention (unscaled for now) |
| `lab_att_2/` | 2 — Scaling + multi-head | `/ sqrt(d_k)` and multi-head |
| `lab_att_3/` | 3 — Position | sinusoidal positional encoding |
| `lab_att_4/` | 4 — The block | residual, LayerNorm, FFN |
| `lab_att_5/` | 5 — Generation | causal mask + the generation loop |

```bash
pip install numpy
python PAPERS/attention/lab_att_1/paper_attention_1.py
```

**The one sentence:** attention is a weighted average of values, and the weights are dot products.

*Note: the sentence "The animal didn't cross the street because it was too tired" is a well-known
teaching example from the community's illustrated explanations, not from the paper. The toy
embeddings are hand-designed so you can see why the answer comes out as it does.*
