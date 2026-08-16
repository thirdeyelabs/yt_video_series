# The Paper That Built ChatGPT — *Attention Is All You Need*, animated

Companion notebooks for the 5-part series on Vaswani et al. (NeurIPS 2017).
Each part adds one piece. By part 5 this folder is **a working transformer in pure numpy that
generates text — no PyTorch.**

| Lab | Part | Adds |
|---|---|---|
| [`lab_att_1/`](lab_att_1/) | 1 — Attention | scaled dot-product attention, **+ verified inside a real model** |
| `lab_att_2/` | 2 — Scaling + multi-head | `/ sqrt(d_k)` and multi-head |
| `lab_att_3/` | 3 — Position | sinusoidal positional encoding |
| `lab_att_4/` | 4 — The block | residual, LayerNorm, FFN |
| `lab_att_5/` | 5 — Generation | causal mask + the generation loop |

```bash
pip install numpy                      # sections 1-7, the whole numpy core
pip install torch transformers         # sections 8-11, the real-model checks
python PAPERS/attention/lab_att_1/paper_attention_1.py
```

**The one sentence:** attention is a weighted average of values, and the weights are dot products.

### Lab 1 also checks the video's claims against a real model

Sections 8-11 open `distilbert-base-uncased` (66,362,880 weights) and verify, rather than assert:

- **The head is chosen by a stated rule** — most attention on `animal` when reading `it` — which
  lands on layer 4, head 0 by itself. 10 of its 72 heads rank `animal` first, so this is *a* head
  that resolves the pronoun, not "how the model works".
- **0.6080 is rebuilt by hand** from `q · k`, a divide by `sqrt(64)`, and a softmax — agreeing with
  the library to `4.53e-07`.
- **The famous flip does not happen.** Change `tired` to `wide` and street's share rises 4.53x, but
  animal still wins, 0.2584 to 0.0937. The head's biggest weight is the adjective itself, 0.3206.
  Only 8 of 72 heads flip outright.
- **A decoder cannot run this test at all.** GPT-2's attention from `it` is byte-identical across
  the two sentences — biggest difference exactly `0.00e+00` — because when it reads `it`, the words
  `tired` and `wide` have not been written yet.

The full stop matters: drop it and the same head gives 0.4554 instead of 0.6080.

*Note: the sentence "The animal didn't cross the street because it was too tired" is a well-known
teaching example from the community's illustrated explanations, not from the paper. The toy
embeddings are hand-designed so you can see why the answer comes out as it does.*
