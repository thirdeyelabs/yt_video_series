# Lab AA.1 — Build an AI Agent From Scratch (the architecture)

Companion to the video **"What an AI Agent Actually Is (the Architecture)."** Build every
mechanism from the episode — the stateless call, `messages[]` as RAM, the `stop_reason` loop,
`run_tools()` with results matched by id, the unhappy path, parallel fan-out, a sub-agent as a
tool, compaction, and CPU-swapping.

**Runs with no API key.** `pip install anthropic`, then `python aa1_agent_architecture.py`. With no
`ANTHROPIC_API_KEY` it uses a deterministic mock model (same response *shape* as the SDK); set the
key to flip the identical agent code to the real `claude-opus-4-8`. That the code doesn't change is
the whole point: the model is a swappable part.
