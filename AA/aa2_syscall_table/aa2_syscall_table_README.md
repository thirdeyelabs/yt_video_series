# Lab AA.2 — Tools Done Properly: Pydantic at the Boundary

Companion to the video **"Tools: the Orchestrator's Syscall Table."** Write each tool once as a
**Pydantic model + a function**, and derive everything from it: the API `input_schema`, the
description the model reads, server-side `strict` rejection, client-side `model_validate` at your
boundary, and typed handler arguments. Plus shaped errors, permission tiers by **reversibility**,
a parallel dispatcher matched by id, and the real agent loop.

**Runs with no API key.** `pip install anthropic pydantic`, then `python aa2_syscall_table.py`. The
boundary demo (valid / wrong-type / extra-field / gated-tool) runs fully offline; set
`ANTHROPIC_API_KEY` to also run the live loop against `claude-opus-4-8`.
