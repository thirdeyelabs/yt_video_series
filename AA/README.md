# AA — AI Agents, Animated (companion labs)

Hands-on notebooks for the **AI Agents, Animated** series. Each lab runs top-to-bottom with **no API
key** (a deterministic mock model with the real SDK's response shape); set `ANTHROPIC_API_KEY` to run
the identical code against the real model.

| Lab | Video | What you build |
|---|---|---|
| [`aa1_agent_architecture/`](aa1_agent_architecture/) | What an AI Agent Actually Is (the Architecture) | The agent from scratch: stateless call, `messages[]` as RAM, the `stop_reason` loop, `run_tools()` matched by id, the unhappy path, parallel fan-out, a sub-agent as a tool, compaction, CPU-swapping |
| [`aa2_syscall_table/`](aa2_syscall_table/) | Tools: the Orchestrator's Syscall Table | Tools done properly with **Pydantic**: one model per tool → schema + `strict` + `model_validate` boundary + typed handlers, error shaping, permission tiers by reversibility, a parallel dispatcher |

```bash
pip install anthropic pydantic
python AA/aa1_agent_architecture/aa1_agent_architecture.py
python AA/aa2_syscall_table/aa2_syscall_table.py
```
