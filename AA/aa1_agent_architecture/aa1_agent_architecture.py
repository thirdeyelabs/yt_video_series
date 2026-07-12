"""
LAB AA.1 — Build an AI Agent From Scratch (the architecture, hands-on)
======================================================================
Companion notebook to "What an AI Agent Actually Is (the Architecture)".

pip install anthropic        # the only dependency (pydantic comes with it)

You will build every mechanism from the episode, in order:
the stateless call, the messages[] RAM, the loop driven by stop_reason, run_tools()
with results matched BY ID, the unhappy path, parallel fan-out, a sub-agent as a tool,
and compaction when RAM fills.

DUAL MODE (so the notebook always runs):
  - No ANTHROPIC_API_KEY set  -> a deterministic MOCK model (same response SHAPE as the
    real SDK, scripted turns). Every mechanism is identical; only the intelligence is canned.
  - ANTHROPIC_API_KEY set     -> flips to the real Anthropic SDK (claude-opus-4-8).
The agent code below does not change between modes. That is the whole point of the
architecture: the model is a swappable part.
"""

# === SECTION 1 - THE MODEL IS A STATELESS FUNCTION (mock or real) ===
# 🎭 Analogy: the amnesiac consultant - brilliant, but you hand over the WHOLE case
# file every single morning; they dictate one memo and forget you exist.
import json
import os
from types import SimpleNamespace as NS

USE_REAL = bool(os.environ.get("ANTHROPIC_API_KEY"))


def block(**kw):
    """A content block with the same attribute shape the real SDK returns."""
    return NS(**kw)


class MockModel:
    """Deterministic stand-in for client.messages.create.

    Same response SHAPE as the SDK (.content = list of blocks, .stop_reason), driven by
    a script of turns. The agent loop cannot tell the difference - which proves the
    model is a swappable CPU."""

    def __init__(self, script):
        self.script = list(script)

    def create(self, *, model, messages, tools=(), max_tokens=1024, **kw):
        assert isinstance(messages, list) and messages, "messages[] is the RAM - never empty"
        turn = self.script.pop(0)
        return NS(content=turn["content"], stop_reason=turn["stop_reason"])


if USE_REAL:
    import anthropic
    _client = anthropic.Anthropic()
    def call_model(messages, tools, script=None):
        return _client.messages.create(model="claude-opus-4-8", max_tokens=1024,
                                       tools=tools, messages=messages)
else:
    def call_model(messages, tools, script):
        return script.create(model="mock", messages=messages, tools=tools)

print(f"mode: {'REAL (claude-opus-4-8)' if USE_REAL else 'MOCK (deterministic, no key needed)'}")


# === SECTION 2 - messages[] IS RAM (append-only state) ===
# The model holds NOTHING between calls. All state lives in this one list, and the
# orchestrator re-transmits ALL of it every call. Append-only is not a style choice:
# editing history invalidates the provider's prompt cache (you re-pay for every token).
messages = [{"role": "user", "content": "What's the weather in Paris? Then say what to pack."}]

def append_assistant(resp):
    messages.append({"role": "assistant", "content": list(resp.content)})

def append_results(results):
    messages.append({"role": "user", "content": results})

assert messages[0]["role"] == "user"
print(f"RAM initialized: {len(messages)} message(s)")


# === SECTION 3 - TOOLS: THE TABLE THE MODEL CAN ONLY ASK FOR ===
# The model cannot touch the network. It can only fill in a typed form addressed at
# this table; run_tools() is the ONLY code that causes side effects (the air-gap).
TOOLS = [{
    "name": "get_weather",
    "description": "Get current weather. Call when the user asks about weather, "
                   "packing, or what to wear.",
    "input_schema": {"type": "object",
                     "properties": {"city": {"type": "string"}},
                     "required": ["city"], "additionalProperties": False},
}]

def weather_api(city):                       # a stand-in external API (deterministic)
    data = {"Paris": "18C, light rain", "Tokyo": "24C, sunny"}
    if city not in data:
        raise TimeoutError("weather API timed out")
    return data[city]

def run_tools(content):
    """Execute every tool_use block LOCALLY; return tool_results matched BY ID.
    Exceptions never escape: the unhappy path is wrapped as text for the model."""
    results = []
    for b in content:
        if getattr(b, "type", None) != "tool_use":
            continue
        try:
            out = {"get_weather": lambda i: weather_api(i["city"])}[b.name](b.input)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)})
        except Exception as e:               # ← the orchestrator catches, the model just reads
            results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                            "content": f"{e}. Retry once, or ask the user for another city."})
    return results

print(f"table: {[t['name'] for t in TOOLS]}")


# === SECTION 4 - THE LOOP (stop_reason is the state machine) ===
# ACTING while the model returns tool_use; RESOLVED on end_turn. That's the whole
# control flow of every agent framework you have ever seen.
script = MockModel([
    {"stop_reason": "tool_use",
     "content": [block(type="tool_use", id="t1", name="get_weather",
                       input={"city": "Paris"})]},
    {"stop_reason": "end_turn",
     "content": [block(type="text",
                       text="18C and rainy in Paris - pack a light rain jacket.")]},
])

laps = 0
while True:                                                   # ← THE AGENT LOOP
    resp = call_model(messages, TOOLS, script)
    append_assistant(resp)
    if resp.stop_reason == "end_turn":                        # -> RESOLVED
        break
    append_results(run_tools(resp.content))                   # act, observe -> ACTING again
    laps += 1

final = next(b.text for b in messages[-1]["content"] if getattr(b, "type", None) == "text")
print(f"RESOLVED after {laps} tool lap(s): {final}")
assert laps >= 1 and "rain" in final.lower()
assert len(messages) == 4          # user, assistant(tool_use), user(result), assistant(answer)


# === SECTION 5 - THE UNHAPPY PATH (errors are just more state) ===
# A tool failure never crashes the loop and never reaches the model as an exception -
# it becomes a tool_result the model READS. Shape it like an instruction, not a stack trace.
bad = run_tools([block(type="tool_use", id="t9", name="get_weather",
                       input={"city": "Atlantis"})])
print("shaped error ->", bad[0]["content"])
assert bad[0]["is_error"] and "Retry" in bad[0]["content"]


# === SECTION 6 - PARALLEL TOOL CALLS (fan out, match by id) ===
# One model turn can request MANY side effects. Results may finish in ANY order -
# you match them by tool_use_id, never by position.
parallel_turn = [
    block(type="tool_use", id="a", name="get_weather", input={"city": "Paris"}),
    block(type="tool_use", id="b", name="get_weather", input={"city": "Tokyo"}),
]
results = run_tools(parallel_turn)
by_id = {r["tool_use_id"]: r["content"] for r in results}
print(f"fan-out: {len(results)} results  ->  a: {by_id['a']}  |  b: {by_id['b']}")
assert set(by_id) == {"a", "b"} and "24C" in by_id["b"]


# === SECTION 7 - A TOOL CAN BE ANOTHER AGENT (fork / join) ===
# 🎭 Analogy: the general contractor never watches the electrician pull wires - the
# whole sub-job comes back as ONE invoice line. A child agent runs its own loop in its
# own FRESH messages[] (isolated RAM) and returns only a compact summary to the parent.
def research_agent(question):
    child_messages = [{"role": "user", "content": question}]   # child RAM: EMPTY, isolated
    child_script = MockModel([
        {"stop_reason": "tool_use",
         "content": [block(type="tool_use", id="c1", name="get_weather",
                           input={"city": "Tokyo"})]},
        {"stop_reason": "end_turn",
         "content": [block(type="text", text="Tokyo: 24C sunny; pack light layers.")]},
    ])
    while True:                                                # the SAME loop, recursively
        r = call_model(child_messages, TOOLS, child_script)
        child_messages.append({"role": "assistant", "content": list(r.content)})
        if r.stop_reason == "end_turn":
            break
        child_messages.append({"role": "user", "content": run_tools(r.content)})
    summary = next(b.text for b in r.content if b.type == "text")
    return summary                            # JOIN: 1 line back; child context is discarded

note = research_agent("What should I pack for Tokyo?")
print(f"child agent (own RAM, {0} tokens leaked to parent) -> {note}")
assert "Tokyo" in note and len(messages) == 4     # parent RAM untouched by the child's laps


# === SECTION 8 - WHEN RAM OVERFLOWS: COMPACTION (lossy paging) ===
# The context window is finite. Long-running agents summarize old turns into one
# compact chip (lossy!) and keep durable details in files (disk). This mini version
# shows the mechanic; production uses a model call to write the summary.
def compact(msgs, keep_last=2):
    if len(msgs) <= keep_last + 1:
        return msgs
    summary = f"[summary of {len(msgs) - keep_last} earlier messages: task + tool results]"
    return [{"role": "user", "content": summary}] + msgs[-keep_last:]

long_history = messages + [{"role": "user", "content": f"turn {i}"} for i in range(6)]
compacted = compact(long_history)
print(f"compaction: {len(long_history)} messages -> {len(compacted)} "
      f"(head is now: {compacted[0]['content'][:40]}...)")
assert len(compacted) == 3


# === SECTION 9 - SWAP THE CPU (the proof of statelessness) ===
# Because ALL state is in messages[], you can change the model between two turns and
# the conversation continues perfectly. Real orchestrators route on this: a cheap
# model for tool laps, the big model for the final answer.
def pick_model(resp_will_be_final):
    return "claude-opus-4-8" if resp_will_be_final else "claude-haiku-4-5"

print("routing:", pick_model(False), "for tool laps ->", pick_model(True), "for the answer")
print("\nLAB COMPLETE - every mechanism from the episode, in",
      "REAL" if USE_REAL else "MOCK", "mode.")
