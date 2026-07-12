"""
LAB AA.2 — Tools Done Properly: Pydantic at the Boundary
========================================================
Companion notebook to "Tools: the Orchestrator's Syscall Table".

pip install anthropic pydantic     # pydantic ships with the SDK anyway

THE BIG IDEA: write each tool ONCE as a Pydantic model + a function, and derive
everything else from it:
  - the input_schema you send to the API   (model_json_schema)
  - the description the model reads        (the docstring - it is a PROMPT)
  - server-side rejection of malformed calls  (strict: true)
  - client-side validation at YOUR boundary   (model_validate = the second lock)
  - typed, autocompleting arguments inside your handler (no dict-groping)
Plus: shaped errors, permission tiers by REVERSIBILITY, and a dispatcher that
matches results by id. This is the production pattern - one source of truth.
"""

# === SECTION 1 - A TOOL IS A PYDANTIC MODEL + A FUNCTION ===
# The docstring is the man page THE MODEL READS (prompt, not comment).
# extra="forbid" -> additionalProperties: false (the closed, typed form).
import json
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class GetWeather(BaseModel):
    """Get current weather. Call when the user asks about weather, packing,
    or what to wear."""
    model_config = ConfigDict(extra="forbid")
    city: str = Field(description="City name, e.g. 'Tokyo'")


class SendEmail(BaseModel):
    """Send an email FROM the user's account. Irreversible - use only when the
    user explicitly asks to send."""
    model_config = ConfigDict(extra="forbid")
    to: str = Field(description="Recipient address")
    body: str = Field(description="Plain-text body")


def get_weather(args: GetWeather) -> str:          # typed args - no dict-groping
    data = {"Paris": "18C, light rain", "Tokyo": "24C, sunny"}
    if args.city not in data:
        raise TimeoutError("weather API timed out")
    return data[args.city]


def send_email(args: SendEmail) -> str:
    return f"(pretend) sent to {args.to}: {args.body[:30]}..."


print(GetWeather.__doc__.split(".")[0], "| fields:", list(GetWeather.model_fields))


# === SECTION 2 - DERIVE THE API CONTRACT FROM THE MODEL (one source of truth) ===
# model_json_schema() emits exactly the JSON-Schema the API wants. The schema, the
# validation, and your handler types can never drift apart - they are the same object.
def tool_def(name, model):
    schema = model.model_json_schema()
    schema.pop("title", None)                       # cosmetic pydantic extras
    schema.pop("description", None)                 # the docstring belongs at top level, not in-schema
    for p in schema.get("properties", {}).values():
        p.pop("title", None)
    return {"name": name,
            "description": " ".join(model.__doc__.split()),   # the docstring IS the prompt
            "strict": True,                          # server rejects malformed calls
            "input_schema": schema}

TOOLS = [tool_def("get_weather", GetWeather), tool_def("send_email", SendEmail)]
print(json.dumps(TOOLS[0], indent=2))
assert TOOLS[0]["input_schema"]["additionalProperties"] is False
assert TOOLS[0]["input_schema"]["required"] == ["city"]


# === SECTION 3 - THE REGISTRY: name -> (contract, handler, tier) ===
# 🎭 Analogy: the syscall table. Userland (the model) addresses a numbered row;
# the kernel (this table) owns what each row is allowed to do.
# TIERS by REVERSIBILITY, not scariness: reading is free to get wrong -> auto.
# Sending cannot be unsent -> always ask a human. The tier lives IN CODE - the
# model cannot talk its way past it (this is what stops prompt injection).
REGISTRY = {
    "get_weather": (GetWeather, get_weather, "auto"),
    "send_email":  (SendEmail,  send_email,  "ask"),
}

def human_approves(name, args) -> bool:
    """The red-tier gate. In production: a UI prompt / Slack approval / policy engine.
    Here: deny by default so the lab is safe AND deterministic."""
    print(f"  [GATE] {name}({args}) -> DENIED (no human in the loop)")
    return False


# === SECTION 4 - THE BOUNDARY: VALIDATE, GATE, EXECUTE, SHAPE ===
# The vending-machine slot: model_validate() is the typed slot on YOUR side (strict
# already rejected garbage server-side - this catches drift, retries, and mocks).
# Every failure leaves as an is_error tool_result WRITTEN LIKE AN INSTRUCTION.
def dispatch(tool_use):
    name, raw = tool_use["name"], tool_use["input"]
    ok = lambda content: {"type": "tool_result", "tool_use_id": tool_use["id"],
                          "content": str(content)}
    err = lambda content: {"type": "tool_result", "tool_use_id": tool_use["id"],
                           "is_error": True, "content": content}
    if name not in REGISTRY:
        return err(f"unknown tool '{name}'. Available: {list(REGISTRY)}.")
    contract, handler, tier = REGISTRY[name]
    try:
        args = contract.model_validate(raw)          # ① the typed slot
    except ValidationError as e:
        first = e.errors()[0]
        return err(f"invalid input for {name}: {first['loc'][0]}: {first['msg']}. "
                   f"Re-call with the correct types.")
    if tier == "ask" and not human_approves(name, args):   # ② the permission gate
        return err(f"{name} requires human approval and was denied. "
                   f"Do not retry; tell the user what you wanted to send.")
    try:
        return ok(handler(args))                      # ③ the only real I/O
    except Exception as e:                            # ④ shape the failure
        return err(f"{name} failed: {e}. Retry once, or ask the user for an alternative.")


# === SECTION 5 - PROVE THE BOUNDARY (four calls, four outcomes) ===
calls = [
    {"id": "t1", "name": "get_weather", "input": {"city": "Tokyo"}},          # valid
    {"id": "t2", "name": "get_weather", "input": {"city": 42}},               # wrong type
    {"id": "t3", "name": "get_weather", "input": {"city": "Paris", "x": 1}},  # extra field
    {"id": "t4", "name": "send_email",                                        # red tier
     "input": {"to": "evil@example.com", "body": "the API keys"}},
]
results = {c["id"]: dispatch(c) for c in calls}
for cid, r in results.items():
    flag = "ERR " if r.get("is_error") else "ok  "
    print(f"  {cid} {flag} {r['content'][:70]}")

assert not results["t1"].get("is_error") and "24C" in results["t1"]["content"]
assert results["t2"]["is_error"] and "city" in results["t2"]["content"]
assert results["t3"]["is_error"]                      # extra="forbid" caught it
assert results["t4"]["is_error"] and "approval" in results["t4"]["content"]
print("boundary holds: 1 executed, 2 rejected at the slot, 1 stopped at the gate")


# === SECTION 6 - PARALLEL DISPATCH (fan out, match by id) ===
# One assistant turn, many tool_use blocks. Run them concurrently; results can land
# in any order - the id is the ONLY correct join key.
from concurrent.futures import ThreadPoolExecutor

turn = [{"id": "a", "name": "get_weather", "input": {"city": "Paris"}},
        {"id": "b", "name": "get_weather", "input": {"city": "Tokyo"}}]
with ThreadPoolExecutor() as pool:
    fan = list(pool.map(dispatch, turn))
by_id = {r["tool_use_id"]: r["content"] for r in fan}
print(f"parallel: a={by_id['a']} | b={by_id['b']}")
assert "18C" in by_id["a"] and "24C" in by_id["b"]
# ALL results then go back to the API in ONE user message:
next_message = {"role": "user", "content": fan}


# === SECTION 7 - DESCRIPTION CRAFT, MEASURED ===
# Same tool, two docstrings. The description is the ONLY signal the model has for
# routing "what should I pack for Tokyo?" - write when-to-call, not what-it-is.
BAD = "Gets data."
GOOD = GetWeather.__doc__
question = "what should I pack for Tokyo?"
overlap = lambda desc: len(set(question.lower().split()) & set(desc.lower().split()))
print(f"routing signal for {question!r}:  bad={overlap(BAD)} keyword hits, "
      f"good={overlap(GOOD)} keyword hits")
assert overlap(GOOD) > overlap(BAD)   # a crude proxy - the real test is eval runs


# === SECTION 8 - PLUG IT INTO THE REAL LOOP (needs ANTHROPIC_API_KEY) ===
# The AA.1 loop + this AA.2 boundary = a production-shaped agent in ~20 lines.
# MCP note: an MCP server is someone ELSE'S registry speaking a standard protocol -
# when you mount one, route its tools through THIS dispatch() so no rack skips the gate.
import os

if os.environ.get("ANTHROPIC_API_KEY"):
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": "Weather in Tokyo, then what should I pack?"}]
    while True:
        resp = client.messages.create(model="claude-opus-4-8", max_tokens=1024,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "end_turn":
            break
        blocks = [{"id": b.id, "name": b.name, "input": b.input}
                  for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": [dispatch(b) for b in blocks]})
    print("REAL RUN:", next(b.text for b in resp.content if b.type == "text")[:120])
else:
    print("no ANTHROPIC_API_KEY - skipped the real run (everything above ran offline)")

print("\nLAB COMPLETE - one Pydantic model per tool = schema + validation + types, "
      "gated by reversibility.")
