# Agents

An `Agent` is configuration: instructions, models, tools, output type and
hooks. It is built once, at import, and shared by every thread in the
process. It holds no history, and nothing changes it after construction.

## Definition

```python
from livekit.agents import Agent, RunContext

scheduler = Agent(                       # Agent[Deps, Patient, str], id "scheduler"
    instructions="Northside Family Clinic scheduling. Confirm, book or move appointments.",
    llm="google/gemma-4-31b-it",
    deps_type=Deps,                      # what the tools need for one run, never saved
    state_type=Patient,                  # what the thread saves after every run
    output_type=str,                     # the type of result.output
)
```

| Parameter | Meaning |
|---|---|
| `instructions` | the system prompt, a `str` or an `Instructions` with audio and text variants; dynamic parts come from `@agent.instructions` |
| `name`, `id` | the agent's id; see below |
| `llm`, `stt`, `tts`, `vad` | models, as objects or model strings; a text agent needs `llm` only |
| `tools` | tools and toolsets; see [tools.md](tools.md) |
| `deps_type`, `state_type`, `output_type` | the three type parameters of `Agent[Deps, State, Output]`; see [state-and-deps.md](state-and-deps.md) |
| `turn_handling`, `tool_handling`, `expressive`, ... | voice settings that override the `LiveSession`'s for this agent; see [voice.md](voice.md) |

`deps_type`, `state_type` and `output_type` make the agent an
`Agent[Deps, State, Output]`: its tools and instruction functions take a
`RunContext[Deps, State]`, and `run()` returns a `RunResult[Output]`, so a
typo in `ctx.deps.x` or a wrong use of `result.output` is a type error. The
state, the output and every tool's arguments are validated against their
types; the developer never handles raw dicts.

The id names the agent's lane in a thread (see [lanes.md](lanes.md)). It is
`name=` or `id=` when given; otherwise a module-level `Agent(...)` takes the
name of the variable it is assigned to, and a subclass takes its class name
in snake case (`class FrontDesk(Agent)` is `front_desk`). An unnamed agent
built anywhere else is `default_agent`.

Tools, instruction functions, validators and `on_enter` are registered with
decorators at import. The first time the agent is bound to a thread they
freeze: a later registration raises `RuntimeError`.

## Running a turn

```python
result = await scheduler.run(
    message,                             # str, or a ChatMessage
    message_history=history,             # the conversation so far; omit for a fresh one
    state=state,                         # the typed state; mutated in place by tools
    deps=deps,                           # per-run services
    usage=None,                          # a RunUsage to add this run's usage to
    max_steps=12,                        # the most model requests one turn may make
)
result.output                            # typed as output_type; None while input_required
result.input_required                    # the question a tool is waiting on, or None
result.new_messages()                    # what this turn added, the message first
result.all_messages()                    # the history after the turn, without instructions
result.usage                             # requests, input_tokens, output_tokens, tool_calls
```

`run()` is pure: a turn on the history it is given, a typed result, nothing
kept after it returns and no knowledge of where the message came from. The
next turn continues from `result.all_messages()`. The model and tool loop
stops at the answer, at an `InputRequired` from a tool, or after `max_steps`
model requests (default 12), where it raises `UnexpectedModelBehavior`.

Inside an `a2a_session` endpoint the framework hands the handler the loaded history
and state and commits what the run added after the handler returns; see
[text-endpoints.md](text-endpoints.md). There, `root.run(message)` runs
whichever agent is on top of the lane's agent stack after a handoff
([multi-agent.md](multi-agent.md)), so the handler stays one line. On a call the same agent runs in the
voice pipeline; see [voice.md](voice.md).

## `RunContext`

Every tool, instruction function and validator receives
`RunContext[Deps, State]` as its first argument:

| Member | What |
|---|---|
| `ctx.deps` | what was passed to `run(deps=)` or `LiveSession(deps=)` |
| `ctx.state` | the thread's typed state, loaded; mutate it in place |
| `ctx.thread` | the durable thread this run belongs to, the same object an `a2a_session` handler gets; see [threads.md](threads.md) |
| `ctx.live_session` | the `LiveSession` of the call, `None` outside a call |
| `ctx.task_id` | the A2A task id of the turn; the idempotency key for outside calls |
| `ctx.usage` | token usage so far; pass it to a delegated `run(usage=ctx.usage)` |
| `ctx.function_call` | the call being executed, inside a tool |
| `await ctx.update(text)` | a progress message while the tool runs: a `working` frame on the text door, speech on a call |

On a call the context also carries the speech: `ctx.speech_handle`,
`ctx.wait_for_playout()`, `ctx.with_filler(...)`, `ctx.foreground()` and
`ctx.disallow_interruptions()`; see [voice.md](voice.md).

`RunContext[None, Patient]` says the agent has no deps; `RunContext[None, None]`
no deps and no state.

## Dynamic instructions

```python
@scheduler.instructions
def known_appointment(ctx: RunContext[Deps, Patient]) -> str:
    a = ctx.state.appointment
    return f"Next appointment: {a.when:%A %H:%M} with {a.provider}." if a else "Ask for last name and date of birth."
```

Rendered on every turn, sync or async, and appended to the static
instructions, so the model always knows what the thread knows. The
instructions are never part of the history.

## Structured output

```python
fnol = Agent(..., output_type=Claim)

@fnol.output_validator
async def claim_number_format(ctx: RunContext[None, Loss], output: Claim) -> Claim:
    if not output.claim_number.startswith("CLM-"):
        raise ModelRetry("The claim number must start with CLM-")
    return output
```

With an `output_type` other than `str`, the run gives the model a
`final_result` tool whose parameters are the type's schema, and the answer
is the call to it. A plain text reply sends the model back to call it.
`output_type` accepts any type: a dataclass, a pydantic model, a union, a
list, a scalar. A type whose schema is not an object is wrapped as
`{"result": ...}` in the tool's schema and unwrapped, so `result.output`
always has the declared type.

A validator receives the output and returns it, or a replacement: its
return value is the output. It raises `ModelRetry` to send the model back
with the reason; arguments that do not match the type do the same.
Each retry is a model request counted against `max_steps`.

In a voice turn the reply is speech, so structured results on a call come
from an `AgentTask[T]`; see [agent-tasks.md](agent-tasks.md).

## Hooks

| Hook | When |
|---|---|
| `@agent.instructions` | on every turn, before the model request |
| `@agent.output_validator` | on the final output of `run()` |
| `@agent.on_enter` | when the agent takes a call, at session start or by handoff; receives the `LiveSession` |
| `on_enter`, `on_exit`, `on_user_turn_completed`, `on_user_turn_exceeded` | overridden in an `Agent` subclass, on a call |

The pipeline nodes of a voice agent, `stt_node`, `llm_node`, `tts_node`,
`transcription_node` and `realtime_audio_output_node`, stay overridable by
subclassing. `run()` calls the model through `llm_node` too.

## What an agent is not

An agent has no `chat_ctx=` argument, no `update_instructions`, no
`update_tools` and no `update_chat_ctx`. To steer the
model between turns, render state through `@agent.instructions` or hand off
to another agent. The history belongs to the thread: inside the agent's
hooks and tools, `agent.chat_ctx` is a read-only view of the history the
current run works on, and outside them it raises.

What is designed beyond this, streaming runs, `toolsets=`, history
processors, model-call hooks and capabilities, is in
[planned-agents.md](planned-agents.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py` and
`04_a_claim_with_a_photo.py`.
