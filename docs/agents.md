# Agents

An `Agent` is configuration: instructions, models, tools, output type and
hooks. It is built once, at import, and shared by every thread in the
process. It holds no history, and nothing changes it after construction.

## Definition

An agent declares its application context type with `Agent[Context]`. Pass an
instance to `run(context=...)`; tools and instruction functions access it as
`ctx.context`.

```python
from livekit.agents import Agent, RunContext

scheduler = Agent[SchedulingContext](
    instructions="Confirm, book or move appointments.",
    llm=MODEL,
    output_type=Appointment,
)
```

| Parameter | Meaning |
| --- | --- |
| `instructions` | Static or channel-specific instructions; dynamic parts come from registered functions |
| `name`, `id` | Stable agent identity; see below |
| `llm`, `stt`, `tts`, `vad` | Model objects or supported strings |
| `tools` | Tools and toolsets |
| `output_type` | Final output contract, including output functions |
| Voice options | Agent overrides for the live session |

`Agent[SchedulingContext]` and `RunContext[SchedulingContext]` carry the
application context type. Tools read application data and services through
`ctx.context`. The output contract determines the type of run results. See
[application context](state-and-deps.md) for persistence responsibilities.

The id names the agent's lane in a thread (see [lanes.md](lanes.md)). It is
`name=` or `id=` when given; otherwise a module-level `Agent(...)` takes the
name of the variable it is assigned to, and a subclass takes its class name
in snake case (`class FrontDesk(Agent)` is `front_desk`). An unnamed agent
built anywhere else is `default_agent`.

Tools, instruction functions, validators and `on_enter` are registered with
decorators at import. The first time the agent is bound to a thread they
freeze: a later registration raises `RuntimeError`.

## Persisted application fields

A run's context marks stored fields with `persisted(default_factory=...)`.
Call `store.load_context(app)` to restore and validate them, then call
`store.save_context(app)` after the run to save changes. Marking a field
alone does not save mutations. Unmarked customers, clients, and services
are application-supplied runtime values. Scope belongs to the
conversation/job, not to the agent definition. See
[context and persisted fields](state-and-deps.md) for the complete example.

## Running a turn

```python
result = await scheduler.run(
    message,                             # str, or a ChatMessage
    message_history=history,             # the conversation so far; omit for a fresh one
    context=context,                     # typed application data and dependencies
    usage=None,                          # a RunUsage to add this run's usage to
    max_steps=12,                        # the most model requests one turn may make
)
result.output                            # typed as output_type; None while input_required
result.input_required                    # the question a tool is waiting on, or None
result.new_messages()                    # what this turn added, the message first
result.all_messages()                    # the history after the turn, without instructions
result.usage                             # requests, input_tokens, output_tokens, tool_calls
```

`run()` operates on the supplied history and context; tools may mutate that
context and perform external effects. The shared Agent definition keeps no
per-run history or application data after it returns. The
next turn continues from `result.all_messages()`. The model and tool loop
stops at the answer, at an `InputRequired` from a tool, or after `max_steps`
model requests (default 12), where it raises `UnexpectedModelBehavior`.

Inside a managed `a2a_session` endpoint the runtime supplies the loaded history
and scoped `ctx.store`. The application constructs its context with fresh
runtime dependencies, explicitly calls `await ctx.store.load_context(app)`,
passes `context=app` to the run, and calls `await ctx.store.save_context(app)`
to save changes. See [context loading](state-and-deps.md#explicit-load-and-save-on-the-store).
The active agent is restored from the execution record after a handoff;
see [multi-agent.md](multi-agent.md). On a call the same agent runs in the
voice pipeline; see [voice.md](voice.md).

## `RunContext`

Every tool, instruction function and validator receives
`RunContext[Context]` as its first argument:

| Member | What |
|---|---|
| `ctx.context` | the application object passed to run(context=...); data, services, and helper methods |
| `ctx.thread` | the durable thread this run belongs to, the same object an `a2a_session` handler gets; see [threads.md](threads.md) |
| `ctx.live_session` | the `LiveSession` of the call, `None` outside a call |
| `ctx.task_id` | the A2A task id of the turn; the idempotency key for outside calls |
| `ctx.usage` | token usage so far; pass it to a delegated `run(usage=ctx.usage)` |
| `ctx.function_call` | the call being executed, inside a tool |
| `await ctx.update(text)` | a progress message while the tool runs: a `working` frame on the text door, speech on a call |

On a call the context also carries the speech: `ctx.speech_handle`,
`ctx.wait_for_playout()`, `ctx.with_filler(...)`, `ctx.foreground()` and
`ctx.disallow_interruptions()`; see [voice.md](voice.md).

`RunContext[None]` indicates no application context. Conversation history and
runtime metadata remain separate from the application object.

## Dynamic instructions

```python
@scheduler.instructions
def known_appointment(ctx: RunContext[SchedulingContext]) -> str:
    a = ctx.context.patient.appointment
    return f"Next appointment: {a.when:%A %H:%M} with {a.provider}." if a else "Ask for last name and date of birth."
```

Rendered on every turn, sync or async, and appended to the static
instructions, so the model always knows what the thread knows. The
instructions are never part of the history.

## Structured output

`output_type` defines the accepted final result. Multi-turn tasks can ask
questions before producing that result; see [agent tasks](agent-tasks.md).

```python
fnol = Agent(..., output_type=Claim)

@fnol.output_validator
async def claim_number_format(ctx: RunContext[ClaimContext], output: Claim) -> Claim:
    if not output.claim_number.startswith("CLM-"):
        raise ModelRetry("The claim number must start with CLM-")
    return output
```

With an `output_type` other than `str`, the run gives the model a
`final_result` tool whose parameters are the type's schema. An accepted call
produces the final result. In a conversational task, ordinary text waits for
human input; in a standalone structured run, the model must produce the
final result before the run returns.
`output_type` accepts any type: a dataclass, a pydantic model, a union, a
list, a scalar. A type whose schema is not an object is wrapped as
`{"result": ...}` in the tool's schema and unwrapped, so `result.output`
always has the declared type.

A validator receives the output and returns it, or a replacement: its
return value is the output. It raises `ModelRetry` to send the model back
with the reason; arguments that do not match the type do the same.
Each retry is a model request counted against `max_steps`.

For multi-turn conversations with typed final results, see
[agent-tasks.md](agent-tasks.md).

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

## History and streaming

To steer the model between turns, render context through
`@agent.instructions` or hand off to another agent. The history belongs to the thread: inside the agent's
hooks and tools, `agent.chat_ctx` is a read-only view of the history the
current run works on, and outside them it raises.

`agent.stream(message, ...)` is the same turn as an async iterator of the
reply's text, one piece as the model writes it, with `run.result` the
`RunResult` once the loop ends.

What is designed beyond this, `toolsets=`, history processors, model-call
hooks and capabilities, is in [planned-agents.md](planned-agents.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py` and
`04_a_claim_with_a_photo.py`.
