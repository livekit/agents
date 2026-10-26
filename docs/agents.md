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
| `output_type` | Final output contract; required for an agent exposed as a tool, including output functions |
| Voice options | Agent overrides for the live session |

`Agent[SchedulingContext]` and `RunContext[SchedulingContext]` carry the
application context type. Tools read application data and services through
`ctx.context`. The output contract determines the type of run results. See
[application context](state-and-deps.md) for persistence responsibilities.

The id identifies the agent definition. It is
`name=` or `id=` when given; otherwise a module-level `Agent(...)` takes the
name of the variable it is assigned to, and a subclass takes its class name
in snake case (`class FrontDesk(Agent)` is `front_desk`). An unnamed agent
built anywhere else is `default_agent`.

Tools, instruction functions, validators and `on_enter` are registered with
decorators at import. The first time the agent is bound to a thread they
freeze: a later registration raises `RuntimeError`.

## Conversation ownership

The caller supplies history and application context for each invocation.
A direct run continues from `result.all_messages()`. A routed conversation
keeps each agent's history in a separate router frame. Persisted drafts live
in application context; runtime clients and services are supplied again on
each request. The shared agent definition stores none of these values.

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
result.status                            # "completed" or "input_required"
# After narrowing on status:
# result.output                          # declared output type when completed
# result.question                        # str when input_required
result.new_messages()                    # this invocation’s input and recorded messages
result.all_messages()                    # supplied history plus this invocation, without instructions
result.usage                             # requests, input_tokens, output_tokens, tool_calls
```

`RunResult[Output]` separates framework status from the application's output.
It is a discriminated union: `status == "completed"` exposes `output: Output`,
and `status == "input_required"` exposes `question: str`. Both branches expose
history and usage. A tool's `InputRequired(...)` sets the framework outcome;
`output_type` contains only the application's final-result contract. See the
[result and A2A mapping](agent-router.md#what-a-run-returns).

`RunResult` describes one agent invocation:

| Member | Purpose |
| --- | --- |
| `status` | Distinguish completion from a request for human input |
| `output` | The accepted application value in the completed branch |
| `question` | The question in the input-required branch |
| `new_messages()` | This invocation’s input and recorded messages, including tool calls/results |
| `all_messages()` | Supplied history plus this invocation; suitable for the next direct run |
| `usage` | Usage accumulated by this invocation |

These methods return this agent invocation's model history, including tool
calls and results. `new_messages()` includes the incoming message recorded for
this invocation; it is not limited to assistant output. `all_messages()` also
includes the supplied history. Neither method combines private histories from
other router frames. User-visible dialogue is a separate view.

Both outcome branches retain the same history and usage access. Streaming
delivers incremental text and events; the settled result uses this contract.
`AgentRouter.run()` returns an envelope containing this result as
`routed.result` and routing state as `routed.state`. Frame selection, parent
tool-call correlation, and context scopes belong to the router.

`run()` operates on the supplied history and context; tools may mutate that
context and perform external effects. The shared Agent definition keeps no
per-run history or application data after it returns. The
next turn continues from `result.all_messages()`. The model and tool loop
stops when it accepts the final output or requests human input. After
`max_steps` model requests (default 12), it raises `UnexpectedModelBehavior`.

Inside a managed `a2a_session` endpoint the runtime supplies the loaded history
and scoped `ctx.store`. The application constructs its context with fresh
runtime dependencies, explicitly calls `await ctx.store.load_context(app)`,
passes `context=app` to the run, and calls `await ctx.store.save_context(app)`
to save changes. See [context loading](state-and-deps.md#explicit-load-and-save-on-the-store).
`agent.run()` executes the agent the caller selected.
[AgentRouter](agent-router.md) selects active frames for routed conversations. On a call the same agent runs in the
voice pipeline; see [voice.md](voice.md).

## `RunContext`

Every tool, instruction function and validator receives
`RunContext[Context]` as its first argument:

| Member | What |
|---|---|
| `ctx.context` | the application object passed to `run(context=...)`; data, services, and helper methods |
| `ctx.store` | the managed store for explicit `load_context` and `save_context` calls |
| `ctx.thread` | the durable thread this run belongs to, the same object an `a2a_session` handler gets; see [threads.md](threads.md) |
| `ctx.live_session` | the `LiveSession` of the call, `None` outside a call |
| `ctx.task_id` | the A2A task ID of the turn; external operations need an operation-specific idempotency key |
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

`output_type` defines the accepted final result. It is mandatory when an agent
is exposed as a tool: accepting a value completes that child and supplies the
parent tool result. An input-required result ends one invocation while the
conversation continues. See [agents as tools](agent-router.md).

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
produces the final result. Conversational text that asks for input produces
a result with `status="input_required"`.
`output_type` accepts any type: a dataclass, a pydantic model, a union, a
list, a scalar. A type whose schema is not an object is wrapped as
`{"result": ...}` in the tool's schema and unwrapped, so `result.output`
has the declared type in the `status="completed"` branch.

A validator receives the output and returns it, or a replacement: its
return value is the output. It raises `ModelRetry` to send the model back
with the reason; arguments that do not match the type do the same.
Each retry is a model request counted against `max_steps`.

For multi-turn conversations with typed final results, see
[agent-router.md](agent-router.md).

## Hooks

| Hook | When |
|---|---|
| `@agent.instructions` | on every turn, before the model request |
| `@agent.output_validator` | on the final output of `run()` |
| `@agent.on_enter` | when the agent becomes active in a live session; receives the `LiveSession` |
| `on_enter`, `on_exit`, `on_user_turn_completed`, `on_user_turn_exceeded` | overridden in an `Agent` subclass, on a call |

The pipeline nodes of a voice agent, `stt_node`, `llm_node`, `tts_node`,
`transcription_node` and `realtime_audio_output_node`, stay overridable by
subclassing. `run()` calls the model through `llm_node` too.

## History and streaming

To steer the model between turns, render context through
`@agent.instructions`. Select another agent explicitly through a direct run,
a router-managed agent tool, or the live session. History belongs to the conversation: inside the agent's
hooks and tools, `agent.chat_ctx` is a read-only view of the history the
current run works on, and outside them it raises.

`agent.stream(message, ...)` is the same turn as an async iterator of the
reply's text, one piece as the model writes it, with `run.result` the
`RunResult` once the loop ends.

Iteration, filtered/prefixed toolsets, history processors, model-call
hooks and capabilities are specified in [planned-agents.md](planned-agents.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py` and
`04_a_claim_with_a_photo.py`.

## Working with other agents

| Pattern | Execution | History |
| --- | --- | --- |
| Direct delegation | A tool explicitly awaits another agent's `run()` | The caller selects the expert's input and history; its result becomes tool output |
| Agent as a tool | An `AgentRouter` manages a child across turns and delivers its typed output to the parent tool call | Each child has its own frame and history |
| Remote agent as a tool | `A2AAgent(url).as_tool()` connects to another endpoint | The remote service owns its conversation; the router retains continuation references |

## Direct delegation

Call another agent directly when the caller can consume its result within
one invocation:

```python
@main_line.tool
async def quick_balance(ctx: RunContext[CustomerContext], account_number: str) -> str:
    result = await billing.run(
        f"Look up the balance for account {account_number}. Do not ask follow-up questions.",
        context=ctx.context,
        usage=ctx.usage,
    )
    if result.status == "input_required":
        return f"Balance lookup needs more information: {result.question}"
    return str(result.output)
```

The outer tool waits for this invocation to finish. Its parent model then
reads the returned tool value. This starts no router frame or persistent
parent-child relationship. To continue the expert directly, the application
must retain its history and supply it to another run.

## Agents as tools

Use `AgentRouter` when a specialist may need human replies before producing
its result. Define the specialist's final `output_type`, then expose it in
the parent's `tools=`:

```python
@dataclass
class BookingRequest:
    destination: str
    nights: int


book_room = Agent[BookingContext](
    id="book-room",
    instructions="Collect preferences and reserve a room after approval.",
    llm=MODEL,
    output_type=Booking,
)

front_desk = Agent[BookingContext](
    id="front-desk",
    instructions="Help customers plan their stay.",
    llm=MODEL,
    tools=[book_room.as_tool(
        tool_name="book_room",
        tool_description="Collect preferences and book a room.",
        parameters=BookingRequest,
        needs_approval=True,
        chat_history="fork",
        merge_chat_history=True,
    )],
)

router = AgentRouter(agent=front_desk)
routed = await router.run(message, state=saved_routing, context=app)
```

The parent model selects the tool and supplies validated `BookingRequest`
arguments. `needs_approval` uses the shared tool approval gate before the
child starts. Actions inside the child, such as reserving or charging, can
require their own approval.

The router runs the child in its own frame. An input-required result keeps
that frame active for the next human reply. An accepted `Booking` completes
the child and fills the parent's original tool call; a fresh parent LLM
invocation then reads the result. Each Python invocation finishes between
human turns.

`chat_history="fresh"` starts with the specialist's instructions and tool input;
`"fork"` also copies selected parent history. `merge_chat_history=True` returns
new visible user/assistant dialogue with the result; `False` returns only the
result. Private child tools and instructions remain in the child frame.
Application-context sharing is independent of these history choices.

`routed.result` is the invocation's `RunResult`; `routed.state` holds frames
and pending calls. The application explicitly loads the routing record before
calling the router and saves the returned state and changed persisted context
fields after each turn, including approval and human-input waits. See the
[router specification](agent-router.md) for the complete entrypoint, history
rules, and switching between routed and direct execution.

## Remote agents as tools

An `A2AAgent` connects to an agent endpoint. Expose it through `.as_tool()`
and attach the returned tool to the parent:

```python
from livekit.agents.a2a import A2AAgent

back_office = A2AAgent(
    "https://northside.livekit.cloud/agents/eligibility/production/eligibility"
)
front_desk = Agent(
    instructions="Help patients with appointments and insurance eligibility.",
    llm=MODEL,
    tools=[back_office.as_tool()],
)

router = AgentRouter(agent=front_desk)
routed = await router.run(message, state=saved_routing, context=app)
```

The remote card supplies the tool's name, description, and supported input.
The remote service owns its history, application data, approval behavior,
and completion contract. The local tool takes its configuration from that
endpoint. Local agent-tool options such as `chat_history`,
`merge_chat_history`, and `needs_approval` do not configure the remote tool.

The router coordinates the conversation while the adapter handles the A2A
protocol. Progress is forwarded to the user as it arrives. An input-required
response keeps the remote conversation active; the next human reply goes to
that conversation. Completion delivers the remote result to the parent tool
call, then the parent model continues. Failures and cancellation propagate
through the tool lifecycle.

Router state retains the remote endpoint and conversation/task references
needed to continue on another request. The remote service retains its private
conversation history. The application explicitly saves routing state and
marked context fields using the same entrypoint as local agent tools.

The adapter manages connection lifecycle, authentication, task correlation,
and delegation metadata. Applications configure the endpoint and attach its
tool; protocol calls and metadata recording are internal.

When the endpoint advertises the LiveKit extension, the adapter uses its
thread-correlation protocol so the remote agent can keep its own lane in
the caller's thread. Other A2A endpoints retain their own remote context.
See [A2A](a2a.md) for the wire protocol and authentication rules.

## Endpoints versus deployments

An endpoint is a door; a deployment is the boundary. Two endpoints in one
deployment share the worker and the history for a thread and differ in who
may call them. A second deployment buys its own worker, its own history,
and its own owner, deploy and version, at the price of a network hop and
state that does not cross. The test is the one for a web service: would
this thing get a URL of its own, and run on its own machines? A different
caller: an endpoint. A different worker or owner: a deployment. To organize
code: neither; use a `Toolset`, a direct agent run, or an agent tool.

See `examples/use-cases/03_insurance_eligibility.py` and
`07_hospital_main_line.py`.
