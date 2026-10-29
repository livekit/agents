# Agents

An `Agent` is configuration: instructions, an LLM, tools, output type and
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
| `instructions` | Static instructions; dynamic parts come from registered functions |
| `name` | Stable agent identity; see below |
| `llm` | The reasoning model |
| `tools` | Tools and toolsets |
| `output_type` | Final output contract; required for an agent exposed as a tool, including output functions |
| `retries` | Retry budgets: an `int` for both, or `AgentRetries`; see [retries](#retries) |

`Agent[SchedulingContext]` and `RunContext[SchedulingContext]` carry the
application context type. Tools read application data and services through
`ctx.context`. The output contract determines the type of run results. See
[application context](context.md) for persistence responsibilities.

The name identifies the agent definition. It is `name=` when given;
otherwise a module-level `Agent(...)` takes the name of the variable it is
assigned to when it is first bound. An unnamed agent built anywhere else is
`default_agent`.

Tools, instruction functions, and lifecycle hooks are registered with
decorators at import. The first time the agent is bound to a thread they
freeze: a later registration raises `RuntimeError`.

## Registering tools

Register typed functions on the shared agent during application setup:

```python
@scheduler.tool
async def lookup_appointment(
    ctx: RunContext[SchedulingContext], reference: str
) -> Appointment:
    """Look up an appointment by its reference."""
    return await ctx.context.clinic.get_appointment(reference)
```

The model selects a tool and supplies its arguments; the framework validates
those arguments, invokes the function with `RunContext`, and records its result
for the next model request. See [tools](tools.md) for approvals, human input,
errors, toolsets, MCP, and local or remote agents exposed as tools.

## Voice configuration

`VoiceSession` owns speech recognition, speech synthesis, VAD, turn detection,
endpointing, interruption behavior, and audio delivery. Configure them when
constructing the session and change them with `session.update_options(...)`.
The same agent definition can be used in text and voice.

Agent selection and voice configuration are separate operations. When an
application wants a different voice or listening policy for a specialist,
it explicitly updates the live session. See
[updating a live session](voice-session.md#updating-a-live-session).

## Conversation ownership

The caller supplies history and application context for each invocation.
A direct run continues from `result.all_messages()`. A text-session conversation
keeps each agent's history in a separate session frame. Persisted drafts live
in application context; runtime clients and services are supplied again on
each request. The shared agent definition stores none of these values.

## Persisted application fields

A run's context marks stored fields with `persisted(default_factory=...)`.
Call `store.load_context(app)` to restore and validate them, then call
`store.save_context(app)` after the run to save changes. Marking a field
alone does not save mutations. Unmarked customers, clients, and services
are application-supplied runtime values. Scope belongs to the
conversation/job, not to the agent definition. See
[context and persisted fields](context.md) for the complete example.

## Running a turn

```python
result = await scheduler.run(
    message,                             # str, or a ChatMessage
    message_history=history,             # the conversation so far; omit for a fresh one
    context=context,                     # typed application data and dependencies
    usage=None,                          # a RunUsage to add this run's usage to
    retries=None,                        # this run's retry budgets; see "Retries"
)
result.status                            # "completed" or "input_required"
# After narrowing on status:
# result.output                          # declared output type when completed
# result.question                        # str when input_required
result.new_messages()                    # this invocation’s input and recorded messages
result.all_messages()                    # supplied history plus this invocation, without instructions
result.usage                             # requests, input/output/total_tokens, tool_calls
```

`RunResult[Output]` separates framework status from the application's output.
It is a discriminated union: `status == "completed"` exposes `output: Output`,
and `status == "input_required"` exposes `question: str`. The branches are
`RunCompleted[Output]` and `RunInputRequired`, exported with `RunResult` from
`livekit.agents`. Both branches expose history and usage. A tool's
`InputRequired(...)` sets the framework outcome; `output_type` contains only
the application's final-result contract. See the
[result and A2A mapping](#running-a-turn).

`RunResult` describes one agent invocation:

| Member | Purpose |
| --- | --- |
| `status` | Distinguish completion from a request for human input |
| `output` | The accepted application value in the completed branch |
| `question` | The question in the input-required branch |
| `new_messages()` | This invocation’s input and recorded messages, including tool calls/results |
| `all_messages()` | Supplied history plus this invocation; suitable for the next direct run |
| `usage` | Usage accumulated by this invocation |
| `expect` | Event assertions over this invocation; see [testing](testing.md#programmatic-tests) |

`RunUsage` records `requests`, `input_tokens`, `output_tokens` and `tool_calls`;
`total_tokens` is `input_tokens + output_tokens`. Pass a shared `RunUsage` as
`usage=` to accumulate usage across delegated runs. Each result still reports
that invocation's usage.

These methods return this agent invocation's model history, including tool
calls and results. `new_messages()` includes the incoming message recorded for
this invocation; it is not limited to assistant output. `all_messages()` also
includes the supplied history. Neither method combines private histories from
other session frames. User-visible dialogue is a separate view.

Both outcome branches retain the same history and usage access. Streaming
delivers incremental text and events; the settled result uses this contract.
`TextSession.run()` returns an envelope containing this result as
`turn.result` and routing state as `turn.state`. Frame selection, parent
tool-call correlation, and context scopes belong to the session.

`run()` operates on the supplied history and context; tools may mutate that
context and perform external effects. The shared Agent definition keeps no
per-run history or application data after it returns. The
next turn continues from `result.all_messages()`. The model and tool loop
stops when it accepts the final output or requests human input.

Inside a managed `a2a_session` endpoint the runtime supplies the loaded history
and scoped `ctx.store`. The application constructs its context with fresh
runtime dependencies, explicitly calls `await ctx.store.load_context(app)`,
passes `context=app` to the run, and calls `await ctx.store.save_context(app)`
to save changes. See [context loading](context.md#explicit-load-and-save-on-the-store).
`agent.run()` executes the agent the caller selected.
[TextSession](text-session.md) reconstructs and coordinates text conversations
from explicitly supplied state on each request.
On a call, `VoiceSession` owns frontend interaction and expert reply routing; see [voice-session.md](voice-session.md).

## `RunContext`

Every tool, instruction function and output function receives
`RunContext[Context]` as its first argument:

| Member | What |
|---|---|
| `ctx.context` | the application object passed to `run(context=...)`; data, services, and helper methods |
| `ctx.store` | the managed store for explicit `load_context` and `save_context` calls |
| `ctx.thread` | the durable thread this run belongs to, the same object an `a2a_session` handler gets; see [threads.md](threads.md) |
| `ctx.live_session` | the `VoiceSession` of the call, `None` outside a call |
| `ctx.task_id` | the A2A task ID of the turn, or `None` outside A2A; external operations need an operation-specific idempotency key |
| `ctx.usage` | token usage so far; pass it to a delegated `run(usage=ctx.usage)` |
| `ctx.function_call` | the call being executed, inside a tool |
| `await ctx.update(text)` | a progress message while the tool runs: a `working` frame on the text door, speech on a call |

On a call the context also carries the speech: `ctx.speech_handle`,
`ctx.wait_for_playout()`, `ctx.with_filler(...)`, `ctx.foreground()` and
`ctx.disallow_interruptions()`; see [tool calls on a call](voice-session.md#tool-calls-on-a-call).

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
conversation continues. See [agents as tools](text-session.md).

```python
class Claim(BaseModel):
    claim_number: str = Field(pattern=r"^CLM-\d+$")
    next_step: str


fnol = Agent(..., output_type=Claim)
```

With an `output_type` other than `str`, the run gives the model a
`final_result` tool whose parameters are the type's schema. An accepted call
produces the final result. Conversational text that asks for input produces
a result with `status="input_required"`.
`output_type` accepts any type: a dataclass, a pydantic model, a union, a
list, a scalar. A type whose schema is not an object is wrapped as
`{"result": ...}` in the tool's schema and unwrapped, so `result.output`
has the declared type in the `status="completed"` branch.

The framework validates the final value against its declared schema. Invalid
arguments return feedback to the model within the run's retry budget.
Use constraints on the output data structure for format checks. An output
function can resolve the final value from application data or an authoritative
service; its parameters define the final-output arguments and its return type
defines the result. Raising `ModelRetry` asks the model to correct the attempted
output. See [authoritative completion](#authoritative-completion-and-approval).

`@agent.output_validator` registers a check that runs on the validated output
before the run accepts it. It receives the `RunContext` and the output and
returns the value to accept, or raises `ModelRetry` to send the model back with
the reason:

```python
@fnol.output_validator
async def cites_real_jobs(ctx: RunContext, output: Claim) -> Claim:
    if not await ctx.context.claims.exists(output.claim_number):
        raise ModelRetry(f"There is no claim {output.claim_number}; use one the customer named.")
    return output
```

For multi-turn conversations with typed final results, see
[text-session.md](text-session.md).

## Retries

Retries follow pydantic-ai. The model gets a budget of consecutive retries
before the run fails with `UnexpectedModelBehavior`:

| Budget | Consumed by | Reset by |
| --- | --- | --- |
| `tools` | each tool on its own: arguments that fail validation, `ModelRetry` from the tool | a call of that tool that did not ask for a retry |
| `output` | a final output that fails validation, `ModelRetry` from an output validator or output function | an accepted output |

```python
from livekit.agents import Agent, AgentRetries

fnol = Agent(instructions="...", llm=MODEL, output_type=Claim, retries=2)
fnol = Agent(instructions="...", llm=MODEL, output_type=Claim,
             retries=AgentRetries(tools=3, output=1))

@fnol.tool(retries=5)                  # this tool's own budget
async def lookup_policy(ctx: RunContext, number: str) -> Policy: ...

result = await fnol.run(message, retries={"output": 3})   # this run only
```

`retries=` takes an `int`, which sets both budgets, or an `AgentRetries`
dict `{"tools": int, "output": int}`. On `Agent(...)` each budget defaults
to `1`. The same `retries=` on a run entrypoint (`agent.run`,
`agent.run_stream`, `session.run` and `session.run_stream`) overrides the
agent's budgets for that run; a key it omits keeps the agent's value. On a
call, each reply spends its agent's budgets. In a `TextSession` turn it applies to every
agent of the session. `retries=` on a tool (`@agent.tool`, `@function_tool`,
`MCPToolOptions`) replaces the `tools` budget for that tool. A budget of `0`
fails on the first retry. There is no separate `output_retries=`.

## Authoritative completion and approval

Tools enforce availability, price, prerequisites, and approval. Resolve final
results against the authoritative service using an output function:

```python
async def booking_result(
    ctx: RunContext[BookingContext],
    confirmation_code: str,
) -> Booking:
    return await ctx.context.hotel.get_confirmed_booking_for_execution(
        confirmation_code,
        execution_id=ctx.execution_id,
    )
```

For a callable `output_type`, its parameters define the final-output arguments.
Validate them before invoking the function. Its accepted return value is the
final typed result; `ModelRetry` asks the model to correct its attempted output.
Output functions resolve already committed effects. Effectful operations have
their own idempotency contract.

`ctx.execution_id` identifies the conversational task across human turns.
Each turn has its own run identity. The framework stores the execution id in
the run's persisted state, so a resumed run keeps it. On A2A it is the
transport task id, which a frontend can supply by continuing that task.

Approval binds to the exact effect terms and draft revision. Changed dates,
room, price, guest, or payment details invalidate prior approval. Recheck
availability and authorization before committing the booking.

## Hooks

| Hook | When |
|---|---|
| `@agent.instructions` | on every turn, before the model request |
| `@agent.on_enter` | when the agent takes the active live role; receives the `VoiceSession` |
| `@agent.on_exit` | when the agent relinquishes the active live role during an orderly agent replacement or session close; receives the `VoiceSession` |

```python
@front_desk.on_enter
async def enter(session: VoiceSession) -> None:
    await session.generate_reply(instructions="Greet the caller and offer assistance.")


@front_desk.on_exit
async def exit(session: VoiceSession) -> None:
    await release_frontend_resources(session)
```

These hooks describe frontend activation, not individual turns or backend
requests. Returning `InputRequired` does not leave the active role. An agent
selected again later enters a new activation, so hooks may run more than once
during a live session. Application code should distinguish an initial greeting
from re-entry when needed.

A `VoiceTask` temporarily suspends the parent interaction without ending its
activation. Starting the task does not call the parent's `on_exit`; restoring
the parent does not call its `on_enter`. This avoids repeating greetings or
releasing parent resources during a bounded inline interaction.

Use `on_exit` for orderly frontend cleanup. A process crash cannot guarantee
its execution; external effects and persistence need their own explicit
contracts. Direct backend `Agent.run()` calls do not trigger live-session hooks.

## History

To steer the model between turns, render context through
`@agent.instructions`. Select another agent explicitly through a direct run,
a session-managed agent tool, or the live session. History belongs to the conversation: inside the agent's
hooks and tools, `agent.chat_ctx` is a read-only view of the history the
current run works on, and outside them it raises.

## Streaming runs

`agent.run_stream(...)` runs the same model/tool loop as `agent.run(...)` and
exposes incremental events through an async context manager. It accepts the
same message, history, context, and usage arguments. Streaming
does not change tool execution, output validation, approval, or human-input
semantics.

Backend `session.run_stream(...)` follows the same lifecycle. Its final value
is the same result/state envelope returned by `session.run(...)`. Frontend
`VoiceSession` presentation and its expert-event queue remain session-owned;
starting a stream does not create a text session.

### Text and a final result

```python
async with assistant.run_stream(
    message,
    message_history=history,
    context=app,
) as stream:
    async for delta in stream.stream_text(delta=True):
        await send_text(delta)

    result = await stream.result()  # RunResult[Output]

if result.status == "input_required":
    question = result.question
else:
    output = result.output

history = result.all_messages()
```

The stream context owns the active execution and its cleanup. Execution starts
when the context is entered, not when the context-manager object is constructed.
The stream cannot be entered twice or reused for another invocation.

`result()` returns the settled, validated result. It is awaitable and never
returns `None` merely because execution is unfinished. Model/provider failures
and cancellation retain their normal exception behavior.
A failed or cancelled run does not manufacture a completed result.

The last visible text fragment is not the completion boundary. Tools, output
functions, usage accounting, history finalization, and run cleanup can follow
it. Normal iterator exhaustion means this invocation has settled; failures are
raised through iteration. Call `result()` for its typed final value.

### Stream handle

`RunStream[Result]` carries the type returned by `result()`. For a direct
agent, that is `RunResult[Output]`. For a text session, it is the session's
existing envelope containing `result: RunResult[Output]` and `state: TextSessionState`.
The stream does not put session metadata into the contained `RunResult`.

| Member | Contract |
| --- | --- |
| `stream_text(*, delta=False)` | Async iterator of user-visible assistant text; one presentation view of this run |
| `stream_events()` | Async iterator of typed semantic `StreamEvent` records from this run |
| `await result()` | Finish any unconsumed work and return the settled typed result; repeat calls return that same result |
| `cancel()` | Request cancellation of unfinished execution; idempotent and safe to call from another task on the same event loop |

The handle is returned by the run context; applications do not construct it:

```python
from collections.abc import AsyncIterator
from typing import Generic, TypeVar


ResultT = TypeVar("ResultT")


class RunStream(Generic[ResultT]):
    def stream_text(self, *, delta: bool = False) -> AsyncIterator[str]: ...

    def stream_events(self) -> AsyncIterator[StreamEvent]: ...

    async def result(self) -> ResultT: ...

    def cancel(self) -> None: ...
```

`stream_text` and `stream_events` are alternative views, not independent runs.
Only one iterator can consume a stream, and that iterator cannot be consumed
concurrently. Starting another iterator raises `RuntimeError`, including trying
to switch from events to text partway through. A consumer needing both text and
tool events uses `stream_events` and selects text events itself.

Call `result()` after the iterator finishes, after explicitly closing an
early-stopped iterator, or without starting an iterator at all. With no active
iterator, it consumes remaining notifications internally while execution
finishes, preventing an unread queue from blocking the run. Calling it while
an iterator is actively consuming raises `RuntimeError`. Internally consumed
notifications are not replayed later; the final history and result still record
what ran. A settled result can be read after the context exits.

Breaking iteration does not by itself cancel execution. To deliberately finish
without displaying the rest, close the iterator and call `result()`. To
stop unfinished work, call `cancel()` or leave the context. Stream events are
single-consumer notifications, not a durable replay subscription.

Buffering is bounded. A slow consumer applies backpressure to notification
production; text, questions, and terminal tool outcomes are not silently dropped.
Cancellation and cleanup must remain responsive while a consumer is blocked.
The frontend's separate progress-coalescing queue does not change this run
stream's event identity or history contract.

### Text semantics

With `delta=True`, each value is the next text fragment. With `delta=False`,
each value is the accumulated text of the current assistant message. A new
message starts a new accumulation. A run can produce several messages around
tool calls; applications displaying separate message bubbles use event message
IDs rather than concatenating snapshots across messages.

The text iterator selects `TextDeltaEvent` content from the same execution.
Tool arguments, tool results, private model reasoning, and raw structured
final-output JSON are not rendered as user-visible text. Progress sent through
`ctx.update(...)` is a separate progress event. A question can be streamed as
text, but its authoritative full wording is `result.question` after settlement.

Partial text is presentation, not an accepted final output or permission to
perform an external effect. A run can stream text before choosing a tool or
asking for human input. Output schemas and output functions apply to the final
value exactly as in `run()`. Structured-output runs may produce no text events;
`result()` still returns their complete typed value. Partial structured
objects are not part of this API.

A retry or interruption cannot retract text already delivered. Message-end
events identify interrupted messages; new attempts have distinct message IDs.
Use the event view when the interface must distinguish these message lifecycles.

### Semantic events

`StreamEvent` is a discriminated union over literal `type` fields. Its payloads
are provider-independent typed records. Checking `event.type` narrows the
available fields. Provider wire events and internal journal records are not
implicitly exposed through this union.

Every event has:

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | Literal discriminator | One of the event types below |
| `sequence` | `int` | Increasing occurrence number within this stream; not a durable reconnect cursor |
| `agent_name` | `str` | Name of the agent responsible for this occurrence |
| `invocation_id` | `str` | Invocation producing it; distinguishes repeated calls to the same agent |
| `frame_id` | `str | None` | Backend session frame when applicable; absent on a direct run |

| Type / record | Additional fields | Meaning |
| --- | --- | --- |
| `"text_delta"` / `TextDeltaEvent` | `message_id: str`, `delta: str` | User-visible assistant text added to that message |
| `"text_end"` / `TextEndEvent` | `message_id: str`, `interrupted: bool` | This assistant message ended; the run may continue |
| `"tool_call"` / `ToolCallEvent` | `call_id: str`, `tool_name: str`, `arguments: dict[str, JsonValue]` | Validated, complete arguments for a selected tool call; not partial argument JSON |
| `"tool_result"` / `ToolResultEvent` | `call_id: str`, `tool_name: str`, `outcome: Literal["completed", "failed", "cancelled"]`, `output: JsonValue`, `error: str | None` | The recorded call outcome; errors have an error description and may use a null output |
| `"progress"` / `ProgressEvent` | `message: str`, `call_id: str | None`, `tool_name: str | None` | An application progress update, such as `ctx.update(...)`; call fields are set when emitted inside a tool |

`JsonValue` is JSON null, boolean, number, string, a list of JSON values, or a
string-keyed dictionary of JSON values. Model/dataclass tool outputs are encoded
into this representation. The stream's tool-result payload is an inspection
record; the final `RunResult.output` retains its declared application type.

Within a message, deltas precede its text-end event. A recorded tool result
follows its tool-call event. Concurrent tools can interleave; correlate by
call ID, not adjacency. An unfinished cancelled call has a cancellation outcome
when execution has enough information to record it. Event exhaustion alone
must not imply that a partially observed external effect was rolled back.

Tool errors handled by the model are tool-result events and the run continues.
An unhandled execution error raises through the stream rather than being
encoded as a successful terminal result. There is no final-output value hidden
in a text-end event; `result()` owns final result retrieval.

```python
async with assistant.run_stream(message, context=app) as stream:
    async for event in stream.stream_events():
        if event.type == "text_delta":
            await ui.append_text(event.message_id, event.delta)
        elif event.type == "text_end":
            await ui.finish_message(event.message_id, interrupted=event.interrupted)
        elif event.type == "progress":
            await ui.show_progress(event.message)
        elif event.type == "tool_call":
            await diagnostics.tool_selected(event.call_id, event.tool_name)
        elif event.type == "tool_result":
            await diagnostics.tool_finished(event.call_id, event.outcome)

    result = await stream.result()
```

These events are application-facing execution data, not all public dialogue.
A text session can emit child tool events with their frame identity without
merging that private history into a parent model's history. Only explicitly
selected presentation text belongs in the frontend's text stream. Applications
choose which event fields their external clients may see.

### Input requests and approvals

`InputRequired` and approval read-backs finish this invocation's stream with
`RunResult.status == "input_required"`. The final question is available through
`result.question`. There is no open stream or suspended backend coroutine
waiting for the person's later answer.

Save retained history, application fields, and text session state explicitly.
The next input starts another invocation with that saved data. Streaming does
not add a separate approval API or change the shared `needs_approval` gate.
A question already streamed is finalized as the same question, not delivered
again merely because the caller reads `result.question`.

### Cancellation and cleanup

`cancel()` requests cancellation promptly at supported execution boundaries.
It does not promise to interrupt arbitrary application code, undo tool effects,
or roll back an independently committed business operation. Cancellation of a
remote call attempts the transport's cancellation protocol; a remote effect
may already have completed.

After cancellation, active iteration and `result()` raise
`asyncio.CancelledError` once cancellation and cleanup settle. There is no
completed `RunResult`, no successful session envelope, and no implicit save of
application fields. Cancellation after successful settlement is a no-op and
does not replace the cached result. Normal completion and cancellation are
serialized by the run so only one terminal outcome wins.

Leaving the async context before settlement requests cancellation and awaits
owned cleanup. It does not silently run the rest to completion. A body exception
or caller cancellation remains the primary exception; cleanup must not replace
it with a fabricated successful result. A normal early context exit can return
after cancelling the run; a later `result()` still raises cancellation.

Providers, tools, and remote operations must implement their own bounded cleanup
and reconciliation policies. A frontend `VoiceTask` uses its live-session
cancellation contract; this run-stream handle does not persist or resume
its Python await.

### Text session streaming

```python
conversation = Conversation()
await ctx.store.load_context(conversation)
await ctx.store.load_context(app)

async with session.run_stream(
    ctx.message,
    state=conversation.session_state,
    context=app,
) as stream:
    async for event in stream.stream_events():
        await handle_backend_event(event)

    turn = await stream.result()

conversation.session_state = turn.state
await ctx.store.save_context(app)
await ctx.store.save_context(conversation)
return turn.result
```

`session.run_stream` accepts the same arguments as `session.run`. It continues
backend child conversations and resolves parent calls under the same history
and output rules. Its stream covers the backend invocation, including child
work performed before the final result or input-required outcome. Child event
identities remain separate; emitting them does not grant access to other
context scopes or merge their private histories.

The returned state is the retained position for the next request. An error or
cancellation does not provide a successful replacement state. Retry from the
last explicitly saved state and reconcile external effects as needed.

### Managed endpoints and persistence

A run's settled result is not a promise that the surrounding handler transaction
has committed. `result()` finalizes the run's result and history; the
application then stages its explicit saves, and the managed endpoint commits
when its handler completes. External business operations keep their own
idempotency and commit boundaries.

```python
@server.a2a_session(endpoint="support", description="Airline support.")
async def support(ctx: A2ASessionContext) -> RunResult[Refund]:
    app = await make_airline_context(ctx)
    await ctx.store.load_context(app)

    async with support_agent.run_stream(
        ctx.message,
        message_history=ctx.message_history,
        context=app,
    ) as stream:
        await ctx.stream(stream.stream_text(delta=True))
        result = await stream.result()

    await ctx.store.save_context(app)
    return result
```

`run_stream` hands delivery to its consumer. It does not also automatically
forward those same text occurrences through the managed endpoint. In this
example, `ctx.stream(...)` performs incremental delivery. Returning the full
result finalizes the existing answer artifact or input-required status; it does
not send the streamed text a second time. A direct `agent.run()` can retain the
managed endpoint's automatic forwarding behavior.

A failed or cancelled run can leave already delivered text visible. It does not
convert that text into an accepted typed result. Save context only at explicit
application-selected boundaries; do not unconditionally save partial mutations
in `finally`. Merely opening, consuming, or closing a stream saves no application
context or text session state.


## Frontend interaction tasks

For a fast brain attached to an active voice session, `VoiceTask` temporarily
owns the live interaction and returns a typed result to its waiting caller.
It is scoped to that frontend session and relies on an in-memory Python await.
It is not a stateless workload API or a backend persistence mechanism. See
[frontend voice interactions](voice-tasks.md#voicetask).

## Working with other agents

| Pattern | Execution | History |
| --- | --- | --- |
| Direct delegation | A tool explicitly awaits another agent's `run()` | The caller selects the expert's input and history; its result becomes tool output |
| Backend agent as a tool | A `TextSession` manages a child across requests and delivers its typed output to the parent tool call | Each child has its own frame and history |
| Remote agent as a tool | `RemoteAgent(url).as_tool()` connects to another endpoint | The remote service owns its conversation; `VoiceSession` or the text session retains continuation references |

Detailed delegation contracts and examples live in
[tools](tools.md#agents-as-tools); `TextSession` owns backend continuation and
`VoiceSession` owns frontend presentation and expert reply routing.
