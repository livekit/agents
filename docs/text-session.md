# TextSession

`TextSession` coordinates text conversations across requests: agent selection,
private histories, pending expert calls, and human input. It can be reconstructed
for each request; pass the previously saved `TextSessionState` to continue the
same logical conversation. Agent definitions remain shared configuration.

Each `run()` builds request-scoped execution from the supplied state. A text
request has a bounded lifetime; its conversation state can outlive the request.
The application explicitly loads and saves that state.
The `TextSession` object, runtime services, and Python execution stack are not
serialized.

The text session invokes `Agent.run()`, tracks agents called as tools, and
delivers each child's accepted output to its caller.

Applications can call any agent directly. Routing state belongs to the session;
application data and services belong to the context supplied by the application.

`VoiceSession` owns frontend expert-tool interactions, event delivery, and human
reply correlation. It does not require a `TextSession`. See
[frontend expert events](voice-session.md#conversation-owned-inbox).

`VoiceTask` has a separate frontend role: a bounded live interaction inside
an active voice session, with its parent waiting in memory. Backend stateless
runs use `Agent.run()`; backend conversations spanning requests use routing
state. See [frontend voice interactions](voice-tasks.md#voicetask).

## Agent definitions and application data

```python
@dataclass
class BookingContext:
    customer: Customer
    hotel: HotelServices
    draft: RoomDraft = persisted(default_factory=RoomDraft)


book_room = Agent[BookingContext](
    name="book-room",
    instructions=(
        "Collect the booking details. Ask for anything missing. "
        "Reserve only after approval. Produce a final Booking only after "
        "the reservation service confirms it."
    ),
    llm=MODEL,
    output_type=Booking,
)
```

An agent used as a tool must declare `output_type`. It defines the value the
parent receives and the condition that completes the child. A standalone
conversational agent can leave `output_type` unspecified.

Construct the agent once and register tools and instructions at import.
Tools update `ctx.context.draft` and call `ctx.context.hotel`. Dynamic
instructions render selected context data for the model.

`persisted(...)` selects stored fields. The application calls `load_context`
to restore them and `save_context` to save changes. It supplies fresh customer
and service objects on each activation. See [application context](context.md)
for the load/save contract and scope ownership.

## Session results

`TextSession.run()` returns `TextSessionTurn[Output]`, an envelope with
`result: RunResult[Output]` and `state: TextSessionState`. These types are
exported from `livekit.agents`. The contained result uses the shared
[agent result contract](agents.md#running-a-turn): a completed typed output or
an input-required question. The envelope retains session frames and pending
calls separately from that invocation result.

A pending execution keeps its stored identity across human turns. After completion,
a new request starts a new execution while retaining session history. On A2A,
`ctx.execution_id` equals the task ID. A pending session must be resumed under
that same task; supplying a different task is refused before execution.

On input required, save the returned state before accepting another request.
The next request starts a new invocation of the recorded active frame; no
Python coroutine remains suspended. A frontend can phrase the endpoint's
question through its own presentation policy. A2A maps the outcome under its
[task-answering contract](protocol/a2a.md#answering-a-task).

## Running a text conversation

```python
@dataclass
class BookingRequest:
    destination: str
    nights: int


front_desk = Agent[BookingContext](
    name="front-desk",
    instructions="Help customers plan their stay.",
    llm=MODEL,
    tools=[
        book_room.as_tool(
            tool_name="book_room",
            tool_description="Collect preferences and book a room for the customer.",
            parameters=BookingRequest,
            needs_approval=True,
            chat_history="fork",
            merge_chat_history=True,
        ),
    ],
)

session = TextSession(agent=front_desk)

turn = await session.run("I need a room", context=app)

turn = await session.run(
    "Two nights",
    state=turn.state,
    context=app,
)
```

The session supplies each agent with its frame's history and the appropriate
application context. It returns an envelope with two members:

| Member | Responsibility |
| --- | --- |
| `turn.result` | `RunResult[Output]`: invocation status, output/question, history, and usage |
| `turn.state` | `TextSessionState`: frames, active conversation, and parent tool-call relationships |

Read `turn.result.status` to handle the outcome. Save `turn.state` and
changed application fields before accepting the next turn, including
turns whose status is `"input_required"`.

Only the active backend frame receives unaddressed input. Its parent remains
inactive while the child owns the backend conversation. When invoked remotely
by a frontend, this backend returns progress, questions, and results through
its endpoint; the caller's `VoiceSession` independently owns their presentation.

## Streaming a backend invocation

`session.run_stream(...)` accepts the same input, state, and context as `run()`.
It is an async context manager with `stream_text`, `stream_events`, and
`result`. The settled value remains the session envelope: `turn.result`
and `turn.state`. Individual child events identify their agent, invocation,
and session frame; they do not merge private histories into the parent.

The last text fragment is not a save boundary. Retrieve the settled envelope,
assign its returned state, and explicitly save related application fields and
routing state, including an input-required outcome. Cancellation or failure
raises and does not manufacture a successful result/state envelope. See
[text session streaming](agents.md#text-session-streaming).

## Loading and saving session state

The application loads routing state and passes it explicitly through `state=`.
The session starts at its root agent when state is `None`; supplied state
identifies the active frame and its history.

Store routing state in an application-defined record with a persisted field:

```python
from dataclasses import dataclass

from livekit.agents import TextSessionState, persisted


@dataclass
class Conversation:
    session_state: TextSessionState | None = persisted(default=None)
```

For each incoming request:

```python
session = TextSession(agent=front_desk)
conversation = Conversation()
await ctx.store.load_context(conversation)
await ctx.store.load_context(app)

turn = await session.run(
    ctx.message,
    state=conversation.session_state,
    context=app,
)

conversation.session_state = turn.state
await ctx.store.save_context(app)
await ctx.store.save_context(conversation)

result = turn.result
if result.status == "input_required":
    await send_question(result.question)
elif result.status == "completed":
    await show_result(result.output)
```

The runtime resolves the thread from the incoming request and supplies its
scoped `ctx.store`. Loading restores that thread's saved routing field;
a new conversation starts with the declared `None` default. `Conversation`
is an application-defined storage record, while `app` is the freshly
constructed application context supplied to agents.

`session.run()` reads the supplied state and returns its envelope. The
application assigns `turn.state` and saves explicitly after each retained turn,
including `status="input_required"`. Runtime service objects remain outside routing
state. Load and save any marked application-context fields explicitly as
well; see [context persistence](context.md#persisting-text-session-state).

Agent-tool names, typed parameters, approvals, and history selection are
configured through [the tool API](tools.md#configuring-an-agent-tool).

## Approval before starting a child

Agent tools use the same `needs_approval=` gate as ordinary tools, including
boolean configuration and read-back functions. The shared approval mechanism
binds the decision to the invocation and its validated arguments. See
[tool approval](tools.md#approval) for the read-back contract.

With `needs_approval=True`, the session returns an input-required result asking
permission to start the requested specialist. The child has not run yet.
Session state retains the pending call's identity, validated arguments, and
parent frame. Save this state and any changed application fields using the
same [load/save flow](#loading-and-saving-session-state) as other human turns.

The next request loads that state and passes the person's reply to
`session.run(...)`. The shared approval mechanism resolves the saved request using the same
read-back and repeated-call rules as ordinary tools. Approval
allows the requested child invocation to start; rejection resolves the parent tool
call with the rejection so the parent can respond. A decision applies to the
recorded invocation and arguments. Changed arguments require a new decision.

Approval to start a specialist and approval for a business action are separate.
A booking specialist may collect preferences before its reservation tool asks
permission to commit the booking. Configure that tool with
`needs_approval=`; bind its approval to the exact reservation terms and
draft revision. See [human input and approvals](tools.md#approval).

The application explicitly saves each returned routing state, including approval
waits. Session state stores pending decisions and call relationships; runtime
services are supplied again by the application on the next request. External
effects still require their own idempotency and authorization checks.

## Calling an agent as a tool

An agent exposed as a tool is a reusable responsibility with a typed result,
such as booking a room or verifying a guest. The parent model selects that
tool and supplies its input. The session creates a child frame and records the
parent frame ID and originating tool-call ID.

1. The session invokes the child with its own history and application context.
2. If the child result has `status="input_required"`, the session returns an
   input-required result with the question and preserves the active child frame.
   The backend caller receives this outcome. A remote frontend can phrase
   the question through its own live-session presentation mechanism.
3. The next human message starts another child invocation with its saved history.
4. A child result with `status="completed"` carries its accepted final output
   and completes the child frame.
5. The session records that value as the originating parent tool call's result,
   transfers visible dialogue if configured, and starts a fresh parent invocation.
   The parent LLM reads the result and selected dialogue, then generates its
   next response.

For `book_room`, the accepted result is a validated `Booking`. The parent
receives that value as its tool result. The output contract itself defines
completion; producing it is the child's final action.

The waiting relationship is serialized data. It retains the parent frame and
call identity while each Python invocation finishes. A parent tool function
is not suspended across human turns.

The session supplies the child's result to the recorded call before invoking
the parent model again. A parent waiting for a child is not sent to the model
with an unresolved call. Ordinary follow-up questions stay within the active
backend child conversation. Frontend presentation belongs to the calling
`VoiceSession`, outside this text session.

## Remote agent tools

`RemoteAgent(url).as_tool()` exposes an A2A endpoint through the same agent-tool
lifecycle. The remote card defines its public tool contract; its service owns
history, approval, and execution. The text session retains the references needed to forward replies and
deliver completion to its parent agent. The same remote tool can be attached
to a frontend agent; there, `VoiceSession` owns those interactions without a
session. Progress, cancellation, and protocol metadata are handled by the adapter. See
[remote agents as tools](tools.md#remote-agents-as-tools).

## Frames and history isolation

Each conversational frame records:

- A unique frame ID and stable agent name.
- Its own model-compatible message history.
- Its parent frame ID and originating parent tool-call ID, when called as a tool.
- The application's context-scope reference.

Session state identifies the active frame and its ordered parent chain. Stack
depth is the number of active frames from root to child; it is not a Python
call-stack depth. Completed frames may remain available for inspection but
are not active ancestors. Multiple frames may invoke the
same agent definition with independent histories and drafts.

## History passed to the child

Starting history and information returned to the parent are independent choices.
Each child owns its model history throughout its conversation.

| Starting history | Child receives |
| --- | --- |
| `chat_history="fresh"` | Its own instructions and the agent-tool input |
| `chat_history="fork"` | Its own instructions, a copy of the parent conversation history, and the agent-tool input |

Forking copies history when the child frame is created. Preserve complete
message/tool exchanges and leave the unfinished parent delegation call out
of the child's input. The child uses its own agent instructions.

After `InputRequired`, resume the recorded child history. The starting history
is chosen once per child frame, not again on every human reply.

## Information returned to the parent

An accepted typed output always becomes the parent tool call's result.
The application also chooses whether the parent receives the dialogue that
occurred while the child owned the conversation:

| Return behavior | Parent receives |
| --- | --- |
| `merge_chat_history=False` | The child's accepted typed output |
| `merge_chat_history=True` | The typed output plus the user and assistant messages exchanged with the person during the child conversation |

Visible dialogue contains user messages and assistant messages actually
delivered to the person. Child instructions, private tool calls/results, and
internal model work remain in the child's frame. The complete child history
remains available in session state for application inspection.

For example, a booking agent may learn that the customer is travelling with
a baby. Returning visible dialogue lets the parent retain that preference
even when it is outside the `Booking` result schema. An address validator may
need to return only a validated address.

| Use case | Starting history | Return behavior |
| --- | --- | --- |
| Validate a supplied address | Fresh | Result only |
| Book a room using previously discussed preferences | Fork | Result and visible dialogue |

The session delivers the chosen information before invoking the parent again.
Keep the parent's tool call/result sequence valid, preserve dialogue ordering,
and record transferred messages so later turns do not append them again.
Only dialogue added during the child conversation is eligible for transfer;
the starting history copied from the parent is not copied back.

Application context sharing is independent of both history choices. The
application supplies a compatible shared context or constructs a child
context with its own persisted draft. Pass `context_factory=` to `TextSession`
for separate child contexts. The sync or async callback receives the agent's name
and a `ContextStore` bound to that child's scope, and returns the application's
live context. It constructs services and explicitly calls `store.load_context`;
tools explicitly call `ctx.store.save_context` to retain changes. Each child
frame has its own scope, preserved in saved state. Reconstructing a session calls
the factory again for each child that runs; objects are reused within one request.
The root uses the supplied `context=`. A shared customer or service reference
can be reused while booking drafts remain separate. See
[application context](context.md) for scope and persistence ownership.

## Persistence and execution ownership

Session state contains serializable frame histories and routing metadata.
The application explicitly loads and saves it alongside the relevant context
fields. Runtime services are reconstructed by application code. Save routing
and application changes consistently so both describe the same completed turn.

Serialize turns for one conversation and use revision checks on saves.
`TextSession(agent=..., max_depth=8)` bounds the stack depth; an agent-tool call
beyond it fails. Resolve stable agent names through the session's configured
agent definitions; an unknown name produces a routing error.

Conversation persistence preserves the position between turns. External effects
require domain validation, idempotency, and reconciliation. Python call stacks
and arbitrary await points are outside the persisted state.

## Direct execution

`agent.run(message, message_history=..., context=...)` always runs that agent.
A stored routing frame is consulted only by `session.run(..., state=...)`.

An application switching to direct execution supplies the selected frame's
history and appropriate context. An active child's history is independently
usable by `Agent.run()`. A parent waiting for a child must first receive the
child outcome for its outstanding tool call.

To return to text-session execution, retain session state and update the selected
frame with the direct run's resulting history. Routing state retains the
parent-call relationship needed to deliver an accepted child result.
