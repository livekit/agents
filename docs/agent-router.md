# AgentRouter and agents as tools

An `Agent` is shared configuration. A conversational task collects information
across turns and produces a validated result. `AgentRouter` coordinates these
conversations: it invokes `Agent.run()`, tracks agents called as tools, and
delivers each child's accepted output to its caller.

Applications can call any agent directly. Routing state belongs to the router;
application data and services belong to the context supplied by the application.

## Agent definitions and application data

```python
@dataclass
class BookingContext:
    customer: Customer
    hotel: HotelServices
    draft: RoomDraft = persisted(default_factory=RoomDraft)


book_room = Agent[BookingContext](
    id="book-room",
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
and service objects on each activation. See [application context](state-and-deps.md)
for the load/save contract and scope ownership.

## What a run returns

`Agent.run()` returns `RunResult[Output]`. The result container describes the
invocation's status; `output_type` describes only the application's final value.
The invocation ends when it returns, including when it needs human input.

| `result.status` | Available data | What happens next |
| --- | --- | --- |
| `"completed"` | `result.output: Output` | The child completes and the router delivers its value to the parent tool call |
| `"input_required"` | `result.question: str` | Deliver the question and save the active conversation |

`RunResult[Output]` is a discriminated union. Checking its literal `status`
narrows the result: the completed branch exposes the declared output type,
and the input-required branch exposes the question. History and usage access
are available on both branches.

History methods on the result describe the agent invocation represented by
that result. They do not flatten the router's private frame histories. Router
state retains those separate histories for application inspection.

The router returns an envelope containing `result: RunResult[Output]` and
`state: RouterState`. The contained run result uses this invocation contract.
Router frames and parent-call bookkeeping belong to the envelope's routing
state. See the [result surface](agents.md#running-a-turn).

The model produces the application's final value. The framework constructs
the result container. An `InputRequired` returned by a tool requests the
input-required result; it is not part of the application's output schema.

Ordinary tool calls run within the invocation. Their recorded results go back
to the model, which chooses its next action. Validation failures return feedback
to the model within the invocation's retry and usage budget. Failure, refusal,
cancellation, and timeout retain their own handling.

### A2A mapping

The transport maps framework status to the corresponding A2A task state:

| Python result | A2A task state | A2A payload |
| --- | --- | --- |
| `status="completed"` | `completed` | The accepted output is delivered as the result artifact |
| `status="input_required"` | `input-required` | The question is delivered in the task status message |

The Python spelling uses an underscore; the wire spelling uses a hyphen.
Application output and framework status remain separate in both interfaces.
See [A2A tasks](a2a.md#answering-a-task) for the transport lifecycle.

## Human input

```python
result = await book_room.run(message, context=app, message_history=history)

if result.status == "input_required":
    await send_question(result.question)
elif result.status == "completed":
    booking = result.output  # Booking
```

A model-generated question produces an input-required result. A tool can also request
input directly:

```python
@book_room.tool
async def clarify_view(ctx: RunContext[BookingContext]):
    return InputRequired("Would you prefer a garden or sea view?")
```

The invocation records a complete message sequence before returning. A tool
requesting input records its question as its outcome; its Python body has
finished. On the next human message, a new invocation reads the updated
history and context. The LLM interprets the reply, asks follow-up questions,
and calls tools to update the draft.

Deliver model-generated text directly. `verbatim=True` preserves exact wording
for tool-generated questions. A separate frontend agent may phrase an expert's
question under the application's presentation policy.

For `output_type=str`, conversational tasks use an explicit final-output action
to distinguish completion from a question. A standalone text run returns its
answer as its output. Completion is determined by the output contract.

## Running a routed conversation

```python
@dataclass
class BookingRequest:
    destination: str
    nights: int


front_desk = Agent[BookingContext](
    id="front-desk",
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

router = AgentRouter(agent=front_desk)

routed = await router.run("I need a room", context=app)

routed = await router.run(
    "Two nights",
    state=routed.state,
    context=app,
)
```

The router supplies each agent with its frame's history and the appropriate
application context. It returns an envelope with two members:

| Member | Responsibility |
| --- | --- |
| `routed.result` | `RunResult[Output]`: invocation status, output/question, history, and usage |
| `routed.state` | `RouterState`: frames, active conversation, and parent tool-call relationships |

Read `routed.result.status` to handle the outcome. Save `routed.state` and
changed application fields before accepting the next turn, including
turns whose status is `"input_required"`.

Only the active frame receives unaddressed human input. Its parent remains
inactive while the child owns the conversation. Voice and text use the same
routing and completion contract.

## Loading and saving router state

The application loads routing state and passes it explicitly through `state=`.
The router starts at its root agent when state is `None`; supplied state
identifies the active frame and its history.

Store routing state in an application-defined record with a persisted field:

```python
from dataclasses import dataclass

from livekit.agents import RouterState, persisted


@dataclass
class Conversation:
    routing: RouterState | None = persisted(default=None)
```

For each incoming request:

```python
conversation = Conversation()
await ctx.store.load_context(conversation)
await ctx.store.load_context(app)

routed = await router.run(
    ctx.message,
    state=conversation.routing,
    context=app,
)

conversation.routing = routed.state
await ctx.store.save_context(app)
await ctx.store.save_context(conversation)

result = routed.result
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

`router.run()` reads the supplied state and returns its envelope. The
application assigns `routed.state` and saves explicitly after each retained turn,
including `status="input_required"`. Runtime service objects remain outside routing
state. Load and save any marked application-context fields explicitly as
well; see [context persistence](state-and-deps.md#persisting-router-state).

## Configuring an agent tool

`agent.as_tool(...)` exposes the agent to a parent model through `tools=`.

| Parameter | Meaning |
| --- | --- |
| `tool_name` | Name the parent model uses to call the tool |
| `tool_description` | Explains when the parent should select this specialist |
| `parameters` | Dataclass or Pydantic model defining the parent's tool arguments |
| `needs_approval` | Requires human approval before starting the requested child invocation |
| `chat_history` | `"fresh"` starts from the tool input; `"fork"` also copies selected parent history |
| `merge_chat_history` | `True` returns the child's visible dialogue alongside its typed result; `False` returns only the result |

The framework validates the parent-provided arguments against `parameters`
before starting the child. These arguments become the child's task input.
The child's `output_type` defines its final result. Runtime services and the
persisted draft are supplied through application context.

History selection and merging are independent. For example,
`chat_history="fork", merge_chat_history=False` lets a specialist use the
parent conversation while returning only its accepted result.

## Approval before starting a child

Agent tools use the same `needs_approval=` gate as ordinary tools, including
boolean configuration and read-back functions. The shared approval mechanism
binds the decision to the invocation and its validated arguments. See
[tool approval](tools.md#approval) for the read-back contract.

With `needs_approval=True`, the router returns an input-required result asking
permission to start the requested specialist. The child has not run yet.
Router state retains the pending call's identity, validated arguments, and
parent frame. Save this state and any changed application fields using the
same [load/save flow](#loading-and-saving-router-state) as other human turns.

The next request loads that state and passes the person's reply to
`router.run(...)`. The shared approval mechanism resolves the saved request using the same
read-back and repeated-call rules as ordinary tools. Approval
allows the requested child invocation to start; rejection resolves the parent tool
call with the rejection so the parent can respond. A decision applies to the
recorded invocation and arguments. Changed arguments require a new decision.

Approval to start a specialist and approval for a business action are separate.
A booking specialist may collect preferences before its reservation tool asks
permission to commit the booking. Configure that tool with
`needs_approval=`; bind its approval to the exact reservation terms and
draft revision. See [human input and approvals](human-in-the-loop.md#approvals).

The application explicitly saves each returned routing state, including approval
waits. Router state stores pending decisions and call relationships; runtime
services are supplied again by the application on the next request. External
effects still require their own idempotency and authorization checks.

## Calling an agent as a tool

An agent exposed as a tool is a reusable responsibility with a typed result,
such as booking a room or verifying a guest. The parent model selects that
tool and supplies its input. The router creates a child frame and records the
parent frame ID and originating tool-call ID.

1. The router invokes the child with its own history and application context.
2. If the child result has `status="input_required"`, the router returns an
   input-required result with the question and preserves the active child frame.
   The parent waits.
3. The next human message starts another child invocation with its saved history.
4. A child result with `status="completed"` carries its accepted final output
   and completes the child frame.
5. The router records that value as the originating parent tool call's result,
   transfers visible dialogue if configured, and starts a fresh parent invocation.
   The parent LLM reads the result and selected dialogue, then generates its
   next response.

For `book_room`, the accepted result is a validated `Booking`. The parent
receives that value as its tool result. The output contract itself defines
completion; producing it is the child's final action.

The waiting relationship is serialized data. It retains the parent frame and
call identity while each Python invocation finishes. A parent tool function
is not suspended across human turns.

The router supplies the child's result to the recorded call before invoking
the parent model again. A parent waiting for a child is not sent to the model
with an unresolved call. Ordinary follow-up questions stay within the active
child conversation.

## Remote agent tools

`A2AAgent(url).as_tool()` exposes a remote endpoint through the same agent-tool
lifecycle. The remote card defines its public tool contract; its service owns
history, approval, and execution. The router retains the references needed
to forward human replies and deliver completion to the parent. Progress,
cancellation, and protocol metadata are handled by the adapter. See
[remote agents as tools](agents.md#remote-agents-as-tools).

## Frames and history isolation

Each conversational frame records:

- A unique frame ID and stable agent ID.
- Its own model-compatible message history.
- Its parent frame ID and originating parent tool-call ID, when called as a tool.
- The application's context-scope reference.

Router state identifies the active frame and its ordered parent chain. Stack
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
| `chat_history="fork"` | Its own instructions, a copy of selected parent conversation history, and the agent-tool input |

A history selector can filter or transform the parent messages before the
child starts. It can pass relevant exchanges, redact fields, or produce a
summary. The selector customizes the starting history rather than creating
a separate conversation mode.

Forking copies history when the child frame is created. Preserve complete
message/tool exchanges and leave the unfinished parent delegation call out
of the child's input. The child uses its own agent instructions.

After `InputRequired`, resume the recorded child history. Starting history
selection is applied once per child frame, not again on every human reply.

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
remains available in router state for application inspection.

For example, a booking agent may learn that the customer is travelling with
a baby. Returning visible dialogue lets the parent retain that preference
even when it is outside the `Booking` result schema. An address validator may
need to return only a validated address.

| Use case | Starting history | Return behavior |
| --- | --- | --- |
| Validate a supplied address | Fresh | Result only |
| Book a room using previously discussed preferences | Fork | Result and visible dialogue |
| Review a selected part of a conversation | Fork with a history selector | Result only |

The router delivers the chosen information before invoking the parent again.
Keep the parent's tool call/result sequence valid, preserve dialogue ordering,
and record transferred messages so later turns do not append them again.
Only dialogue added during the child conversation is eligible for transfer;
the starting history copied from the parent is not copied back.

Application context sharing is independent of both history choices. The
application supplies a compatible shared context or constructs a child
context with its own persisted draft. A shared customer or service reference
can be reused while booking drafts remain separate. See
[application context](state-and-deps.md) for scope and persistence ownership.

## Authoritative completion and approval

Tools enforce availability, price, prerequisites, and approval. Resolve final
results against the authoritative service using an output validator or an
output function:

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
Each turn has its own run identity.

Approval binds to the exact effect terms and draft revision. Changed dates,
room, price, guest, or payment details invalidate prior approval. Recheck
availability and authorization before committing the booking.

## Persistence and execution ownership

Router state contains serializable frame histories and routing metadata.
The application explicitly loads and saves it alongside the relevant context
fields. Runtime services are reconstructed by application code. Save routing
and application changes consistently so both describe the same completed turn.

Serialize turns for one conversation and use revision checks on saves. Bound
agent-tool calls and nesting depth. Resolve stable agent IDs through the router's
configured agent definitions; an unknown ID produces a routing error.

Conversation persistence preserves the position between turns. External effects
require domain validation, idempotency, and reconciliation. Python call stacks
and arbitrary await points are outside the persisted state.

## Direct execution

`agent.run(message, message_history=..., context=...)` always runs that agent.
A stored routing frame is consulted only by `router.run(..., state=...)`.

An application switching to direct execution supplies the selected frame's
history and appropriate context. An active child's history is independently
usable by `Agent.run()`. A parent waiting for a child must first receive the
child outcome for its outstanding tool call.

To return to routed execution, retain router state and update the selected
frame with the direct run's resulting history. Routing state retains the
parent-call relationship needed to deliver an accepted child result.
