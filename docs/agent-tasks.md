# Agent tasks

An agent is shared configuration. A task is one conversational execution of an agent,
with its own draft state, history, pending interactions, and eventual output.
Define behavior with `Agent(...)` and register tools and instructions with
its decorators.

## Definition, draft, and output

```python
book_room = Agent[BookingContext](
    id="book-room",
    instructions=(
        "Collect and refine the booking details. Ask for anything missing. "
        "Reserve only after approval. Produce the final Booking only after "
        "the reservation service confirms it."
    ),
    llm=MODEL,
    output_type=Booking,
)
```

`Agent[BookingContext]` describes the application context type; output_type
supplies the final-result contract. Construct the definition once and register
tools/instructions at import. A run receives context explicitly; the shared
agent never contains mutable job data.

```python
@dataclass
class BookingContext:
    customer: Customer
    hotel: HotelServices
    draft: RoomDraft = persisted(default_factory=RoomDraft)
```

Tools read/write `ctx.context.draft` and call `ctx.context.hotel`. Identity,
services, helper methods, and mutable data share one local context object.
Dynamic instructions render selected context data for the model.
Output types remain validated final-result contracts.

Mark saved user data with `persisted(default_factory=...)`.
`store.load_context` explicitly restores and validates those fields; the
application calls `store.save_context` to save their changes. The
application supplies fresh customer and service objects. See [application context](state-and-deps.md) for field keys,
context-scope ownership, collections, compatibility, and load/save lifecycle.

A draft is scoped to a booking context, not the shared Agent definition.
Multiple booking jobs can use the same book_room definition with independent
drafts. Handoffs can share a scope, and an explicit fresh child context creates
a new one. Active-agent identity and conversation history are stored separately
in framework execution records.

## Agent-loop outcomes

| Agent loop produces | Runtime behavior |
| --- | --- |
| Ordinary tool call | Run tool, record output, continue the model loop |
| Person-facing conversational text | Checkpoint InputRequired; deliver text and release execution |
| Explicit InputRequired from a tool | Save pending call and question; wait for a human reply |
| Call of a child agent | Persist child execution and parent receiver; parent waits |
| Valid accepted final output | Complete execution and deliver typed result |
| Terminal failure, refusal, cancellation, or timeout | Persist and deliver distinct terminal outcome |

An ordinary reply ends a turn. Accepted final output completes the task.

For structured output, expose an explicit final-output mechanism alongside
ordinary conversational text. Text must remain available for questions; do
not retry every question as an invalid Booking. Validation failures in final
output return feedback to the model within the execution's retry/usage budget.

For `output_type=str`, intermediate and final text cannot be distinguished by
type. Conversational tasks use an explicit final-output action even
for strings. A normal standalone text `agent.run()` retains its existing
one-turn behavior, where an answer is that run's output. These execution
modes must be explicit in runner/context; never guess completion from wording.

## Questions and automatic resumption

A model-generated question/conversational reply is checkpointed as
`InputRequired` with an agent-loop receiver. The runtime delivers the text directly. The next human message appends
to history and resumes the same agent.

A Python tool can also return a question directly:

```python
@book_room.tool
async def clarify_view(ctx: RunContext[BookingContext]):
    return InputRequired("Would you prefer a garden or sea view?")
```

Here the tool call is pending: the human answer becomes its output before
the model continues. The original tool body is not called again to interpret
the answer. The LLM can clarify, answer side questions, and change the draft.
`verbatim=True` preserves exact wording; a separate frontend agent may phrase
an expert's question under an explicit presentation policy.

Waits and delivery must commit before releasing execution. Voice and text
share this contract. Only the foreground execution receives unaddressed human
input; a parent paused for a child cannot compete for the same reply.

## Booking tools and authoritative completion

Tools enforce availability, price, prerequisites, and approval. The output
schema alone cannot prove that a model-generated booking actually exists.
Validate completion against the authoritative service, either in an output
validator or an output function.

```python
async def booking_result(
    ctx: RunContext[BookingContext],
    confirmation_code: str,
) -> Booking:
    # Check ownership and that this reservation belongs to this execution.
    return await ctx.context.hotel.get_confirmed_booking_for_execution(
        confirmation_code,
        execution_id=ctx.execution_id,
    )


book_room = Agent[BookingContext](
    id="book-room",
    instructions=BOOKING_INSTRUCTIONS,
    llm=MODEL,
    output_type=booking_result,
)
```

When `output_type` is a function, its parameters define the final-output
arguments the model must provide. The framework validates those arguments
before calling the function. The function may request a model retry; its
accepted return value is the final typed output. The function is a terminal output surface,
not an ordinary tool whose result is returned to the model. Its input schema,
return type, context types, and retries need full typing and runtime validation.
Output functions should prefer validation/read-only resolution of already
committed effects; side effects still require their own idempotency contract.

`ctx.execution_id` identifies the task across human turns. Each A2A request
or turn has its own identity.

Booking tools such as set_stay, choose_room, record_guest, and payment-reference
collection update `ctx.context.draft`. An `@book_room.instructions` function
renders the current draft. Booking writes return tool results; accepted final
output completes the task.

Approval must bind to exact effect terms and draft identity/revision. Changed
dates, room, price, guest, or payment details invalidate prior approval even
when the tool has no arguments. A model interpreting a reply cannot authorize
different terms. Recheck current availability before committing a booking.

## Child executions

A parent agent tool delegates a reusable responsibility:

```python
@front_desk.tool
async def start_booking(ctx: RunContext[CustomerContext]):
    app = ctx.context
    return Call(
        book_room,
        context=BookingContext(customer=app.customer, hotel=app.hotel),
    )
```

The explicit child context above gets a fresh durable scope and RoomDraft;
customer and hotel references are reused as runtime dependencies. Child
creation, its initial persisted fields, and parent tool receiver commit
consistently, then the parent pauses. The child's accepted Booking output
fills the pending call and the parent LLM phrases the result. The original
Python tool is not rerun and no cross-turn Python stack is restored.

`Call(child)` without a context override intentionally shares compatible
context and persisted scope. Both agents read and modify the same application
data. A handoff also retains the current scope. Restoration uses recorded scope IDs; it never recreates a new
draft just because another worker receives the turn. History transfer remains
explicit and independent of shared application data.

Use children for reusable conversations such as VerifyGuest. Ordinary booking
questions stay in book_room.

## Deferred work

Durable execution, external-operation waits, schedulers, journals for recovery,
and deployment compatibility are outside this phase.
Graphs and [code workflows](code-workflows.md) are also deferred.

The current proposal covers conversational tasks, human input, typed final
output, child agents, and [persisted context fields](state-and-deps.md).
Persistence does not restore Python stacks or automatically retry external effects.

Acceptance checks cover intermediate text alongside typed final output,
string-output disambiguation, output-function retries and provenance checks,
shared and isolated child contexts, persisted fields, model- versus
tool-originated questions, approval invalidation, and parent resumption.
