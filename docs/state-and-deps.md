# Application context and persisted fields

Declare the application context type with `Agent[Context]`, pass an instance
with `run(context=...)`, and access it in tools through `ctx.context`.
One typed object contains identity, mutable application data, services, and
helper methods.

Persistence is opt-in **per field**, not per entire context. Mark a user's
validated data structure with `persisted(...)`; explicitly load its saved
values with `load_context` and save changes with `save_context`. Ordinary
fields remain runtime dependencies/data.

This proposal covers conversation and draft persistence. A durable execution
engine, automatic task recovery, scheduler wake-ups, and operation journals
are deferred. Transaction and effect examples below explain persistence limits;
they do not require a new durable runtime in this phase.

## A complete context

```python
from dataclasses import dataclass

from pydantic import BaseModel

from livekit.agents import Agent, RunContext, persisted


class RoomDraft(BaseModel):
    offer_id: str | None = None
    payment_reference: str | None = None
    confirmation_code: str | None = None


@dataclass
class BookingContext:
    customer: Customer
    hotel: HotelServices

    draft: RoomDraft = persisted(
        default_factory=RoomDraft,
    )


book_room = Agent[BookingContext](
    id="book-room",
    instructions=BOOKING_INSTRUCTIONS,
    llm=MODEL,
    output_type=Booking,
)


@book_room.tool
async def choose_room(ctx: RunContext[BookingContext], room_type: str) -> str:
    app = ctx.context
    offer = await app.hotel.find_offer(room_type)
    app.draft.offer_id = offer.id
    return describe_offer(offer)
```

Customer, HotelServices, Booking, and the domain service methods are
application-defined. The runtime sees the draft marker through dataclass
field metadata. It validates the declared type and never attempts to infer
persistence from whether an arbitrary object happens to serialize.

| Value | Included by `save_context`? | How it is restored |
| --- | --- | --- |
| `draft` | Yes, under `draft` | Decode and validate RoomDraft before user code |
| `customer` | No | Application supplies/loads it for each activation |
| `hotel` | No | Application supplies a live service for each activation |
| History, active agent, waits, approvals | Separately, by the runtime | Load the execution record |

The application constructs runtime context as usual:

```python
context = BookingContext(
    customer=await customers.get(authenticated_customer_id),
    hotel=hotel_services,
)
```

Call `save_context` to save the marked draft. Refresh customer information
from its authoritative source for each activation. Stable principal/tenant
and context-scope references belong in the execution envelope so reactivation
can resolve the same authorized subject without waiting for a new HTTP request.

`RunContext[BookingContext]` is the framework wrapper. Only `ctx.context` is
the application object; execution/thread IDs, usage, pending call, progress,
approval, and live-session access are runtime metadata. Local context is not
automatically sent to the model. Render selected facts through instructions
or tools. Agent definitions never hold mutable execution context.

## Conversation data versus booking data

A root conversation can also mark its own application data. For example:

```python
class ConversationPreferences(BaseModel):
    language: str = "en"
    last_booking_code: str | None = None


@dataclass
class CustomerContext:
    customer: Customer
    hotel: HotelServices
    preferences: ConversationPreferences = persisted(
        default_factory=ConversationPreferences,
    )
```

These preferences belong to the root conversation scope. BookingContext's
draft belongs to the separate booking scope created by the explicit child
call below. The customer's account profile remains authoritative in the
customer service/database; conversation preferences do not silently update
it. A child's final Booking result lets the parent update its own
preferences.last_booking_code explicitly. State is not merged on child exit.

The runtime separately saves the active agent and parent/child stack. A
handoff updates that active-agent identity while keeping the compatible
context scope; a child suspends its parent and uses its recorded scope.
History, preferences, draft, and customer profile are not interchangeable
forms of memory.

## Explicit load and save on the store

The runtime supplies `ctx.store`, already bound to the current thread,
execution, context scope, and transaction ownership. It opens/releases the
connection and manages leases; application code does not open storage, close
connections, or pass thread/scope IDs for ordinary context persistence.
Construct the application context explicitly, then load its marked fields.

`customers` and `hotel_service` are application dependencies.
`ctx.customer_id` carries the authenticated customer identity.

```python
@server.a2a_session(endpoint="booking", description="Book a hotel room.")
async def booking(ctx: A2ASessionContext):
    store = ctx.store

    app = BookingContext(
        customer=await customers.get(ctx.customer_id),
        hotel=hotel_service,
    )
    await store.load_context(app)

    result = await book_room.run(ctx.message, context=app)

    await store.save_context(app)
    return result.output
```

load_context mutates only marked fields and returns the same typed object
for callers preferring assignment. Existing saved values replace defaults
before the agent runs; a new scope uses declared defaults. Runtime customer
and service fields are untouched. The store tracks the loaded revision so
save_context can reject stale writes. Framework scope/revision metadata is
not serialized as application fields.

Top-level helpers are equivalent conveniences using the supplied store:

```python
await load_context(app, store=ctx.store)
await save_context(app, store=ctx.store)
```

Use the scoped store for both methods. Entrypoints, tools, and tests receive
the store from their execution host. The runtime manages its connection.

A store view is activation-scoped and must not be retained in persisted
context or reused after its runtime ends. A child receiver uses the child's
recorded scope; a handoff or shared-context child retains the existing one.
Returning from the handler lets the runtime finish its commit and release
resources.

## Explicit saves

`persisted(...)` selects fields for storage. Mutating a marked field changes
only the in-memory object until the application calls `save_context`.
`agent.run(...)` and returning from a handler do not implicitly save context.

For each activation:

1. Construct the context with fresh runtime dependencies.
2. Call `await store.load_context(app)` to restore and validate marked fields.
3. Run the agent with `context=app`.
4. Call `await store.save_context(app)` to save the resulting application data.

Save after each turn whose changes should be retained, including turns that
return `InputRequired`. `save_context` validates and serializes all marked
fields, including nested in-place mutations, and checks the loaded revision
to reject stale writes. Unmarked fields remain in memory.

In a managed transaction, `save_context` stages the field write for the
runtime's commit alongside conversation changes. Outside that transaction,
it commits a version-checked field write. An explicit save is required in
either case. The runtime owns connection and transaction cleanup.

A crash before commit loses uncommitted changes. On failure, reload the last
saved values. Save only changes the application intends to retain; placing
`save_context` in `finally` can persist incomplete work. Loading or saving
context does not preserve a Python call stack.

An external effect may succeed before its local result is committed. Stable
operation identities and idempotency/reconciliation are required; saving a
draft cannot make an external API atomic with AgentDB. See
[the AgentDB access contract](storage.md#application-records-and-context-fields).

Custom field storage adapters remain an advanced option. They must specify
ownership and commit behavior; external stores cannot be described as atomic
with execution storage without an actual protocol. The default persisted
field path requires no handwritten snapshots or repositories.

## How many drafts? Scope and cardinality

A field declares one value in one context scope. It does not declare a global
singleton for an Agent or a Python context class. Storage identity is:

```text
(thread/tenant boundary, context_scope_id, field_name)
```

The scope ID is durable, not an object address. Framework records preserve
which executions refer to it, including after worker restart.

| Operation | Context/data scope |
| --- | --- |
| Start a new conversation | New root scope |
| Handoff within an execution | Keep current compatible context and scope |
| `Call(child)` without a context override | Share current compatible context and scope |
| `Call(child, context=new_context)` | Allocate a fresh child context scope |
| Resume an existing execution | Restore its saved scope; never allocate a new one |

A new booking job should pass a fresh BookingContext when it needs a separate
draft. Shared context is intentional for specialists working on one case.
Identity/service references may be reused without sharing the draft:

```python
@front_desk.tool
async def start_booking(ctx: RunContext[CustomerContext]):
    app = ctx.context
    return Call(
        book_room,
        context=BookingContext(customer=app.customer, hotel=app.hotel),
    )
```

The child scope and initial values are allocated/persisted once with child
creation; a retried parent transition must not create a second job or scope.
No implicit copying/merging of application objects, cross-tenant sharing, or
reuse of a finished booking's draft. Any future explicit scope-sharing handle
needs separate authorization; field names do not grant access to another scope.

For multiple drafts in the same scope, declare a collection:

```python
@dataclass
class CaseContext:
    hotel: HotelServices
    drafts: dict[str, RoomDraft] = persisted(
        default_factory=dict,
    )


ctx.context.drafts[booking_id] = RoomDraft()
```

The application chooses booking_id. Collection keys and the field's storage
key are distinct. The runtime does not infer business cardinality or select
which booking is current. Large collections need storage limits or a repository
rather than unbounded context snapshots.

## Field names and schema changes

The Python field name is its storage name.
Two contexts with the same field name remain isolated by scope. Inherited or
shared-scope declarations must not assign incompatible contracts to the same
stored field. Collection entry keys remain application-chosen data, distinct
from the field name that identifies the collection.

Renaming a persisted field is a storage schema change. It requires an explicit
migration/alias policy; do not silently create an empty value under the new
name and abandon old saved data. The framework must compare schema manifests
before applying defaults to existing scopes.

Only marked fields are saved, even if unmarked values are JSON-compatible.
Fail serialization with a field path for unsupported values; never pickle
clients, silently drop marked fields, or persist credentials by convention.
Pydantic models, supported dataclasses, TypedDicts, and JSON collections need
defined codecs and validation. A missing required field blocks execution;
a declared default may supply a compatible newly added field. Shape and size
limits must be enforced before commit.

Store schema identity and field-name manifests with snapshots. Removing a field
or unmarking it must not silently destroy saved data. Defaults do not justify
silently accepting arbitrary incompatible changes. Retain old contracts long
enough to detect remove/reintroduce cases and validate pending outputs against
their saved receivers.

Runtime dependency recreation and schema evolution remain separate concerns.
Define a context factory/provider at the application boundary for service and
customer loading, followed by explicit load_context.

## Concurrency and execution ownership

Sharing a scope means sharing persisted values, not obtaining an independent
snapshot to mutate concurrently. Serialize stateful activations against that
scope and fence commits so parent/child updates cannot overwrite one another.
Within one activation, aliases referring to a shared scope must refer to the
same hydrated values. Independent background jobs should use separate scopes
or explicit application concurrency control. Their completion reaches the
parent as an event/result, not as a hidden draft mutation.

## Persistence boundaries

The application supplies a typed context object containing its data and runtime
services. Only fields marked `persisted(...)` are stored. Explicit loading
restores those fields; explicit `save_context` calls record their changes without serializing
service clients or other unmarked values.

Conversation history, application fields, and external effects have separate
lifecycles. Saving a draft does not record an external operation's outcome or
make that operation safe to repeat. Services must enforce idempotency and
reconcile uncertain outcomes. A durable execution runtime and external wake-up
recovery are deferred; context persistence does not provide them.
