# Application context and persisted fields

The target API follows OpenAI-style local context: `Agent[Context]`,
`run(context=...)`, and `ctx.context`. One typed object can contain identity,
mutable application data, services, and helper methods. There are no separate
public `deps_type` or `state_type` parameters.

Persistence is opt-in **per field**, not per entire context. Mark a user's
validated data structure with `persisted(...)`; explicitly load its saved
values and let managed saves checkpoint changes. Ordinary fields remain runtime dependencies/data.

This proposal covers conversation and draft persistence. A durable execution
engine, automatic task recovery, scheduler wake-ups, and operation journals
are deferred. Transaction and effect examples below explain persistence limits;
they do not require a new durable runtime in this phase.

This is the end-state contract. Current code still uses deps/state; see
[TODO.md](TODO.md). `persisted` and the context/scoping signatures below are
proposed API, not implemented imports. The dataclass field helper and typed
Agent/output inference need strict type tests before implementation lands.

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

| Value | Saved automatically? | How it is restored |
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

No snapshot class, repository, serializer, or save call is required for the
marked draft. Customer information may be refreshed from its authoritative
source instead of becoming an obsolete saved copy. Stable principal/tenant
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

This entrypoint is proposed syntax; ctx.store and its context methods are not
implemented yet. customers/hotel_service are application dependencies, and
ctx.customer_id represents trusted identity metadata, not a currently exposed
A2ASessionContext field.

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

Keep canonical methods on the store. There is no implicit global connection
or second entrypoint-specific persistence policy. Tools, scheduled wake-ups,
and tests receive an appropriately scoped store from their execution host;
they need no fake HTTP entrypoint to load/save context. Manually opening a
connection remains low-level infrastructure, not the ordinary user API.

A store view is activation-scoped and must not be retained in persisted
context or reused after its runtime ends. A child receiver uses the child's
recorded scope; a handoff or shared-context child retains the existing one.
Returning from the handler lets the runtime finish its commit and release
resources; users do not manage that lifecycle with try/finally.

## Automatic checkpoints are the safety net

Explicit loading makes the hydration boundary visible. After a context has
been loaded/enrolled in a durable execution, the runtime saves its marked
fields automatically at defined execution checkpoints as well as accepting
explicit save_context requests. Standalone agent.run is not automatically
durable merely because a context contains markers; it needs a managed
execution/store association. A background wake-up uses the registered
application context factory followed by the same load operation.

For a managed activation:

1. Obtain the application's fresh runtime context and explicitly load the
   persisted fields for the recorded scope.
2. Validate all marked values before instructions, tools, or output functions.
3. Run the model/tool activation.
4. Checkpoint marked fields with settled tool results, history, consumed input,
   pending receivers, status, and delivery outbox records.

Checkpoint before releasing execution for human input, an operation, or a
child, and on accepted final output. Capture nested in-place mutations;
correctness cannot depend solely on observing attribute setters.

During an active run transaction, save_context participates in that unit of
work; it must not independently commit a new draft while leaving its tool
result/history behind. A call from an unfinished tool is staged or rejected
if no coherent checkpoint can be formed; it never promises to persist a
Python program counter. Outside a managed run, save_context performs an
explicit version-checked field transaction. The API must expose/document
transaction ownership; an independent manual save is not a full conversation
checkpoint. The final save in an entrypoint may be idempotent/redundant after
an automatic checkpoint, but must not overwrite a newer revision.

On an uncommitted failure, discard local mutations and reload the last
checkpoint. Do not automatically commit arbitrary partial state in except
or finally. A hard process kill may execute neither block. Explicit saves
and automatic checkpoints do not make every assignment durable immediately.
Services are never restored from JSON; constructors/default factories must
not perform business effects.

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

The Python field name is its storage name. There is no separate key argument.
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
their saved receivers. No user-written version number is required, and no
automatic migration engine is assumed to exist today.

Runtime dependency recreation and schema evolution remain separate concerns.
Define a context factory/provider at the application boundary for service and
customer loading, followed by explicit load_context; this does not require
handwritten draft snapshot functions.
The exact registration and default-field discovery syntax is implementation
work, including full support for context subclasses and alternate model types.

## Concurrency and execution ownership

Sharing a scope means sharing persisted values, not obtaining an independent
snapshot to mutate concurrently. Serialize stateful activations against that
scope and fence commits so parent/child updates cannot overwrite one another.
Within one activation, aliases referring to a shared scope must refer to the
same hydrated values. Independent background jobs should use separate scopes
or explicit application concurrency control. Their completion reaches the
parent as an event/result, not as a hidden draft mutation.

## Framework precedent and LiveKit extension

[OpenAI local context](https://openai.github.io/openai-agents-python/context/)
combines application data and dependencies; its
[RunState serialization hooks](https://openai.github.io/openai-agents-python/human_in_the_loop/#serialization-options)
allow explicit context serialization and reconstruction. Automatic per-field
`persisted`, field-based storage names/scopes, atomic draft checkpoints, and automatic
conversational InputRequired are LiveKit extensions, not OpenAI guarantees.
History persistence, settled tool checkpoints, and safe external-effect
recovery have different contracts; saving a draft alone supplies none of the
other two. A durable execution runtime and external wake-up recovery are
deferred; persistence here does not promise them.
