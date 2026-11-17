# Application context and persisted fields

Declare the application context type with `Agent(context_type=Context)`, pass an instance
with `run(context=...)`, and access it in tools through `ctx.context`.
One typed object contains identity, mutable application data, services, and
helper methods.

Persistence is opt-in **per field**, not per entire context. The context is a
dataclass instance; mark a user's validated data structure with
`persisted(...)`, explicitly load its saved values with `load_context` and save
changes with `save_context`. Ordinary fields remain runtime dependencies/data.
The [persistence guide](persistence.md) owns loading, saving, session binding,
snapshots, transactions and durability.

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


book_room = Agent(
    context_type=BookingContext,
    name="book-room",
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
| Current agent, private histories, local and remote work | Through a conversation binding | See [session persistence](persistence.md#session-state-and-explicit-binding) |
| A2A transport history and task records | Separately, by the runtime | Load the thread records |

The application constructs runtime context as usual:

```python
context = BookingContext(
    customer=await customers.get(authenticated_customer_id),
    hotel=hotel_services,
)
```

Call `save_context` to save the marked draft. Refresh customer information
from its authoritative source for each activation. Resolve the authenticated principal and context scope from the incoming
request and saved routing references before loading application data.

`RunContext[BookingContext]` is the framework wrapper. Only `ctx.context` is
the application object; the other members (execution and task IDs, the thread,
usage, the call being executed, progress updates and live-session access) are
runtime metadata, listed in [RunContext](agents.md#runcontext). Local context is not
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

These preferences belong to the conversation's context, which a child agent
called as a tool shares.
The customer's account profile remains authoritative in the
customer service/database; conversation preferences do not silently update
it. A child's final Booking result lets the parent update its own
preferences.last_booking_code explicitly. State is not merged on child exit.

Session state is separate from application data. TextSession coordinates
[agent contexts and routing](text-session.md#shared-state-and-private-histories);
[persistence](persistence.md#session-state-and-explicit-binding) defines how that
state is bound or saved. Every local agent reads the context supplied to the turn.
History, preferences, draft, and customer profile are not interchangeable
forms of memory.

A child can start with fresh or forked history. On completion, the parent can
receive its typed result alone or the result plus visible dialogue. These
history choices do not share or merge application context fields; context
scopes retain their explicit ownership. See the [history policies](text-session.md#local-agents-as-tools).

## How many drafts? Scope and cardinality

A field declares one value in one context scope. It does not declare a global
singleton for an Agent or a Python context class. Storage identity is:

```text
(thread/tenant boundary, context_scope_id, field_name)
```

The scope ID identifies stored application data.

| Operation | Context/data scope |
| --- | --- |
| Start a new conversation | New root scope |
| Call an agent as a tool | Share the parent's context and scope |
| Run a child from a tool with `child.run(..., context=...)` | The context and scope the tool supplies |
| Resume an existing execution | Restore its saved scope; never allocate a new one |

An agent called as a tool works on its parent's case, so it shares the parent's
context. A specialist that needs a separate draft is called from an ordinary
tool that builds its context, reusing identity and service references without
sharing the draft:

```python
@front_desk.tool
async def book(ctx: RunContext[CaseContext], request: str) -> Booking:
    """Book a room for the guest."""
    booking_context = BookingContext(customer=ctx.context.customer, hotel=ctx.context.hotel)
    result = await booking_agent.run(request, context=booking_context)
    return result.output
```

The tool explicitly loads and saves the child context's marked fields.
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
