# Persistence

Application fields, session state and transport history have different save
boundaries. This page owns their shared public persistence contract. Define
application data and runtime dependencies with [context](context.md); choose
[TextSession](text-session.md) or [VoiceSession](voice-session.md) for execution.
Database runs, leases, reconciliation and write receipts are described in
[storage internals](internals/storage.md).

| Data | How it is saved |
| --- | --- |
| Marked application fields | Explicit `save_context`; binding does not save ordinary field mutations |
| Bound session state | Incremental conversation checkpoints during execution |
| Detached session snapshots | Assign an inspection snapshot to a marked field and explicitly save it |
| A2A transport history and task records | The managed endpoint's checkpoints and final commit |
| External business effects | The application's authoritative service and idempotency protocol |

A settled agent result and a successful database commit are separate outcomes.
A checkpoint can be durable before the enclosing turn commits its application
fields. Loading saved data reconstructs state between invocations; it does not
resume Python stacks, live media or an external operation.
The framework provides no durable execution runtime or external wake-up recovery.

## Explicit load and save on the store

The runtime supplies `ctx.store`, already bound to the current thread,
execution, context scope, and transaction ownership. It opens/releases the
connection and manages leases; application code does not open storage, close
connections, or pass thread/scope IDs for ordinary context persistence.
Construct the application context explicitly, then load its marked fields.

`BookingContext` and `book_room` are defined in the [context example](context.md#a-complete-context).
`customers` and `hotel_service` are application dependencies.
`customer_id` is resolved by application authentication before constructing the context.

```python
@server.a2a_session(endpoint="booking", description="Book a hotel room.")
async def booking(ctx: A2ASessionContext):
    store = ctx.store

    app = BookingContext(
        customer=await customers.get(customer_id),
        hotel=hotel_service,
    )
    await store.load_context(app)

    result = await book_room.run(
        ctx.message, message_history=ctx.message_history, context=app
    )

    await store.save_context(app)
    return result
```

`load_context` mutates only marked fields and returns the same typed object
for callers preferring assignment. Existing saved values replace defaults
before the agent runs; a new scope uses declared defaults. Runtime customer
and service fields are untouched. The store tracks the loaded revision so
`save_context` can reject stale writes; saving an object that store did not load
raises `ValueError`. Framework scope/revision metadata is not serialized as
application fields.

Storage failures raise a `StorageError` from `livekit.agents.storage`:
`VersionConflict` (a stale save), `LeaseLost`, `ThreadNotFound`, `TooLarge`,
`NotSerializable`, `StorageUnavailable` (retryable: the database was unreachable or
busy) or `RunRejected` (a `reject` partition refused the run; `running_task_id` names the
running turn).

Top-level helpers are equivalent conveniences using the supplied store:

```python
await load_context(app, store=ctx.store)
await save_context(app, store=ctx.store)
```

Use the scoped store for both methods. Entrypoints, tools, and tests receive
the store from their execution host. The runtime manages its connection.

A store view is activation-scoped and must not be retained in persisted
context or reused after its runtime ends. A child agent called as a tool uses
its parent's store.
Returning from the handler lets the runtime finish its commit and release
resources.

## Explicit saves

Mutating a marked application field changes only the in-memory object until
`save_context` is called. Returning from `agent.run()` or a handler does not save
those mutations. Streaming uses the same boundary: `stream.result()` settles
the invocation, not the surrounding handler transaction. Bound session state
checkpoints automatically. Assigning an inspection snapshot to an application field requires
an explicit save, like any other field assignment.
See [stream delivery](agents.md#managed-endpoints-and-persistence) for forwarding
text through an endpoint.

For each activation:

1. Construct the context with fresh runtime dependencies.
2. Call `await store.load_context(app)` to restore and validate marked fields.
3. Run the agent with `context=app`.
4. Call `await store.save_context(app)` to save the resulting application data.

Save after each turn whose changes should be retained, including turns that
have `status="input_required"`. `save_context` validates and serializes all marked
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
[the AgentDB access contract](#application-records-and-transactions).

Custom field storage adapters remain an advanced option. They must specify
ownership and commit behavior; external stores cannot be described as atomic
with execution storage without an actual protocol. The default persisted
field path requires no handwritten snapshots or repositories.

## Session state and explicit binding

Agent.run, TextSession and VoiceSession accept an application context with an optional persisted
SessionState field:

```python
from dataclasses import dataclass
from livekit.agents import SessionState, TextSession, persisted

@dataclass
class App:
    services: Services
    session: SessionState = persisted(default_factory=SessionState)
    draft: Draft = persisted(default_factory=Draft)

app = App(services=services)
await ctx.store.load_context(app)
binding = ctx.store.bind(app)
result = await TextSession(agent=main).run(ctx.message, context=binding)
```

`ctx.store.bind(app)` declares that session execution manages the selected conversation.
Constructing the declaration performs no I/O and changes no application fields. Execution
activates it, keeps the same app.session object current, checkpoints conversation progress,
and unbinds during cleanup. Tools receive the original app through ctx.context.

Bound `agent.run(...)` and `agent.run_stream(...)` use the same native conversation owner as
TextSession. Pass `context=ctx.store.bind(app)` to continue it; a bound run rejects competing
explicit `message_history`. `Agent.run()` still returns its invocation's RunResult.

Inspect `app.session.current_agent.chat_ctx` for the current agent's private model context and
`app.session.root_agent` for the top-level agent. `app.session.chat_history` contains accepted
human input and assistant content actually delivered across local agent changes. Private tools,
instructions and synthetic child tasks stay out of that delivered history. These are generated
read-only views; `snapshot()` captures a detached view without saving it. Root and current are
None before the conversation starts.

Use `ctx.store.bind()` when execution has no application context. A `thread=`
association alone does not enable VoiceSession persistence.

The marked field is discovered by its SessionState type, not its name. Without such a field,
bound sessions restore their own state internally. A declared field must be hydrated through
the same store before execution. Ordinary application fields remain explicitly saved with
save_context; binding does not automatically save draft changes.

Plain `context=app` supplies application dependencies to a fresh run. It does not discover
or restore app.session. Use store binding for every conversation continuation.

`await ctx.store.load_session()` returns an immutable detached read of committed state.
It never binds and never aliases app.session. Bound state can lead stored state until a
checkpoint completes. `await ctx.store.flush()` waits for accepted conversation checkpoints;
it does not commit ordinary application saves staged for the managed turn.

SessionState.snapshot() returns a detached inspection view. It is not an execution input:
Agent.run, TextSession and VoiceSession continue the selected conversation through store.bind().
TextSession returns RunResult directly, like Agent.run; there is no result/state envelope or
state= continuation argument. Runtime services, Python stacks and media connections are not
serialized. Storage uses one shared conversation encoding.
Durable local Agent-as-tool continuations run in text. Voice handoffs and VoiceTasks remain
transient; starting voice with unresolved text-local agent work is refused without resetting it.
Closing voice releases its binding while preserving an application-supplied ContextStore for
subsequent execution.
See [TextSession](text-session.md#running-a-conversation) and
[voice persistence](voice-session.md#the-calls-thread).

### Inspecting remote communication

`app.session.remote_agents` exposes retained communication with each remote. Its `messages`
index contains observed A2A messages in both directions, including direct replies without a
task or context. Its `tasks` index contains task views; each view's `task` is the upstream SDK
Task, with its observed status, artifacts and optional history. `input_requests` and
`cancel_requested` describe local task handling separately from that protocol status.

```python
remote = app.session.remote_agents[remote_id]
task = remote.tasks[task_id]
observed_status = task.task.status
events = remote.events
context = remote.contexts[task.task.context_id]
```

`contexts` groups the same messages and tasks by their context IDs. It does not maintain a
second message history. `events` retains requests, responses and observed errors in local
journal order. This communication journal is separate from the person's `chat_history`:
receiving remote text does not mean it was delivered to the person.

Inspection performs no network calls and describes retained observations, which may be
partial. A task can exist without a full history or status snapshot. Collections and SDK
values returned to Python are copies; editing them does not change the conversation. Retained
remote/task views follow their bound SessionState. Views obtained through `state.snapshot()`
stay detached and fixed.

### Resuming a remote call

When an unresolved remote-tool call resumes, the native owner finds its latest outgoing
request by the originating `ToolCallRef`. Another local call does not consume that receipt.
The saved endpoint remains pinned to the original destination.

An input-required observation is saved before its question is presented. Its exact input
occurrence travels with the response; sending reserves that occurrence and the outgoing
message together. A newer question cannot silently redirect an older answer. If delivery is
uncertain, the same response identity remains available for reconciliation.
Observing the same question without evidence that the earlier response arrived keeps that
response uncertain; it does not authorize another answer.

- An intent saved before dispatch can be sent with its original SDK body, method, request ID
  and message ID.
- A request that may have been sent and has a known task is reconciled with `GetTask`.
  Observation then continues where the peer supports it. The business message is not resent.
- An accepted message-only reply is returned from saved state without another exchange.
- A potentially sent request with no accepted task or reply remains uncertain. Recovery
  reports that uncertainty instead of guessing an outcome or starting the work again.

Protocol content rejected during validation stays in the journal for diagnosis; it cannot
settle a restored call. A snapshot or resubscription cannot promise every missed progress
update was recovered. Restoring a call also does not restore a Python stack, cancel its remote
task, or prove that an earlier spoken result was heard.

## Application records and transactions

Draft persistence and direct application writes can use the same AgentDB.
Choose the access pattern according to the data and its commit requirements:

| Access | Purpose | Commit behavior |
| --- | --- | --- |
| persisted context fields | Save marked application data with explicit `save_context` | Join the managed commit, or commit directly outside a managed transaction |
| Explicit application SQL/transactions | Business records, unique operation IDs, invoices, outbox entries | `transaction()` joins the managed commit; `independent_transaction()` commits on block exit |

Use the same thread database when the data belongs to that thread. A
customer-wide invoice or global room inventory may require a different
application/database scope; thread deletion/expiry must not accidentally
remove records whose retention or ownership outlives the conversation.
Sharing AgentDB does not imply all business records belong in one thread.

The runtime opens storage internally and supplies a scoped handle:

```python
store = ctx.store
await store.load_context(app)
# Run agent/tools with the loaded application context.
await store.save_context(app)
```

Connection open/close, thread/scope resolution, and lease ownership are
framework responsibilities.

The scoped store provides context load/save and application database access
through two transaction forms:

```python
async with ctx.store.transaction() as tx:   # joins the turn's managed commit
    await tx.execute(
        "INSERT INTO notes (booking_id, body) VALUES (?, ?)", (booking_id, body)
    )

async with ctx.store.independent_transaction() as tx:   # durable when the block exits
    rows = await tx.fetch("SELECT id, status FROM charges WHERE booking_id = ?", (booking_id,))
    charge = await tx.fetch_one("SELECT * FROM charges WHERE id = ?", (charge_id,))
```

`tx` is a `Transaction`:

| Method | Returns |
| --- | --- |
| `await tx.execute(sql, params)` | nothing; runs one statement |
| `await tx.fetch(sql, params)` | `list[Row]` |
| `await tx.fetch_one(sql, params)` | one `Row`, or `None` |

A `Row` (`list[SqlValue]`, both from `livekit.agents.storage`) is a list of
column values in select order: `None`, numbers or strings (a boolean reads back
as 0 or 1); parameters are `SqlValue`s: `None`, booleans, numbers or strings. There are no BLOBs: a BLOB column
read as a row value is refused, so select `hex(...)` instead. Reads observe committed records and writes
apply together when the transaction commits, so a transaction reads before
its first write; a read after a write is refused.

Writes in `transaction()` commit or roll back with the turn, like a nested
savepoint. Outside a managed turn, `transaction()` behaves like
`independent_transaction()`. `independent_transaction()` commits when its
block exits, whatever the turn does next. Application SQL is unrestricted: it may read and write any
table, including the framework's history, state, task and event tables;
keeping those consistent is then the application's responsibility. Do not
place raw SQL inside the draft.

For a payment, an independently committed preparation transaction is essential:

```text
independent_transaction(): create/retrieve unique pending charge operation
  -> commit before dispatch
Payment provider: charge exact saved terms using operation ID as idempotency key
independent_transaction(): record outcome and related business changes
  -> draft/checkpoint references that authoritative outcome
```

If preparation were staged until the end-of-turn commit, a charge could happen
before its operation record existed durably, so preparation uses
`independent_transaction()`, durable on block exit. Do not hold an open SQL
transaction across network calls to a payment provider. Async dispatch can use
an outbox committed with the operation record.

After a provider timeout or crash between effect and outcome commit, reconcile
the same operation; absence of a result is not proof no charge happened. Domain
uniqueness constraints guard repeated semantic requests even when the LLM emits
another tool call ID. Persisted context fields are working data, not a
replacement for those constraints. Both can share one AgentDB while keeping
separate commit boundaries.

## Concurrency and execution ownership

Sharing a scope means sharing persisted values, not obtaining an independent
snapshot to mutate concurrently. Serialize stateful activations against that
scope and fence commits so parent/child updates cannot overwrite one another.
Within one activation, aliases referring to a shared scope must refer to the
same hydrated values. Independent background jobs should use separate scopes
or explicit application concurrency control. Their completion reaches the
parent as an event/result, not as a hidden draft mutation.
