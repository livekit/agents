# Storage

> Context persistence: design. See [application context](state-and-deps.md)
> for field storage and scopes. The tables below describe current storage.

Every thread has its own database: the history, the journal, the typed
state and the tasks, in one place the framework loads before a run and
commits after it. Nothing lives in the process between runs, so any worker
can take the next message, and a crash loses at most the run in flight.

## Application records and context fields

Draft persistence and direct application writes can use the same AgentDB.
Choose the access pattern according to the data and its commit requirements:

| Access | Purpose | Commit behavior |
| --- | --- | --- |
| persisted context fields | Save marked application data with explicit `save_context` | Join the managed commit, or commit directly outside a managed transaction |
| Explicit application SQL/transactions | Business records, unique operation IDs, invoices, outbox entries | Commit at application-selected durable boundaries |

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

The scoped store provides context load/save and application database access.
Application transactions declare whether they join the managed commit or
commit independently. Framework-owned history, state, task, and event tables
remain protected from application writes.

Do not place raw SQL inside the draft or assume that an arbitrary application
write automatically joins the turn's transaction. Expose whether an operation
participates in the managed unit of work or creates an independently committed
transaction. Protect framework-owned schema/tables from accidental application
DDL and writes through the application interface.

For a payment, an independently committed preparation transaction is essential:

```text
AgentDB transaction: create/retrieve unique pending charge operation
  -> commit before dispatch
Payment provider: charge exact saved terms using operation ID as idempotency key
AgentDB transaction: record outcome and related business changes
  -> draft/checkpoint references that authoritative outcome
```

If preparation were staged until the end-of-turn commit, a charge could happen
before its operation record existed durably. Explicit transaction completion
must therefore mean durable commit before returning. Do not hold an open SQL
transaction across network calls to a payment provider. Async dispatch can use
an outbox committed with the operation record.

After a provider timeout or crash between effect and outcome commit, reconcile
the same operation; absence of a result is not proof no charge happened. Domain
uniqueness constraints guard repeated semantic requests even when the LLM emits
another tool call ID. Persisted context fields are working data, not a
replacement for those constraints. Both can share one AgentDB while keeping
separate commit boundaries.

## What is saved

The database holds one lane per agent that took part (see
[lanes.md](lanes.md)) and one journal for the thread.

|         |                                                                                               | Owned by                                                                       |
| ------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| history | per lane: every message, tool call and result, item by item                                    | the developer: truncating, inserting a summary, deleting an item are row edits |
| state   | per lane: the typed object, written by a run that changed it                                   | the agent's tools                                                              |
| agent stack | per lane: who is speaking, as an agent name or an `AgentTask` class with its arguments     | the framework                                                                  |
| tasks   | per lane: one row per turn and per background task: status, question, result, its own history | the framework                                                                  |
| inbox   | per lane: messages received while a turn ran, under the `batch` policy, until a turn drains them | the framework                                                               |
| journal | per thread: every event, append-only; see [journal.md](journal.md)                            | the framework                                                                  |

What is not saved: `deps`, by construction; media; and nothing from memory,
since no frame or local variable survives a run. Documents and blobs are a
later addition.

Payloads are JSON, the same form the A2A wire carries, so the rows are
queryable in place and a schema change is an added key. State is
JSON-shaped and under 64 KiB: a field that cannot be written as JSON raises
`NotSerializable` naming it, a larger state raises `TooLarge`.

## The tables

The schema is livekit/protocol `livekit/agent/thread_schema_v1.sql`, which
`livekit-protocol` carries as `agent::THREAD_SCHEMA_V1`; the framework runs it
on a database's first write. JSONB columns hold JSON. The file
tags each column with who reads it: the platform (the threads service's
GetThreadItems and GetThreadTasks), the A2A methods, or the framework.
`seq` is a position within a lane; `pos` on `items` and `tasks` is the
insertion order, never reused, the platform's page cursor; `id` on
`events` and `inbox` is the insertion order.

| Table            | One row per                  | Columns |
| ---------------- | ---------------------------- | ------- |
| `schema_version` | database                     | `version` |
| `items`          | history item                 | `pos`, `id` (the item's id), `lane`, `seq`, `message_id`, `payload`: the item as `livekit.agent.ChatContext.ChatItem` protobuf JSON, which the threads service reads |
| `tasks`          | A2A task: a turn or an `AgentTask` | `pos`, `task_id`, `lane`, `seq`, `name` (the `AgentTask` class, empty for a turn), `message_id`, `status`, `status_message`, `updated_at`, `data`, `internal` |
| `lanes`          | lane                         | `lane`, `version`, `state`, `agent_stack`, `lease_holder`, `lease_token`, `lease_until`, `waiter`, `drained_id` (the last inbox `id` a run consumed) |
| `events`         | journal entry                | `id`, `ts_ms`, `lane`, `run_id`, `type`, `payload` |
| `inbox`          | message received during a run | `id`, `lane`, `task_id`, `message_id`, `message`, `received_at` |

An item's `payload` keeps what the model needs to see the conversation
again: messages and their text, function calls with their `extra`, function
call outputs, handoffs. It does not keep image or audio content, `group_id`,
a tool output's `reply_required`, or the message ids an inbox batch merged
into one user message (those are on the tasks, as `livekit/mergedInto`). A
row in any other shape fails the load.

A task row is the A2A `Task`: `id` is `task_id`, `contextId` the thread id,
`status` is `status`, `status_message` and `updated_at`, and `data` holds
the rest under the A2A names (`history`, `artifacts`, `metadata`).
`internal` is the framework's own: the `AgentTask` arguments, the pending question
(`ask`), `deadline` and `parent`. A row found `working` when a new run picks
it up belonged to a worker that died.

## The backend

```python
from livekit.agents.storage import AgentDBStorage

server = AgentServer(storage=AgentDBStorage())
```

`AgentDBStorage` keeps threads on the platform's agentstore: agent-db for
the thread's database and the answer streams, and the threads service that
mints thread ids (`AT_`). There is one backend. Local development and the
tests use the developer's own staging project; nothing stands in for it. `a2a_session(storage=)` overrides the
server's choice for one text door. A call uses the server's storage and
receives its thread as `ctx.thread`.

Storage only adds durability. A server with no storage still speaks
standard A2A: every message still gets an A2A task id, and `ctx.thread` is
an in-memory thread with the same API that nothing persists, so it ends
with the process.

| Argument                 | Environment variable when omitted                                         |
| ------------------------ | ------------------------------------------------------------------------- |
| `api_key`, `api_secret`  | `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`                                    |
| `lease_ttl`              | `LIVEKIT_AGENTDB_LEASE_TTL` in seconds, else 10 seconds                    |
| `renew_interval`         | `LIVEKIT_AGENTDB_RENEW_INTERVAL` in seconds, else 3 seconds                |
| `holder`                 | this worker's lease holder id, `<hostname>-<pid>` by default               |

## Where it connects

Every worker, LiveKit Cloud's included, reaches agentstore only through
LiveKit's public API, at one base URL derived from `LIVEKIT_URL`
(`wss://<project>.staging.livekit.cloud` uses
`https://beta-api.staging.livekit.cloud`, `wss://<project>.livekit.cloud`
`https://beta-api.livekit.cloud`).

| Service | Under the base URL |
| --- | --- |
| the data plane, a WebSocket | `/v0/agentstore/db` |
| the answer streams | `/v0/agentstore/streams` |
| threads, `ThreadService` | `/v0/projects/p_/threads` |
| databases, `DatabaseService` | `/v0/projects/p_/databases` |

`p_` is the project of the request's token. The worker mints its own tokens
from the project's key and secret: the data plane and the streams take an
`agent.databaseAdmin` token, one hour, minted per dial; the threads and
databases services a short-lived one per request. HTTP requests carry it as
`Authorization: Bearer`; the data plane carries it as `access_token` on
the WebSocket upgrade, so a proxy admits the connection before the first
frame, and again in its `Hello`.

`CreateThread` creates the thread's database, so every thread has one
from its creation. The framework does not touch a thread: its expiry moves
with `UpdateThread` and its `idle_ttl`.

## The unit of work

A run is a unit of work: everything it reads and writes is committed
together when it ends, or not at all.

- **Load.** Taking the lane's lease loads its history, state, agent stack and
  journal events in one round trip. The whole history is loaded; compaction, when a
  conversation outgrows a context window, sits in front of the run and is
  not part of storage.
- **Commit.** Leaving the run writes the history, the state when it changed,
  the agent stack, the events, the task rows and the usage in one write,
  fenced on the version loaded and the lease token, and releases the lease.
  A run that lost its lease meanwhile gets `VersionConflict` and writes
  nothing.
- **Fail.** A run that raises writes only its `run_ended` event and
  releases the lease.
- **A failed save** ends the task `failed` with `livekit/reason`
  `storage-unavailable`. Nothing of the run was written, and resending the
  message runs it again.
- **No double answers.** A message is identified by its `messageId`, unique
  per lane: a resend returns the task the first send created, finished or
  still running, and runs nothing new. A run id that already committed
  yields a closed run that writes nothing.
- **An early row.** Each message's task row is written `submitted` before
  its run waits for the lease, so `tasks/get` from any worker finds it at
  once; it goes `working` when the run takes the lane and reaches its
  terminal state in the commit.
- **A dead worker.** A run whose worker died is not run again: its task
  fails `worker-lost` when the next holder finds it, and the client resends.

Durable execution, operation outboxes, and durable timers are deferred.
Context persistence remains in scope; see [state-and-deps.md](state-and-deps.md).
