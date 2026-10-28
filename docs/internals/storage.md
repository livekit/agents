# Storage

Every thread has its own database containing transport history, the journal,
A2A tasks, and explicitly saved application fields. The framework loads and
commits transport records around each managed turn. Applications explicitly
load and save persisted context and routing state. A new worker continues
from committed records; in-memory changes are lost on a crash.

Application-facing field saves and independently committed business records
follow the [context transaction contract](../context.md#application-records-and-transactions).
The storage sections below define how managed transport writes are implemented.

## Configuration

`Storage` is a host-neutral Rust trait, exposed as the public abstract storage
type accepted by `AgentServer(storage=)` and `a2a_session(storage=)`.
`AgentDBStorage` and `SQLiteStorage` implement the same contract. Handlers,
thread handles, and context stores depend on that contract rather than a
concrete backend. Python classes configure and wrap Rust implementations;
Python inheritance is not the backend extension mechanism.

```python
from livekit.agents.storage import AgentDBStorage, SQLiteStorage

# LiveKit Cloud.
server = AgentServer(storage=AgentDBStorage())

# Local development or self-hosting, with persistence on this machine.
server = AgentServer(storage=SQLiteStorage(directory=".livekit/storage"))
```

`a2a_session(storage=)` overrides the server's choice for one text door. A
call uses the server's storage and receives its thread as `ctx.thread`.
Participants sharing a thread must use the same storage namespace: the same
Cloud project or the same local directory. A local thread id does not identify
a thread in Cloud storage.

Storage only adds durability. A server with no storage still speaks
standard A2A: every message still gets an A2A task id, and `ctx.thread` is
an in-memory thread with the same API that nothing persists, so it ends
with the process.

## Rust interface

Storage follows the [Rust plan](rust-core.md#storage-and-a2a).

The engine has three layers:

| Layer | Responsibility |
| --- | --- |
| `Storage` | Thread lifecycle and opening a thread; implemented by AgentDB and local SQLite backends |
| Shared `Thread`, `Lane`, `Run` and `Lease` | Typed reads, submission/deduplication, lane acquisition, inbox policy, fencing and managed commit decisions |
| Internal `SqliteConnection` | Execute the shared engine's reads and atomic statement batches against Cloud or local SQLite |

The pure storage core decides plans and outcomes. Runtime adapters execute I/O,
schedule renewals, and own pools and resources. The backends share the schema
and turn state machine. Host-visible records and bindings are generated from
Rust declarations through the ABI machinery.

The internal adapter deliberately targets SQLite semantics. A future non-SQL
backend could implement the public storage contract with a different engine,
but would have to satisfy the same thread, read and managed-turn guarantees.
The SQL adapter does not promise compatibility with arbitrary databases.

### Thread lifecycle

The server-facing `Storage` trait opens shared engine handles:

```rust
#[async_trait]
pub trait Storage: Send + Sync {
    async fn create_thread(&self, request: CreateThread) -> Result<ThreadInfo, StorageError>;
    async fn get_thread(&self, id: &ThreadId) -> Result<ThreadInfo, StorageError>;
    async fn list_threads(&self, query: ThreadQuery) -> Result<ThreadPage, StorageError>;
    async fn update_thread(&self, request: UpdateThread) -> Result<ThreadInfo, StorageError>;
    async fn delete_thread(&self, id: &ThreadId) -> Result<(), StorageError>;
    async fn open(&self, id: &ThreadId, holder: LeaseHolder) -> Result<Thread, StorageError>;
    async fn close(&self) -> Result<(), StorageError>;
}
```

| Operation | Contract |
| --- | --- |
| `create_thread` | Accept optional scope, attributes and idle TTL; mint an `AT_` id and return only after its metadata and database exist. A partially created thread must not become usable. |
| `get_thread` | Return thread metadata; unknown, deleted or expired threads fail with `ThreadNotFound`. |
| `list_threads` | Page thread metadata by the supported scope and cursor, with the same ordering and expiry filtering on both backends. |
| `update_thread` | Apply explicit attribute and idle-TTL updates and return the updated metadata. This is the activity boundary that moves expiry. |
| `delete_thread` | Make the thread unavailable, coordinate with active handles, and reclaim its metadata and database; repeating deletion is safe. |
| `open` | Resolve an existing thread and return the shared engine's concrete `Thread` handle, bound to a worker holder. Opening alone does not acquire a lane lease. |
| `close` | Release backend-owned resources after active handles have settled; repeated close is safe. |

Thread creation/deletion spans registry and database resources. Backends must
recover partial lifecycle operations; the trait does not imply a SQL transaction
can span a Cloud service and a thread database. Expiry uses the same visibility
and deletion rules. Cloud runs its service sweeper; SQLite checks expiry on
access and performs local cleanup without requiring a running worker to make
expired threads invisible.

### SQLite execution

`SqliteConnection` is internal to the Rust runtime. AgentDB executes over its
WebSocket data plane; local SQLite executes against a file without blocking the
async executor. Statements, bound values and results are independent of either
backend's wire format or driver types.

```rust
#[async_trait]
pub trait SqliteConnection: Send + Sync {
    async fn query(&self, statement: Statement) -> Result<QueryResult, StorageError>;
    async fn batch(&self, statements: Vec<Statement>) -> Result<BatchResult, StorageError>;
    async fn close(&self) -> Result<(), StorageError>;
}
```

| Operation | Contract |
| --- | --- |
| `query` | Execute a read and return typed columns and rows, without changing the database. |
| `batch` | Execute ordered statements in one atomic transaction, returning each statement's rows, including `RETURNING`, and affected-row count. Success means the transaction committed. |
| `close` | Release this connection; repeated close is safe. Runtime cleanup also handles cancellation and exceptional exit. |

Connections preserve the distinction between a rejected write and a commit
whose acknowledgment was lost. A timeout or disconnect must not be reported as
proof of rollback. The engine applies the existing run-id idempotency and
version/fence rules when resolving uncertain outcomes. Both backends enforce
the same schema compatibility, serialization limits and storage error contract.

## Reading thread items and tasks

Opening a thread returns a handle with typed content reads. These are shared
engine operations implemented using `SqliteConnection`; callers do not write
SQL and backends do not implement a separate history/task repository.

| Thread operation | Returns |
| --- | --- |
| `list_items(query)` | A page of chat items with an optional lane filter, an opaque continuation cursor and a bounded limit |
| `list_tasks(query)` | A page of task records with lane/status filters and explicit paging and ordering for the requested listing |
| `task(task_id)` | One task record, or no record if it does not exist in this thread |
| `journal(after_id, limit)` | A page of journal events for authorized runtime/operations callers |

Content reads require neither an active agent run nor a lane lease. A backend
can open a thread solely for reading. Reads observe committed records; staged
context writes and in-flight history are not exposed. The worker holder passed
to `open` identifies a potential writer and does not grant write ownership.

Item pagination uses insertion position (`items.pos`), with an optional lane
filter. Task queries distinguish insertion-order thread browsing (`tasks.pos`)
from A2A `tasks/list`'s last-updated ordering. Each listing has a stable
tie-breaker and an opaque cursor bound to its filters/order; a cursor is not
an offset, and paging across calls does not promise a frozen snapshot while
other runs commit. Request/response records and their host bindings are
declared in Rust.

`GetThreadItems` and `GetThreadTasks` are the Cloud threads service's external
API operations. They remain available with their existing authorization and
wire contracts; they project these same stored item/task records. They are
distinct from the A2A endpoint's task listing.

### Frontend access

Backend read capabilities are shared; a hosted browser-facing API is a Cloud
capability. With AgentDB, a frontend can call `GetThreadItems` and
`GetThreadTasks` through the threads service to render a conversation, using
an authorized thread read grant. Project API secrets stay on the backend.

| Caller | AgentDBStorage | SQLiteStorage |
| --- | --- | --- |
| Worker or application backend | Typed reads through the thread handle | The same typed reads against local files |
| Browser frontend | Cloud threads API with an authorized read grant | An application-provided HTTP API that calls the thread handle |

Local SQLite files live on the worker's machine. The browser cannot use the
Cloud threads API to read those files, and a local thread id is not registered
in Cloud. To render a local conversation, the application serves authorized
read routes from a process that can access the same storage directory. Those
routes validate caller access and return item/task pages from the thread
handle. SQLiteStorage does not automatically expose HTTP routes or provide
Cloud's token validation, grant enforcement or hosted read service.

Providing a built-in local HTTP facade is outside this storage contract.
Applications can build one without writing SQL or invoking an agent. Their
frontend uses the application API rather than assuming the Cloud threads
service exists for every storage backend.

Conveniences such as `chat_history(agent=)` and `lane(...).history()` can
assemble the corresponding pages. They do not replace the paginated read
operations needed for large conversations and task listings. Saved application
fields and the operations journal retain their separate access boundaries;
external item/task reads do not grant access to either.

## What is saved

The database holds one lane per configured handler that took part (see
[lane mechanics](#lanes)) and one journal for the thread.

| Record | Scope and contents | Ownership |
| --- | --- | --- |
| History | Per lane: messages, tool calls and results | Application-editable model context |
| State | Explicitly saved context fields and routing state, identified by scope | Application chooses load/save; framework validates and commits |
| Tasks | Per lane: A2A status, question, result and transport history | Framework |
| Inbox | Per lane: transport messages waiting under the `batch` policy | Framework |
| Journal | Per thread: append-only events | Framework; see [journal](#journal) |

Unmarked context values, including runtime clients, are not saved. Python
stack frames and local variables are not restored between turns. Media bytes
are stored separately from thread records. Documents and blobs are a
later addition.

Payloads are JSON and queryable in place. Schema changes require compatibility
validation and explicit migrations where needed; adding a key is not sufficient
to guarantee compatibility. State is JSON-shaped and under 64 KiB: a field that cannot be written as JSON raises
`NotSerializable` naming it, a larger state raises `TooLarge`.

## Lanes

A lane is an endpoint or handler's storage and concurrency boundary within a
thread. `AgentHandler(lane=...)` selects it; the worker's configured agent
name is the default. It holds transport history and tasks. A local agent
invocation does not allocate a lane merely because it has another agent name.

Local routing frames and their private histories are saved through
application-owned session state. Several frames may reference the same agent
definition while remaining separate conversations.

```
thread AT_...                one database
├─ lane "front-desk"         history, application data, tasks of the front desk
├─ lane "eligibility"        the same for the expert it delegates to
└─ journal                   every event from every lane, append-only
```

The database and journal are shared by all lanes. History, tasks, inbox and
lease ownership are per lane; saved fields retain their explicit context-scope
identity. See [what is saved](#what-is-saved).

A phone line and text endpoint configured for the same lane share transport
history. Their application data remains subject to explicit load/save.

Each direct `agent.run()` executes the selected agent. An optional
[TextSession](../text-session.md) owns local routing frames and private histories;
the application stores that routing state explicitly. Remote endpoints using
the same project thread own their configured lanes.

Context sharing is explicit. Shared persisted scopes serialize updates;
separate child contexts have independent fields. See
[application context](../context.md).

The lane inbox batches incoming transport messages. The live session's
[conversation-owned inbox](../voice-session.md#conversation-owned-inbox) holds
expert events and reply correlation in process for the originating frontend
conversation. It is push-driven and does not require AgentDB. Persistence for
frontend event recovery is optional and separate from lane request batching.
Sharing a lane or thread database does not share that frontend queue. Neither
mechanism requires application code to manage another public queue API.

### One writer at a time

A lane has exactly one writer at a time, held two ways:

- **The lease.** A run takes the lane's lease before it loads anything and
  keeps it until its commit. The lease lives for 10 seconds and the holder
  renews it every 3; both are storage options. A worker that dies stops renewing, and the lease lapses within its lifetime. A holder
  whose renewal arrives late keeps the lane if nobody took it meanwhile:
  the lease token, not the clock, is what another run has to replace.
- **The fence.** Every commit is conditional on the version the run loaded
  and on its lease token, and bumps the version. A run that outlived its
  lease finds the lane moved on, gets `VersionConflict`, and writes nothing.

The lease coordinates run admission; the database fence rejects stale managed
writes after takeover, including when a paused former owner resumes before
learning it lost ownership. Observed lease loss stops new conversational work
and cancels the run's invocation while preserving cleanup responsibility.
It cannot forcibly stop application code or undo an already-started action.

Independent application writes and external actions use their receiving
system's conditional-write, fencing and idempotency contracts. Idempotency
deduplicates an operation; it does not by itself reject a stale owner's action.
Uncertain effects require reconciliation. Sharing a thread database or taking
its lane lease does not extend the managed fence to another transaction or
service. See [application records and transactions](../context.md#application-records-and-transactions).

Runs waiting for a lane execute in the order their messages arrived: each
message's `submitted` row is written before its run waits for the lease,
and the rows' order is the order the waiters run in. Under `interrupt` and
`queue`, a waiter that has not taken the lane within 60 seconds gives up,
and its task fails with `livekit/reason` `lease-timeout`. Under `batch` a
waiter keeps waiting: the run that drains the inbox owns its messages.

The `concurrent` policy is the one exception: its runs take no lease and
commit append-only, with no fence, and write no state.

### Voice and text share a lane

A call's `VoiceSession`, given `thread=ctx.thread`, keeps the conversation
in memory and takes the lane's lease for each voice turn, from the end of the
person's turn to the commit of the reply, then releases it. A text that arrives between two voice
turns runs on any worker that takes the lane. When the call's next turn
takes the lease and finds a newer version than the one it last committed,
it reloads the transport history. Application code must reload marked context
fields before using updates saved by another activation.

### Two brains, one thread

When a call delegates to an expert that shares the caller's thread (under
[explicit authority](../protocol/a2a.md#delegation-and-conversation-data); the
card's `capabilities.extensions` entry is only a capability hint), the
caller's worker writes the caller's lane and the expert's worker writes the
expert's. Neither writes the other's rows. The expert's result reaches the caller through the delegate tool; the
expert's own turns stay in its lane. A tool reads another agent's history
with `ctx.thread.lane(agent).history()`; the journal, shared, orders every
lane's events in one sequence.

## The unit of work

A managed turn is a unit of work for framework records and explicitly staged
context writes. Independently committed application transactions and external
effects have their own commit boundaries.

- **Load.** Taking the lane's lease loads its history, state and
  journal events in one acquire/load batch. The whole history is loaded;
  compaction, when a conversation outgrows a context window, sits in front of the run and is
  not part of storage.
- **Commit.** Leaving the run writes the history, explicitly staged context/routing writes,
  the events, the task rows and the usage in one write,
  fenced on the version loaded and the lease token, and releases the lease.
  A run that lost its lease meanwhile gets `VersionConflict` and writes
  nothing.
- **Fail.** A run that raises writes only its `run_ended` event and
  releases the lease.
- **A rejected save** ends the task `failed` with `livekit/reason`
  `storage-unavailable`; its staged turn changes were not committed.
- **A lost acknowledgment** is not a failure. The task stays `working` while
  the runtime checks the database for that run; the request returns the
  current task after a bounded wait, and `tasks/get` or a resubscribe later
  shows the real outcome. The framework never reports a failure it did not
  observe.
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

### Application writes and failure boundaries

Persisted context requires explicit load/save; see
[context.md](../context.md). Marked field writes staged by `save_context` join
the managed commit. Independent application transactions and external effects
retain their own durable boundaries and idempotency requirements.

Application SQL is unrestricted: it may read and write any table, including
the framework's. `async with ctx.store.transaction()` joins the managed commit;
`async with ctx.store.independent_transaction()` commits durably when its block
exits. No SQL transaction stays open across model or tool network calls.

Durable execution, operation outboxes, and durable timers are outside this
storage contract.

## Backends

### Local SQLite storage

`SQLiteStorage(directory=...)` requires an explicit directory. Relative paths
resolve against the working directory once at initialization, before workers
start. The resolved absolute path is logged at startup and passed unchanged to
job processes. Reusing the directory restores saved threads across restarts;
choosing another directory creates a separate storage namespace.

Use a project-local `.livekit/storage` directory and add it to the project's
`.gitignore`. Storage does not silently choose a home-directory location or
fall back to Cloud when local storage cannot be opened.

```text
.livekit/storage/
├─ registry.sqlite
└─ threads/
   ├─ AT_abc.sqlite
   └─ AT_def.sqlite
```

The registry holds thread metadata for lookup, listing and lifecycle operations.
Each thread has one database file named `<thread_id>.sqlite`, containing all
its lanes, history, saved application fields, tasks, inbox and journal. SQLite
may create its own WAL and shared-memory sidecar files alongside these files.

The local backend mints `AT_` ids and creates the thread database when creating
a thread. It validates ids before constructing paths; a supplied id is never
treated as an arbitrary filesystem path. Unknown ids follow the common
`ThreadNotFound` contract.

Separate processes on the same machine can share a directory and use the
shared engine's database-backed leases and fences. Local disk storage provides
no replication across machines. Local storage requires no Cloud account or
running agent-db service.

### Cloud AgentDB storage

`AgentDBStorage` uses LiveKit Cloud's agentstore: agent-db holds thread
databases and the threads service mints `AT_` ids and manages thread metadata.

| Argument                 | Environment variable when omitted                                         |
| ------------------------ | ------------------------------------------------------------------------- |
| `api_key`, `api_secret`  | `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`                                    |
| `lease_ttl`              | `LIVEKIT_AGENTDB_LEASE_TTL` in seconds, else 10 seconds                    |
| `renew_interval`         | `LIVEKIT_AGENTDB_RENEW_INTERVAL` in seconds, else 3 seconds                |
| `holder`                 | this worker's lease holder id, `<hostname>-<pid>` by default               |

#### Connection and authentication

Every worker using `AgentDBStorage` reaches agentstore through LiveKit's public
API, at one base URL derived from `LIVEKIT_URL`
(`wss://<project>.staging.livekit.cloud` uses
`https://beta-api.staging.livekit.cloud`, `wss://<project>.livekit.cloud`
`https://beta-api.livekit.cloud`).

| Service | Under the base URL |
| --- | --- |
| the data plane, a WebSocket | `/v0/agentstore/db` |
| threads, `ThreadService` | `/v0/projects/p_/threads` |
| databases, `DatabaseService` | `/v0/projects/p_/databases` |

`p_` is the project of the request's token. The worker mints its own tokens
from the project's key and secret: the data plane takes an
`agent.databaseAdmin` token, one hour, minted per dial; the threads and
databases services a short-lived one per request. HTTP requests carry it as
`Authorization: Bearer`; the data plane carries it as `access_token` on
the WebSocket upgrade, so a proxy admits the connection before the first
frame, and again in its `Hello`.

The threads service owns lifecycle operations and expiry. Updates through
`UpdateThread` move expiry according to `idle_ttl`; managed turn commits do not
implicitly touch a thread.

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
| `tasks`          | A2A task | `pos`, `task_id`, `lane`, `seq`, `name` (an optional application task label), `message_id`, `status`, `status_message`, `updated_at`, `data`, `internal` |
| `lanes`          | lane                         | `lane`, `version`, `state`, `lease_holder`, `lease_token`, `lease_until`, `waiter`, `drained_id` (the last inbox `id` a run consumed) |
| `events`         | journal entry                | `id`, `ts_ms`, `lane`, `run_id`, `type`, `payload` |
| `inbox`          | message received during a run | `id`, `lane`, `task_id`, `message_id`, `message`, `received_at` |

An item's `payload` keeps what the model needs to see the conversation
again: messages and their text, function calls with their `extra`, function
call outputs. It does not keep image or audio content, `group_id`,
a tool output's `reply_required`, or the message ids an inbox batch merged
into one user message (those are on the tasks, as `livekit/mergedInto`). A
row in any other shape fails the load.

A task row is the A2A `Task`: `id` is `task_id`, `contextId` the thread id,
`status` is `status`, `status_message` and `updated_at`, and `data` holds
the rest under the A2A names (`history`, `artifacts`, `metadata`).
`internal` holds framework transport metadata, including the pending question
(`ask`), `deadline`, `parent` and durable references to the task's delivery
streams. The task id is minted by the framework, independent of any stream id.
A row found `working` when a new run picks it up belonged to a worker that died.

## Journal

Every thread keeps two records of itself. The **history** is what the model
sees: messages, tool calls and results, one per lane, owned by the developer
and editable, since truncating it, inserting a summary or deleting an item
are the developer's tools for managing context. The **journal** is what
happened, owned by the framework and append-only: who ran, when, on which
channel, what a tool did. A developer cannot edit it, which is the point.

### Why it exists

- **Derivations that must be trusted.** A run that started and never ended,
  the channel each request came on, the order of events across agents: all
  derived from the journal, never from the editable history. Who is
  speaking in a text-session conversation is recorded in the application's session state.
- **A record that survives compaction.** After a long conversation is
  summarized, the history is short and the journal still holds every turn.
- **The same events for every observer.** The journal's events are the
  same `AgentSessionEvent` objects the voice runtime emits to the console,
  dashboards and evals. A thread opened later replays the same objects
  production recorded.
- **Debugging a thread after the fact.** Opening a thread's journal shows
  what the agent did and why, tool by tool, including what a crash left
  unfinished.

### What is in it

One journal per thread, shared by every lane. Each event carries:

| Field     | What                                              |
| --------- | ------------------------------------------------- |
| `id`      | its position in the thread's journal             |
| `ts_ms`   | when it was written, in milliseconds              |
| `lane`    | the agent it belongs to                           |
| `run_id`  | the run it happened in                            |
| `type`    | one of the types below                            |
| `payload` | the event, as JSON                                |

| Event                     | What it records                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `run_started`             | a run began: its task id and channel. A `run_started` with no `run_ended` after it is how an interrupted run is found |
| `run_ended`               | how the run ended: `outcome` (`completed`, `input-required`, `failed`, `canceled`) and `reason`                                        |
| `usage`                   | the run's model requests, tokens and tool calls                                                                                        |
| `conversation_item_added` | an item entered the history: a message, a function call, a function call output                                            |
| `tool_execution_updated`  | one call's lifecycle: started, a progress update from `ctx.update()`, ended                                                           |
| `error`                   | a failure the runtime recorded                                                                                                        |

On a call, the voice runtime's own events are recorded as well: agent and
user state changes, transcripts, interruptions, end-of-turn and
answering-machine detection. They are the same `AgentSessionEvent`
payloads the console shows and are not listed here.

### Reading it

Inside a run, `await ctx.thread.journal(after_id=0, limit=100)` returns one
page in order; pass the last `id` to read the next. Outside the agent the
journal is readable only through the admin-only
[admin API](../clients.md#admin-reads), which the console uses; the public
threads API and A2A do not expose it.

### What is not in it

Audio and images; persisted application fields and routing state, which are explicit snapshots
rather than a stream of edits; and anything the developer writes to the history by
hand, which is why the two are separate.

## Verification

Run the same storage contract suite against SQLite and AgentDB. AgentDB
integration tests use a Cloud project; local tests need only a temporary
directory. Cover:

- Thread lifecycle, expiry visibility, paging, and recovery from partial
  creation or deletion.
- Persistence across restart, schema compatibility, explicit context saves,
  serialization limits, and application transactions joined to or
  independent of the managed commit.
- Competing processes, paused owners resuming after takeover, lease renewal/loss,
  cancellation and stopped admission after observed loss, stale revisions and
  fenced commits. Application action fixtures verify their declared
  conditional-write, fencing and idempotency behavior separately.
- Duplicate messages, arrival order, inbox handling, commit failure, worker
  loss, pending human input, and uncertain commit acknowledgments.

Assert authoritative committed records as well as caller outcomes. A rejected
staged write and a committed write whose acknowledgment was lost are different
cases. Backend tests must establish both.
