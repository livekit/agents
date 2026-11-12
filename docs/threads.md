# Threads

A thread is one conversation between a person and an application, across
channels and over time. A text today, a call tomorrow that continues it:
one thread, one memory. It is the unit everything else is scoped to: the
history, the state, the tasks, the A2A context.

## What it is not

- Not a call. A call is a job: a worker process attached to a room for the
  call's duration. The thread outlives it. The runtime a call attaches to
  is a `VoiceSession`; see [voice-session.md](voice-session.md).
- Not a request. A text message is a run inside the thread; see below.
- Not a person. One person may have several threads, one per chat in a
  sidebar for instance, and the application decides which.
- Not an agent's. Any agent of the project can continue a thread; each
  writes its own partition.

## What it holds

|             |                                                                                                                                                                |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| an id       | `AT_...`, minted by the platform. A client learns it as the `contextId` of the first reply and sends it from then on; the handler reads it as `ctx.thread.id`  |
| scope       | an optional opaque identifier the application sets for the person or account the thread is about; the one dimension threads are listed by, and a grant's scope |
| attributes  | a small string map the application sets and reads back; a title, a preview, flags                                                                              |
| partitions       | one per configured handler partition: transport history, saved application data and tasks; see [partition concepts](#partitions-and-session-frames)                           |
| the journal | the framework's append-only record of what happened, shared by every partition; see [journal](internals/storage.md#journal)                                         |

Several endpoints can take part in one thread, each in its configured partition.
Local agents coordinated by a session have separate frames within saved session
state; those frames do not each create a transport partition. A tool reads another agent's history
with `ctx.thread.partition(partition_key).history()`.

## Partitions and session frames

`a2a_session(partition_key=...)` and `rtc_session(partition_key=...)` select saved
history, tasks and write ownership inside a thread. Both resolve omitted keys to
the protocol's `DEFAULT_PARTITION_KEY` (`"default"`) before accessing storage,
independently of agent, handler and endpoint names. Phone and chat
continue the same history when their thread and partition key match. A remote
expert selects a separate key when it needs independent conversation state.
`AgentHandler` only groups entry points; it has no persistence identity.
A partition is not allocated for every local agent invocation.
`TextSessionState` keeps its own agent frames, private histories, and pending
calls; those frames do not each create a partition.

The journal is the framework's append-only event record, separate from editable
model history and explicitly saved context fields. Public thread methods read
those records; lease/fence, table, and event mechanics are in
[storage internals](internals/storage.md).

## How a thread is found

By its id, and nothing else. A text request names its thread on every request; a call's
thread is named once, by whoever dispatches the agent. The agent never picks a thread
itself, and a call never creates one.

### Text

| The request                                                                              | Who names the thread                                                         | The thread                                                                                                     |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| A2A `message/send` with `contextId` (a browser, a backend, `AgentClient`, another agent) | the caller, authorized by its token's `agentThread` grant or a project token | that thread; an id the platform does not know fails the task `thread-not-found`                                |
| A2A `message/send` without `contextId`                                                   | nobody                                                                       | a new thread, in the grant's `scope`; a retried first message (same `messageId`) gets the same thread and task |
| an SMS or a WhatsApp message                                                             | LiveKit Cloud's ingress, by the sender's `phone:+E164` key (get-or-create)   | that thread, posted as `contextId`; see [channels.md](channels.md)                                             |
| an email                                                                                 | the ingress, by the sender's `email:<address>` key (get-or-create)           | the same                                                                                                       |

### Calls

A call's thread is the `thread_id` of its job, copied from the agent dispatch that started it.
Every route below is signed with the project's key, so the agent trusts the id without a grant.

| The call                                      | Where `thread_id` is set                                                                | Who sets it                                                                                                         |
| --------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| a browser joins and its join creates the room | the token's `roomConfig.agents[].threadId`                                              | the backend that mints the token, choosing a thread the user's `agentThread` grant covers                           |
| a backend creates the room with its agents    | `CreateRoom`'s `RoomConfiguration.agents[].thread_id`                                   | the backend                                                                                                         |
| an agent joins a room that exists             | `CreateAgentDispatch(thread_id=)`                                                       | the backend                                                                                                         |
| an outbound phone call                        | the dispatch of the agent that places or answers it (`CreateAgentDispatch(thread_id=)`) | the backend, from the thread it stored against the lead                                                             |
| an inbound phone call                         | the dispatch the SIP dispatch rule creates                                              | LiveKit Cloud, by the caller's `phone:+E164` key (get-or-create), so a call continues the texts and calls before it |

| The job's `thread_id`            | `ctx.thread` and the call                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------- |
| a known thread     | that thread, open before the entrypoint runs; a partition that does not load fails `start()` |
| empty              | `None`: the call runs in memory and no thread is created                                 |
| unknown or deleted | `None`: the refusal is logged; the call runs in memory                                   |

Whoever dispatches the call already has its thread id and hands it to its frontend;
the room does not publish it. A tool with the project's key reads it from the
dispatch (`ListDispatch`).

A terminal call follows the same rule: a simulated job without `thread_id` runs without a
durable thread. A launcher that needs persistence must create a thread and supply its id.

A backend that wants to find a conversation again stores the thread id in its own tables,
against its user, lead or order, and passes it on as `contextId` or in the dispatch; see
[clients.md](clients.md).

## One person, several conversations

A thread is a conversation, not a person, so how many a person has is the
application's choice.

| Channel       | Conversations per person         | How                                                                                                                                                   |
| ------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| web chat, API | as many as the application wants | a thread per chat, created with the person as `scope`, listed by scope for a sidebar, titled with `update(attributes=)`; see [clients.md](clients.md) |
| the same      | one                              | the backend keeps one thread id per person and sends it every time                                                                                    |
| calls         | the one the dispatch names       | the dispatch's `thread_id` (see [Calls](#calls)); with none, the call runs in memory and is not kept                                                  |

Tags, folders and search are the application's own table, keyed by the
thread id. Each thread is its own memory: what the agent learned in one
of a person's threads it does not know in another. An application that
wants one memory keeps one thread per person.

For SMS, WhatsApp and email routing, see [channels.md](channels.md).

## Runs

A run is one input to one result, tools included: a text message, a voice
turn. The framework loads the agent's partition before a run and commits it
after, so a run can execute on any worker. Application fields require explicit
load/save; returning from a handler does not save them.
Runs of one agent in a thread execute one at a time, in the order their
messages arrived, unless the endpoint chose otherwise; see
[partition mechanics](internals/storage.md#partitions) and [server.md](server.md).

## `ctx.thread`

On every context, in a tool during any run and in an endpoint's handler,
`ctx.thread` is the durable thread. On a call, the live call itself is
`ctx.live_session`, `None` outside a call; only it speaks. In an
`rtc_session` entrypoint, `ctx.thread` is the call's thread, `None` when the
dispatch named none, and `VoiceSession(thread=ctx.thread)` gives it to the
call. A call without a thread runs in memory; a call whose thread does not load
fails to start (see [voice-session.md](voice-session.md#the-calls-thread)).
An `AgentServer` always has storage: its `storage=`, else the local
`SQLiteStorage()`. A direct `agent.run()` outside a server has none, and
`ctx.thread` raises `RuntimeError` there. See [storage.md](internals/storage.md).

| Member                                                                             | Returns              | What                                                                                   |
| ---------------------------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| `id`                                                                               | `str`                | the thread id                                                                          |
| `info()`                                                                           | `ThreadInfo`         | the platform's record: scope, attributes, `created_at`, `last_active`, `expires_at`    |
| `update(attributes=, idle_ttl_seconds=)`                                           | `ThreadInfo`         | change the attributes a listing returns, or the idle period                            |
| `partition(partition_key)`                                                                      | `Partition`               | one persistence partition of this thread                                                        |
| `partition(partition_key).history()`                                                            | `ChatContext`        | another agent's history in this thread                                                 |
| `chat_history(partition_key=None)`                                                         | `ChatContext`        | this agent's history, or another agent's                                               |
| `partition(partition_key).tasks(status=, limit=)`                                               | `list[TaskRecord]`   | A2A task records for the selected partition                                                 |
| `list_items(agent=None, after=None, limit=100)`                                    | `ThreadItemPage`     | one page of committed chat items in insertion order; the page's `after` reads the next |
| `list_tasks(agent=None, status=None, order="inserted", page_token=None, limit=50)` | `TaskPage`           | one page of task records, in insertion order or last updated first (`order="updated"`) |
| `task(task_id)`                                                                    | `TaskRecord \| None` | one task record, `None` when the thread has none of that id                            |
| `journal(after_id=0, limit=100)`                                                   | `list[EventRow]`     | the journal, one page in order                                                         |

## Lifecycle

A thread is created on first contact, a message with no `contextId`, or
explicitly through the threads service; a call never creates one.
Creating it creates its database. It expires after its idle period, `idle_ttl`, set at
creation or by `update(idle_ttl_seconds=)`; an update counts as activity
and pushes the expiry. Deleting a thread deletes everything it holds.
There is nothing to open or close.

`livekit/mergedInto` on a task is unrelated to threads: messages answered
in one turn; see [server.md](server.md).


## Session binding and snapshots

The entrypoint's store selects the thread and partition. `ctx.store.bind(app)` explicitly
authorizes TextSession or VoiceSession to advance that conversation. A persisted session
field is discovered only within this explicit binding; passing ordinary context does not
implicitly restore or update a conversation.

`await ctx.store.load_session()` reads a detached committed snapshot without executing an
agent. `ctx.thread.chat_history()` reads the selected partition's messages. Separate reads
are not one atomic historical snapshot. During execution, a bound app.session reflects
accepted in-memory state, which can lead committed storage until the next checkpoint.

Delegation into an active caller/ancestor's same store/thread/partition is rejected before
task admission, including the default key. Different threads or stores remain independent;
ordinary requests to endpoints sharing a key remain valid and queue under that partition.
