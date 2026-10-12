# Threads

A thread is one conversation between a person and an agent, across
channels and over time. A text today, a call tomorrow that continues it:
one thread, one memory. It is the unit everything else is scoped to: the
history, the state, the tasks, the A2A context.

## What it is not

- Not a call. A call is a job: a worker process attached to a room for the
  call's duration. The thread outlives it. The runtime a call attaches to
  is a `LiveSession`; see [voice.md](voice.md).
- Not a request. A text message is a run inside the thread; see below.
- Not a person. One person may have several threads, one per chat in a
  sidebar for instance, and the application decides which.
- Not an agent's. Any agent of the project can continue a thread; each
  writes its own lane.

## What it holds

|             |                                                                                                                                                                   |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| an id       | `AT_...`, minted by the platform. A client learns it as the `contextId` of the first reply and sends it from then on; the handler reads it as `ctx.thread.id`     |
| scope       | an optional opaque identifier the application sets for the person or account the thread is about; the one dimension threads are listed by, and a grant's scope    |
| attributes  | a small string map the application sets and reads back; a title, a preview, flags                                                                                 |
| lanes       | one per agent that took part: that agent's history, state, agent stack and tasks; see [lanes.md](lanes.md)                                                        |
| the journal | the framework's append-only record of what happened, shared by every lane; see [journal.md](journal.md)                                                           |

Several agents can take part in one thread: a front desk and the expert it
delegates to, each in its own lane. A tool reads another agent's history
with `ctx.thread.lane(agent).history()`.

## How a thread is found

By its id, and nothing else.

| A request carries                     | The thread                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------- |
| the id, as the A2A `contextId`        | that thread; an id the platform does not know fails the task `thread-not-found`     |
| no `contextId`                        | a new thread                                                                       |
| a job dispatch for a call             | the `thread_id` in the dispatch's JSON metadata, else a new thread, as `ctx.thread` before the entrypoint runs |

A backend that wants to find a conversation again stores the thread id in
its own tables, against its user, lead or order, and passes it on as
`contextId` or in the dispatch; see [backend-api.md](backend-api.md).
Finding a call's thread by the caller's number is planned; see
[planned-channels.md](planned-channels.md).

## One person, several conversations

A thread is a conversation, not a person, so how many a person has is the
application's choice.

| Channel       | Conversations per person         | How                                                                                                                                                  |
| ------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| web chat, API | as many as the application wants | a thread per chat, created with the person as `scope`, listed by scope for a sidebar, titled with `update(attributes=)`; see [backend-api.md](backend-api.md) |
| the same      | one                              | the backend keeps one thread id per person and sends it every time                                                                                    |
| calls         | the one the dispatch names       | the backend or the console puts the thread id in the dispatch metadata; with none, every call is a new thread                                          |

Tags, folders and search are the application's own table, keyed by the
thread id. Each thread is its own memory: what the agent learned in one
of a person's threads it does not know in another. An application that
wants one memory keeps one thread per person.

SMS, WhatsApp and email are planned; see [planned-channels.md](planned-channels.md).

## Runs

A run is one input to one result, tools included: a text message, a voice
turn. The framework loads the agent's lane before a run and commits it
after, so a run can execute on any worker and the next one sees everything.
Runs of one agent in a thread execute one at a time, in the order their
messages arrived, unless the endpoint chose otherwise; see
[lanes.md](lanes.md) and [text-endpoints.md](text-endpoints.md).

## `ctx.thread`

On every context, in a tool during any run and in an endpoint's handler,
`ctx.thread` is the durable thread. On a call, the live call itself is
`ctx.live_session`, `None` outside a call; only it speaks. In an
`rtc_session` entrypoint, `ctx.thread` is the call's thread, `None` on a
server without storage, and `LiveSession(thread=ctx.thread)` gives it to
the call. Without storage, a run's `ctx.thread` is an in-memory thread
with the same API and no persistence; see [storage.md](storage.md).

| Member                                         | What                                                                          |
| ---------------------------------------------- | ----------------------------------------------------------------------------- |
| `id`                                           | the thread id                                                                 |
| `info()`                                       | the platform's record: scope, attributes, created, last active, expiry        |
| `update(attributes=, idle_ttl_seconds=)`       | change the attributes a listing returns, or the idle period                   |
| `lane(agent).history()`                        | another agent's history in this thread                                        |
| `chat_history(agent=None)`                     | this agent's history, or another agent's                                      |
| `lane(agent).tasks(status=, limit=)`, `background_tasks(agent=None)` | a lane's tasks: every turn, or only the submitted background tasks |
| `journal(after_id=0, limit=100)`               | the journal, one page in order                                                |

## Lifecycle

A thread is created on first contact, a message with no `contextId` or a
call with no thread in its dispatch, and never by agent code. Creating it
creates its database. It expires after its idle period, `idle_ttl`, set at
creation or by `update(idle_ttl_seconds=)`; an update counts as activity
and pushes the expiry. Deleting a thread deletes everything it holds.
There is nothing to open or close.

`livekit/mergedInto` on a task is unrelated to threads: messages answered
in one turn; see [text-endpoints.md](text-endpoints.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py`, and
`planned-12_many_chats.py` for a sidebar.
