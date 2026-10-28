# The backend API

Everything a backend does with threads from outside the agent: the
platform's threads service, the tokens it mints, and a client for an
endpoint. The wire underneath is [a2a.md](protocol/a2a.md).

## Threads

A thread (`AT_`) is a record of the platform's threads service, reached
through LiveKit's public API like the rest of agentstore. Its client comes with the storage,
under the project credential:

```python
from livekit.agents.storage import AgentDBStorage, ThreadNotFound

storage = AgentDBStorage()   # LIVEKIT_URL, and the project's key and secret
threads = storage.threads

info = await threads.create_thread(scope=user_id, attributes={"title": "New chat"})
thread_id = info.thread_id
page = await threads.list_threads(user_id, limit=30)                    # the scope, newest first
older = await threads.list_threads(user_id, before=page.before, limit=30)

info = await threads.get_thread(thread_id)                             # raises ThreadNotFound
await threads.update_thread(thread_id, attributes={"title": "Renewal"})   # idle_ttl_seconds=
await threads.delete_thread(thread_id)                                  # its content goes too

await storage.aclose()
```

| Call | Request | Response |
|---|---|---|
| `CreateThread` | optional `scope`, `attributes`, `idle_ttl` | the thread, with its database created |
| `GetThread` | `thread_id` | the thread |
| `ListThreads` | `scope`, optional `before`, `limit` | the scope's threads, newest first, and the `before` cursor of the next page, empty at the end |
| `UpdateThread` | `thread_id`, `attributes` (an empty value deletes the entry), `idle_ttl` | the thread; counts as activity |
| `DeleteThread` | `thread_id` | nothing; its content goes with it |
| `GetThreadItems` | `thread_id`, optional `agent`, `after`, `limit` | the history in order, one page: each item's `agent`, `seq`, `message_id` and chat item, and the `after` cursor of the next page |
| `GetThreadTasks` | `thread_id`, optional `status` | A2A task records: `task_id`, optional task label, `status`, `updated_at` |

A thread is a `ThreadInfo`: `thread_id`, `scope`, `attributes`,
`created_at`, `last_active`, `expires_at` (0: never), `idle_ttl_seconds`,
`database_id`. Times are timestamps and TTLs durations on the wire; the
Python client reads and writes them as seconds. `list_threads` returns a
`ThreadPage`, `threads` and the `before` cursor, empty on the last page. A
thread lives in the project's data region and belongs to the project, not
to one agent: any agent of the project continues it.

A thread's `scope` is its subject: an opaque identifier the application
sets for the person or account the thread is about, hashed if it is
personal. It is the one dimension threads are listed by: newest first,
paginated, each row with the attributes the application set, a title or a
preview. A thread per chat with the user as scope is how a sidebar works.

## A thread handle

A handle addresses a conversation in the platform's threads service, using
the project credential or a thread grant. Sending a message separately names
the agent endpoint through `AgentClient`:

```python
from livekit import api
lk = api.LiveKitAPI()

thread = lk.threads.get(thread_id)                                       # a handle; nothing fetched yet
thread = await lk.threads.create(scope=user_id)                          # a new thread for a subject
threads = await lk.threads.list(scope=user_id, limit=30)                 # newest first, with info loaded

thread.id
info = await thread.info()                                                 # scope, attributes, created, last active
await thread.update(attributes={"title": "Renewal"})
await thread.delete()

items = await thread.chat_history()                       # list[ChatItem]: the thread's history
items = await thread.chat_history(agent="eligibility")    # one agent's lane of the thread
```

`create`, `get`, `list`, `info`, `update`, `delete`, and `chat_history`
use the threads service. Thread history selects configured transport lanes;
it does not flatten private `TextSessionState` frame histories. See
[thread reads](#reading-a-thread).

An endpoint client owns message delivery, task progress, replies, and
cancellation. A thread ID alone does not choose an endpoint or an agent.
Inside a managed run, `ctx.thread` also exposes scoped lane access.
`TextSession` coordinates execution separately; a thread handle is not a
rehydrated session.

## The boundary

The platform holds what continues a conversation: the id, the history,
state, tasks and journal, the one listing by scope. Your database holds
what your product says about a conversation: which thread belongs to
which lead or order, tags, folders, pins, assignment to a person, unread
counts, search across users, reports. Keyed by the thread id, which is
stable and yours to store. A tagging feature is one table of your own.

## Reading a thread

A thread's history and its tasks come from the threads service's
`GetThreadItems` and `GetThreadTasks`, under the project credential or a
grant with `read`; the agent serves no content routes of its own. The
state, lanes and journal are not readable through these public APIs; only
the [admin API](#admin-reads) reads them. Inside a run,
`ctx.thread` is the durable thread: `info()`, `chat_history(agent=)`,
`update(attributes=)` and `lane(...).tasks(...)`
([threads.md](threads.md)).

## Tokens

`Credentials(api_key, api_secret)` mints and verifies the project's
tokens, from `livekit.agents.a2a`:

```python
from livekit.agents.a2a import AgentThreadGrant, Credentials

credentials = Credentials(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
token = credentials.thread_grant_token(
    AgentThreadGrant(scope=user_id, create=True, write=True, read=True),
    identity=user_id,
    ttl_seconds=3600,
)
```

| Call | Returns |
|---|---|
| `thread_grant_token(grant, identity="", ttl_seconds=3600)` | a thread grant for a browser or a device |
| `project_token(identity="worker", ttl_seconds=3600)` | a project credential carrying the `agentEndpoint` call grant, which passes the node to any endpoint of any agent in the project |
| `verify(token)` | the `Principal`: `kind` (`project` or `grant`), `grant`, `identity`, `claims`; raises `Unauthorized` |

`AgentThreadGrant` takes `scope` or `thread_ids` and the flags `create`,
`write` and `read` ([a2a.md](protocol/a2a.md), Tokens); `can_create()`,
`can_list(scope)`, `can_read(thread)`, `can_write(thread)` and
`can_delete(thread)` are its checks. Its `list` and `delete` flags are the
threads service's browser operations
([thread handles](#a-thread-handle)). With that token the browser talks to the platform's threads API, and
what it cannot do is see another scope, set attributes, or anything the
grant does not list. What stays on your backend: minting tokens, setting
attributes such as a title when the agent does not, and your product's own
tables.

## A client for an endpoint

```python
from livekit.agents.a2a import AgentClient

client = AgentClient(endpoint_url, token=token)
reply = await client.send("Where is my order?", thread_id=thread_id)
print(reply.task_id, reply.thread_id)              # known once the server acknowledged the task
async for update in reply:                          # progress, then the answer
    print(update.state, update.text)
await client.send(answer, thread_id=thread_id, task_id=waiting_task_id)   # answer a waiting task
await client.cancel(task_id, thread_id=thread_id, reason="the user closed the chat")
await client.aclose()
```

`AgentClient(endpoint_url, token=None, conn_options=None)` accepts the shared
[connection policy](models.md#connection-options) and has four calls: `connect`, `send`,
`cancel` and `aclose`. It runs on the framework's Rust A2A client, the one
the server side uses: `connect()`, which `send` and `cancel` call for you,
reads the card once and speaks JSON-RPC over HTTP with server-sent events.
`send(message, thread_id=, task_id=, metadata=, chat_ctx=)` takes a `str` or
a `Message` (`from livekit.agents.a2a import Message`), the typed A2A message
record: `parts` (text, data or file), `metadata` (typed LiveKit keys and an
application map) and an optional `message_id`. With no `thread_id` it
continues the thread of its last reply, and with none it starts a new one. The
endpoint accepts a `thread_id` only
when the token it verifies covers that thread ([tokens](protocol/a2a.md#tokens)).
`chat_ctx` goes as a `chat_ctx` part only to a card that lists the LiveKit
extension ([conversation data](protocol/a2a.md#conversation-data-part)). Each update is a
`TaskUpdate`: `state`, `text`, `task_id`, `context_id`, `metadata` with the
full URI keys; `livekit/<name>` in these docs is shorthand only. A `working`
update with text is progress, and the terminal update carries the whole answer.
`artifacts` contains typed `Artifact` records (`artifact_id`, `name`,
`description`, `parts`, `metadata`). `offers_extension` is whether the
card lists the LiveKit extension, a capability hint that grants nothing.

Construct message parts with `TextPart(text=...)`, `DataPart(data=...)` or
`FilePart(url=..., filename=..., media_type=...)` from `livekit.agents.a2a`.
A file part can carry inline `data` instead of a URL when its endpoint accepts
uploads. LiveKit endpoints require an application reference in a URL part;
the application resolves and authorizes that reference. `MessageMetadata` holds
typed LiveKit fields such as `kind`, `channel` and `verbatim`, plus an
`application` map. Application keys are preserved; short `livekit/...` keys in
that map have no framework meaning.

```python
from livekit.agents.a2a import Message, MessageMetadata, TextPart

reply = await client.send(Message(
    parts=[TextPart(text="Where is my order?")],
    message_id="order-question-1",
    metadata=MessageMetadata(application={"order_id": order_id}),
))
```

The returned `Reply` is an async iterator. `await reply.aclose()` releases its
response stream when stopping consumption early. Concurrent pulls raise
`StreamStateError` without disturbing the accepted pull.

A resume is a `send` with the thread id, its `contextId`: there is no
separate get or resume call. Retrying a first message with the same
`message_id` returns the same thread and task, including on another worker.

Nothing here is required: `tasks/get`, `tasks/list` and
`tasks/resubscribe` are served to any A2A client.

## Admin reads

The developer console reads a thread's state, lanes and journal through the
AgentDB admin service, `AgentDBService` in livekit/protocol
`livekit_agentdb.proto`, served by LiveKit's public API under
`/v0/projects/p_/databases`: `GetDatabase`, `ListDatabases`, `DumpDatabase` (a
consistent SQLite export of a thread's database, `ThreadInfo.database_id`, as a
time-limited download URL), `CreateDatabase` and `DeleteDatabase`; queries go
over the data plane ([connection](internals/storage.md#connection-and-authentication)).
It is admin-only: it requires an `agent.databaseAdmin` token minted from the
project's key and secret, runs on a developer's backend, and is never exposed to
a customer frontend. Customer frontends use only the threads API and A2A above,
under an authorized grant.

## Calls

```python
import json

dispatch = api.CreateAgentDispatchRequest(
    agent_name="sales", room=room, metadata=json.dumps({"thread_id": thread.id})
)
await lk.agent_dispatch.create_dispatch(dispatch)                              # the job carries the thread
await lk.sip.create_sip_participant(api.CreateSIPParticipantRequest(room_name=room, sip_trunk_id=TRUNK, sip_call_to=lead.phone))
```

The dispatch carries the thread id, and the job hands it to the call's
`VoiceSession`.

## From the browser

A chat UI needs no backend of its own beyond the one call that mints a
token ([tokens](#tokens)). The browser uses the
same client with it:

```javascript
const threads = new ThreadsClient({ token });             // scoped: this subject
const chats = await threads.list({ limit: 30 });           // only this subject's
const thread = threads.get(chats[0].id);
const history = await thread.chatHistory();
```

The backend mints the grant; the browser receives only that scoped credential.
To send a message, the client must also be configured with the deployed agent's
endpoint URL and include the chosen thread ID. Use the endpoint-client
[send/reply/cancel contract](#a-client-for-an-endpoint); listing
threads does not implicitly select where a message should execute.

With a grant, `list` takes the subject from the token (the grant's `list`
flag) and `delete` needs the grant's `delete` flag.
