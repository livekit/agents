# Text endpoints

A text endpoint is a door into a thread over the A2A protocol. Web chats,
API clients and partner agents all come through it, and the handler never
knows which.

## The handler

```python
from livekit.agents import AgentServer
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

server = AgentServer(storage=AgentDBStorage(), agent_name="scheduling")

@server.a2a_session(endpoint="scheduling", description="Book, confirm or move an appointment.", state=Patient)
async def text(ctx: A2ASessionContext[Patient]) -> str:
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    result = await scheduler.run(ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps)
    return result.output
```

Called once per turn with the agent's lane already loaded; a batched burst
is one call. The return value is the answer; the framework commits the lane
and delivers it. A handler that raises ends the task `failed`.

A handler that yields instead of returning delivers the answer as it is
written: each piece reaches the client at once, and the answer is the last
run's result, its question when a tool asked for input. `agent.stream()` is
`agent.run()` as that iterator:

```python
@server.a2a_session(endpoint="scheduling", description="Book, confirm or move an appointment.")
async def text(ctx: A2ASessionContext) -> AsyncIterator[str]:
    async for delta in scheduler.stream(ctx.message, message_history=ctx.message_history):
        yield delta
```

A handler that awaits `agent.run()` streams too, the run forwards the
model's text to the client itself; yielding is for a handler that wants the
pieces in its own hands.

A module defines its doors on an `AgentHandler` and leaves the server to
the entrypoint, which serves them with `server.add(handler, ...)`:

```python
back_office = AgentHandler(lane="backend")

@back_office.a2a_session(endpoint="backend", description="The back office.", public=False)
async def text(ctx: A2ASessionContext) -> str: ...
```

`lane=` names the lane its doors write; left out, the worker's own. A
handler with a lane of its own keeps its conversations apart from the
worker's on the same thread ([lanes.md](lanes.md)), which is how two
agents share one process in `examples/`. `livekit.agents.a2a_session` and
`rtc_session` are the same decorators for a single door with no handler;
`server.a2a_session` and `server.rtc_session` serve one in one step.

The agent the handler runs is the root. After a tool hands off, the handoff
is saved on the lane's agent stack, and `root.run(message)` runs whichever
agent is on top of the stack, so the handler stays one line; see
[multi-agent.md](multi-agent.md).

| `A2ASessionContext` | What                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `message`           | the turn's user message, a `ChatMessage`: every part of every message in the batch, in order; `message.text_content` |
| `messages`          | the individual A2A messages of the batch: `message_id`, `parts`, `metadata`, `channel`, `received_at`, `.text`, `.data` |
| `message_history`   | the agent's history so far                                                                                           |
| `state`             | the typed state, loaded                                                                                              |
| `thread`           | the durable thread; see [threads.md](threads.md)                                                                  |
| `task_id`           | the A2A task this turn answers                                                                                       |
| `tasks`             | the lane's recent tasks                                                                                              |
| `caller_history`    | a delegating caller's conversation, when it sent one; see [multi-agent.md](multi-agent.md)                           |
| `update(text)`      | a progress message to the client while the turn runs                                                                 |
| `stream(source)`    | forward text chunks from an async iterable as they arrive; returns the joined text                                   |

| Decorator parameter                 | What                                                                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `endpoint`                          | the last path segment of the endpoint's URL                                                                                              |
| `description`                       | required: what another agent reads to decide whether to call                                                                             |
| `state`                             | the state type, loaded per run                                                                                                           |
| `public`                            | `True` by default: reachable without a LiveKit token. `False` admits only callers holding a project token             |
| `auth`                              | the developer's own check: a callable taking the request and returning a truthy value to admit it                                       |
| `inbox`, `debounce`                 | see below                                                                                                                                |
| `storage`                           | this endpoint's storage when it is not the server's                                                                                      |
| `version`                           | the card's version                                                                                                                       |

The agent is named once, `AgentServer(agent_name=)`: the lane every door of
the server writes, and the first segment of every door's address,
`/agents/{agent_name}/{deployment}/{endpoint}`. `LIVEKIT_AGENT_NAME` is the
default and `LIVEKIT_AGENT_NAME_OVERRIDE` a launcher's force. The voice door,
`@server.rtc_session(endpoint=)`, is named the same way, the agent's name when
left empty.

The endpoint is served at
`https://<project>.livekit.cloud/agents/<agent>/<deployment>/<endpoint>`,
with `default` as the deployment of a worker registered without one. Every
endpoint publishes an agent card at `.well-known/agent-card.json`, so a
stock A2A client works with no LiveKit code: it sends a first message, gets
a `contextId` back, and sends it from then on. The card declares a bearer
scheme for a LiveKit project token or thread grant, always listed and
marked required only when the endpoint is not public.

The endpoint serves the conversation and nothing else. A thread's history
and tasks are read from outside through the platform's threads service;
see [backend-api.md](backend-api.md).

## A message while a turn is running

Two texts seconds apart, a person who keeps typing, two agents calling at
once: each endpoint says what happens, per channel. The channel is the
message's `livekit/channel`, or `delegation` when its `livekit/kind` is
`delegation`, or `a2a` when it carries neither.

| Policy       | Behaviour                                                                                                                                                                                                                                                                      | Default for                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| `live`       | the message waits only while the floor is busy: a reply is produced or a blocking tool runs. A tool that called `ctx.update()` goes on in the background and frees the floor; its end answers its own task. Messages that waited together run as one turn, the earlier ones `livekit/mergedInto` the last. A message that interrupts (`livekit/interrupt` absent or `true`) stops the reply at its next safe point; a delegation does not. See [delegation.md](delegation.md) | `a2a`, `delegation`, and any unlisted channel |
| `interrupt`  | the running turn stops at its next safe point, its task ends `canceled` with `livekit/interrupted: true` and `livekit/interruptionReason: "new-message"`, and the new message gets its turn. Nothing already sent is retracted; tools that ran stay in the history            | opt-in                           |
| `queue`      | each message waits for the running turn, then gets its own turn and its own reply, in the order the messages arrived; nothing merged, nothing cancelled                                                                                                                       | opt-in                           |
| `batch`      | the message is queued in the lane's inbox; after `debounce`, every queued message is answered in one turn. The reply goes on the last message's task; the earlier ones complete with no artifact and `livekit/mergedInto` pointing at it                                     | `sms`, `whatsapp`, `email`       |
| `concurrent` | runs take no lease and may overlap; they append to the history and write no state and no agent stack. Only for endpoints with no `state`                                                                                                                                      | opt-in                           |
| `reject`     | the request fails with `UnsupportedOperationError`, whose data names the running task as `runningTaskId`; no task is created. The client cancels the running task first                                                                                                      | opt-in                           |

```python
@server.a2a_session(
    endpoint="support",
    description="Orders, returns and refunds.",
    inbox={"a2a": "queue"},                   # overrides part of the default map
    debounce=timedelta(seconds=1),            # one value for every channel, or a map by channel
)
```

`inbox` is one policy for every channel or a map that
overrides part of the default. `debounce` is the wait before a turn starts
so a burst has landed, under `batch`, `interrupt` and `queue`: two seconds
for `sms`, `whatsapp` and `email` and zero for `a2a` and `delegation` by
default; `live` never waits for a burst. Under `interrupt` and `queue`, a
message that has not had its turn after 60 seconds fails `lease-timeout`;
under `live` and `batch` it keeps waiting, since the run that drains the
inbox owns its messages.

## Tasks on the wire

Every turn is an A2A task in the thread's context. Its `submitted` row is
written before the turn waits for the lane, so a client can `tasks/get`,
`tasks/list`, `tasks/cancel` and `tasks/resubscribe` it from any worker at
once. Each of these finds the thread from `contextId` in its params: A2A's
task methods carry only the task id, so a stock client that sends just the
id gets `TaskNotFoundError`. The answer is also a durable stream, so a client
that lost its connection resumes from any offset with `tasks/resubscribe`.

- A resent `messageId` returns the task the first send created; nothing
  runs twice.
- `tasks/cancel` on a running task stops its tools at their next wait point
  outside `ctx.uninterruptible()`, and the task ends `canceled` with what
  they did; the answer to the cancel waits for that.
- A tool's `InputRequired` ends the turn `input-required` with the question
  as the reply. The answer continues that task, by its `taskId`, or, for a
  message from the person with none, as the question in front of them; see
  [human-in-the-loop.md](human-in-the-loop.md).
- A message with the `taskId` of a terminal task fails with
  `UnsupportedOperationError`: a follow-up is a new task.

The LiveKit extension, `https://livekit.io/a2a/ext/agents/v1`, adds
typed chat items and these metadata keys, written `livekit/<name>` for
short. A stock client ignores them.

| Key                                                 | Meaning                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `livekit/channel`                                   | the channel a message came on; `a2a` when absent                                             |
| `livekit/kind`                                      | `delegation` on a message another agent sent                                                 |
| `livekit/interrupt`                                 | `false` on a message that must not stop the reply being produced; absent means it does      |
| `livekit/callerLane`                                | on a delegation in the caller's thread, the caller's lane                                    |
| `livekit/background`                                | on a `working` task whose work goes on after its turn                                       |
| `livekit/queued`                                    | on an `input-required` task whose question waits behind the one in front of the person      |
| `livekit/cancelRequested`                           | on a `working` task a client canceled, until its tools stop                                 |
| `livekit/interrupted`, `livekit/interruptionReason` | on a `canceled` task the server ended for a newer message                                    |
| `livekit/mergedInto`                                | on a `completed` task whose message was answered by another task's turn                      |
| `livekit/reason`                                    | why a task failed: `thread-not-found`, `lease-timeout`, `worker-lost`, `storage-unavailable` |
| `livekit/verbatim`                                  | the part of an answer or a question that reaches the person exactly as written               |

`livekit/mergedInto` marks one case: a message a `live` or `batch` turn
answered with a later one. An answer to a question continues the waiting
task under its id, with nothing merged.

## Clients

```python
from livekit.agents.a2a import AgentClient

client = AgentClient("https://acme.livekit.cloud/agents/assistant/production/assistant", token=token)
reply = await client.send("Where is my order?", thread_id=thread_id)
reply.task_id, reply.thread_id
async for update in reply: ...                                 # progress, then the answer
await client.send("Yes, the generic is fine", thread_id=thread_id, task_id=task_id)   # answer a waiting task
await client.cancel(task_id, thread_id=thread_id)
```

Thin over the stock `a2a-sdk`; any A2A client works without it.

Background tasks, a delegation not awaited or a `livekit/task` message, are planned;
see [planned-agent-tasks.md](planned-agent-tasks.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py` and
`04_a_claim_with_a_photo.py`.
