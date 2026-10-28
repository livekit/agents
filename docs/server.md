# AgentServer and entrypoints

`AgentServer` serves application entrypoints. Modules can declare doors on
an `AgentHandler` and the server assembles them. Text endpoints receive A2A
requests; voice entrypoints attach a `VoiceSession` to a room. Both receive
runtime thread/storage access, while application context loading and saving
remain explicit.

Voice entrypoints accept `on_simulation_end`, called with a `SimulationContext`
after a simulated conversation finishes. It exposes the simulator verdict and
lets application checks veto the result. See [testing and simulations](testing.md#simulations)
for registration, context fields and verdict semantics.

## Text entrypoints

```python
from livekit.agents import AgentServer, RunResult
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

server = AgentServer(storage=AgentDBStorage(), agent_name="scheduling")

@server.a2a_session(endpoint="scheduling", description="Book, confirm or move an appointment.")
async def text(ctx: A2ASessionContext) -> RunResult[Appointment]:
    app = SchedulingContext(clinic=clinic_service)
    await ctx.store.load_context(app)
    result = await scheduler.run(
        ctx.message, message_history=ctx.message_history, context=app
    )
    await ctx.store.save_context(app)
    return result
```

Return the full `RunResult` so the transport can map `status="input_required"`
and `question` to A2A `input-required`, or `status="completed"` and `output`
to a completed task artifact. The handler saves application fields explicitly
before returning either result.

Called once per turn with the agent's lane already loaded; a batched burst
is one call. The return value is the answer; the framework commits the lane
and delivers it. A handler that raises ends the task `failed`.

For explicit streaming, consume `agent.run_stream(...)`, forward text through
`ctx.stream(...)`, and return its complete `RunResult` after saving context.
The returned result finalizes the same answer rather than delivering the text
again. See [managed streaming endpoints](agents.md#managed-endpoints-and-persistence)
for the complete entrypoint and cancellation contract.

A handler using `await agent.run(...)` can use automatic managed forwarding.
`run_stream(...)` instead gives its consumer control of delivery, so the same
text occurrences are not forwarded by both paths.

A module defines its doors on an `AgentHandler` and leaves the server to
the entrypoint, which serves them with `server.add(handler, ...)`:

```python
back_office = AgentHandler(lane="backend")

@back_office.a2a_session(endpoint="backend", description="The back office.")
async def text(ctx: A2ASessionContext) -> str: ...
```

`lane=` names the lane its doors write; left out, the worker's own. A
handler with a lane of its own keeps its conversations apart from the
worker's on the same thread ([lane mechanics](internals/storage.md#lanes)), which is how two
agents share one process in `examples/`. `livekit.agents.a2a_session` and
`rtc_session` are the same decorators for a single door with no handler;
`server.a2a_session` and `server.rtc_session` serve one in one step.

The handler explicitly calls an agent or a [TextSession](text-session.md).
A direct `agent.run()` executes that agent. A text-session handler loads the routing
state, passes it to `session.run()`, and saves the returned state.

| `A2ASessionContext` | What                                                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `message`           | the turn's user message, a `ChatMessage`: every part of every message in the batch, in order; `message.text_content` |
| `messages`          | the individual A2A messages of the batch, each a [`Message`](clients.md#a-client-for-an-endpoint) record: `message_id`, `parts`, `metadata`, `channel`, `received_at`, `.text`, `.data` |
| `message_history`   | the agent's history so far                                                                                           |
| `store`             | managed storage for explicit application-context load/save |
| `thread`           | the durable thread; see [threads.md](threads.md)                                                                  |
| `task_id`           | the A2A task this turn answers                                                                                       |
| `tasks`             | the lane's recent tasks                                                                                              |
| `caller_history`    | a delegating caller's conversation, when it sent one; see [multi-agent composition](agents.md#working-with-other-agents)                           |
| `update(text)`      | a progress message to the client while the turn runs                                                                 |
| `stream(source)`    | forward text chunks from an async iterable as they arrive; returns the joined text                                   |

| Decorator parameter                 | What                                                                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `endpoint`                          | the last path segment of the endpoint's URL                                                                                              |
| `description`                       | required: what another agent reads to decide whether to call                                                                             |
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
required.

The endpoint serves the conversation and nothing else. A thread's history
and tasks are read from outside through the platform's threads service;
see [clients.md](clients.md).

## Voice entrypoints

```python
@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    app = SchedulingContext(clinic=clinic_service)
    session = VoiceSession(
        llm=REALTIME_MODEL, context=app, thread=ctx.thread
    )
    await session.start(agent=scheduler, room=ctx.room)
    await session.store.load_context(app)
    await session.generate_reply(instructions="Greet the caller and offer help.")
```

The runtime resolves the dispatched thread before the handler runs. The
application constructs the frontend context and starts its voice session.
Speech, turn detection, agent replacement, and frontend event delivery follow
the [VoiceSession contract](voice-session.md).

## Handler outcomes

Tool and handler transitions have distinct meanings. An ordinary tool's
`Completed(value)` supplies that tool's result to the model. A handler can
use transport transitions to answer its A2A task.

```python
from livekit.agents import Completed, Failed, Rejected, InputRequired, Working

Completed(value)                 # tool result, or a completed transport task
Failed(error)                    # failed: the error text is the last message
Rejected(reason)                 # will not run this input
InputRequired(question)          # waiting on the person; asked in the facing agent's words
Working(message=None)            # transport working status; no Python checkpoint
```

When returned as handler transitions, these map to A2A task states `completed`, `failed`, `rejected`,
`input-required` and `working`.

## A message while a turn is running

Two texts seconds apart, a person who keeps typing, two agents calling at
once: each endpoint says what happens, per channel. The channel is the
message's `livekit/channel`, or `delegation` when its `livekit/kind` is
`delegation`, or `a2a` when it carries neither.

| Policy       | Behaviour                                                                                                                                                                                                                                                                      | Default for                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| `interrupt`  | the running turn stops at its next safe point, its task ends `canceled` with `livekit/interrupted: true` and `livekit/interruptionReason: "new-message"`, and the new message gets its turn. Nothing already sent is retracted; tools that ran stay in the history            | `a2a`, and any unlisted channel  |
| `queue`      | each message waits for the running turn, then gets its own turn and its own reply, in the order the messages arrived; nothing merged, nothing cancelled                                                                                                                       | `delegation`                     |
| `batch`      | the message is queued in the lane's inbox; after `debounce`, every queued message is answered in one turn. The reply goes on the last message's task; the earlier ones complete with no artifact and `livekit/mergedInto` pointing at it                                     | `sms`, `whatsapp`, `email`       |
| `concurrent` | runs take no lease and may overlap; they append to the history and cannot save application fields or routing state. Only for endpoints that do not write application data                                                                                                                                      | opt-in                           |
| `reject`     | the request fails with `UnsupportedOperationError`, whose data names the running task as `runningTaskId`; no task is created. The client cancels the running task first                                                                                                      | opt-in                           |

```python
@server.a2a_session(
    endpoint="support",
    description="Orders, returns and refunds.",
    inbox={"a2a": "queue"},                   # overrides part of the default map
    debounce=timedelta(seconds=1),            # one value for every channel, or a map by channel
)
async def support(ctx: A2ASessionContext):
    ...
```

`inbox` is one policy for every channel or a map that
overrides part of the default. `debounce` is the wait before a turn starts
so a burst has landed, under `batch`, `interrupt` and `queue`: two seconds
for `sms`, `whatsapp` and `email` and zero for `a2a` and `delegation` by
default. Under `interrupt` and `queue`, a message that has not had its turn
after 60 seconds fails `lease-timeout`; under `batch` it keeps waiting, since
the run that drains the inbox owns its messages.

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
- `tasks/cancel` on a running turn ends it `canceled`; the turn yields at its
  next boundary and a running tool finishes.
- A tool's `InputRequired` ends the turn `input-required` with the question
  as the reply. The next message in the thread answers it; see
  [human input](tools.md#human-input).
- A message with the `taskId` of a terminal task fails with
  `UnsupportedOperationError`: a follow-up is a new task.

LiveKit endpoints add typed chat items and these metadata keys, under the
namespace `https://livekit.io/a2a/ext/agents/v1` and written `livekit/<name>`
for short. They need no extension negotiation; the card's
`capabilities.extensions` entry is only a capability hint and grants no thread
access ([explicit authority](protocol/a2a.md#delegation-and-conversation-data)).
A stock client ignores them.

| Key                                                 | Meaning                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `livekit/channel`                                   | the channel a message came on; `a2a` when absent                                             |
| `livekit/kind`                                      | `delegation` on a message another agent sent                                                 |
| `livekit/interrupted`, `livekit/interruptionReason` | on a `canceled` task the server ended for a newer message                                    |
| `livekit/mergedInto`                                | on a `completed` task whose message was answered by another task's turn                      |
| `livekit/reason`                                    | why a task failed: `thread-not-found`, `lease-timeout`, `worker-lost`, `storage-unavailable` |
| `livekit/verbatim`                                  | the part of an answer or a question that reaches the person exactly as written               |

`livekit/mergedInto` marks three cases: a message a `batch` turn answered
with a later one, a message that continued a waiting task, and a waiting
task a new message answered.


Endpoint clients and browser access follow the [client API](clients.md).
