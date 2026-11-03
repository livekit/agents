# AgentServer and entrypoints

`AgentServer` serves application entrypoints. Modules can declare doors on
an `AgentHandler` and the server assembles them. Text endpoints receive A2A
requests; voice entrypoints attach a `VoiceSession` to a room. Both receive
runtime thread/storage access, while application context loading and saving
remain explicit.

Voice entrypoints accept `on_session_end`, called with the `JobContext` once
the job's session has ended, and `on_simulation_end`, called with a `SimulationContext`
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
and `question` to A2A `input-required`, `status="completed"` and `output`
to a completed task artifact, or `status="no_output"` to a completed task with
no artifact. The handler saves application fields explicitly
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

`lane=` names the lane its text doors write; left out, the worker's own. A
handler with a lane of its own keeps its conversations apart from the
worker's on the same thread ([lane mechanics](internals/storage.md#lanes)), which is how two
agents share one process in `examples/`. Such a handler has text doors only:
the voice door, the call's fast brain, takes no lane and writes the worker's,
so `rtc_session()` on a handler with `lane=` raises. `livekit.agents.a2a_session` and
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

The decorator returns an `A2AEndpoint`: its `name`, its `url` once the server knows its
project and deployment, and `card()`, the agent card as JSON. Pass it wherever a delegate
or a client wants the endpoint.

`AgentServer(preload_modules=[...])` imports preload modules in the supplied
order before job readiness. Import failures stop initialization and identify
the module; see [Python preloading](internals/rust-core.md#python-preloading).

The Python entrypoint runs the server with `cli.run_app(server)`. The shared
`lk-agents-cli` crate parses `start` and `console`; the Go `lk agent` command
owns development reloads, console interaction, and terminal presentation:

```python
from livekit.agents import AgentServer, cli

server = AgentServer(storage=AgentDBStorage())
server.add(front_desk, back_office)

if __name__ == "__main__":
    cli.run_app(server)
```

| `AgentServer` option | What |
| --- | --- |
| `agent_name`, `storage`, `preload_modules` | see above and below |
| `ws_url`, `api_key`, `api_secret` | the LiveKit project; default `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` |
| `num_idle_processes` | job processes kept warm ahead of dispatch; `min(ceil(cpus), 4)` in production, none in development |
| `load_threshold` | load at which the worker stops taking jobs; production default 0.7 |
| `job_memory_warn_mb`, `job_memory_limit_mb` | a job process's memory warning and kill thresholds |
| `drain_timeout` | maximum seconds to wait for active work after stopping admission; default 3600; zero skips waiting |
| `shutdown_timeout` | total process shutdown budget in seconds, including draining and cleanup; default 3610; must exceed `drain_timeout` |
| `listeners` | the worker's `public`, `health` and `prometheus` listeners; see [Listeners](#listeners) |

## Worker load and admission

The default load signal is CPU utilization normalized to effective CPU capacity, averaged
by elapsed time over 2.5 seconds. CPU counters refresh at most once per 500 ms, including
during dispatch bursts. Linux accounting uses the process's cgroup, visible ancestor quotas
and CPU sets, preserving fractional CPU limits in both cgroup versions. `NUM_CPUS` explicitly
overrides detected capacity.

Admission reserves at least 0.175 normalized CPU load for each pending job, increasing the estimate when
observed per-job utilization is higher. A launched job retains an additional minimum startup reservation
for five seconds, covering the CPU measurement window. This temporarily counts some
measured startup CPU twice to protect existing sessions while the new load settles. Prewarming controls process readiness;
it does not set admission capacity. The per-job estimate is internal and does not establish
a concurrent-session limit.

A worker stops admitting jobs at `load_threshold` and resumes below 90% of that threshold
(0.63 with the default). Status reports and individual job offers use the same hysteresis.
Admission waits for the first valid CPU measurement and stops if measurements are five
seconds old. Invalid custom load values do not refresh that deadline; a custom load callback
has a five-second timeout. Offers waiting for measurements expire within five seconds.
Draining always stops admission, including when capacity limits are disabled.

CPU utilization does not establish memory or latency capacity. Qualify the worker and its
`load_threshold` under representative sustained load and dispatch bursts, observing session
latency and memory as well as CPU. Scale out below the admission threshold so new workers
become ready before existing workers fill up.

## Listeners

A worker has three listeners, each with its own host and port: public traffic never shares
a port with operational information.

```python
server = AgentServer(
    listeners={
        "public": {"host": "127.0.0.1", "port": 8080, "proxy_headers": True},
        "health": {"host": "0.0.0.0", "port": 8081},
        "prometheus": {"host": "0.0.0.0", "port": 9090},
    }
)
```

| Listener | Serves | Opens | Default |
| --- | --- | --- | --- |
| `public` | agent endpoints and [HTTP routes](#http-routes) | when the worker serves its own endpoints: registered with a self-hosted LiveKit server, or under `lk agent console`; on LiveKit Cloud, Cloud routes them | `0.0.0.0:8080` |
| `health` | `/`, `/ready` and `/worker` | always | `0.0.0.0:8081` |
| `prometheus` | `/metrics` | when given | off |

Each listener takes `host` and `port`; `public` also takes `proxy_headers`. A key or a whole
listener left out keeps its default; an unknown listener or key is refused when the server
is built. In development and the console, every listener binds `127.0.0.1` on a free port.
The worker logs every listener at startup, including the ones that are off, with its address
or the reason it is off.

`proxy_headers` (default `False`) reads the client address and scheme from
`X-Forwarded-For` and `X-Forwarded-Proto`. Enable it only when every request reaches the
worker through a proxy that sets these headers; otherwise a client can choose the address
and scheme it is seen with.

The health and metrics listeners are unauthenticated and report the SDK version, load and
running jobs. Kubernetes probes and Prometheus reach them by pod IP: never expose them
through a Service or Ingress, and restrict them with a NetworkPolicy where the cluster
supports one.

### Health and metrics

`/` answers whether the worker is up, for startup and liveness probes; `/ready` whether it
should receive traffic, for readiness probes. A failure is `503` with the state as its body:

| State | `/` | `/ready` |
| --- | --- | --- |
| starting: not registered yet, or the job process pool has not filled for the first time | `503 starting` | `503 starting` |
| ready: registered, and the pool has warmed its idle processes once | `200 OK` | `200 OK` |
| reconnecting after a dropped connection | `200 OK` | `503 reconnecting` |
| draining | `200 OK` | `503 draining` |
| failed: reconnection gave up, or the pool cannot start | `503 failed` | `503 failed` |
| stuck: the host's event loop has not responded for 5 seconds | `503 stuck` | `503 stuck` |

The pool counts once: later refills after dispatches leave readiness unchanged, and with no
idle processes (development) a started pool is enough. Load does not change readiness; a
full worker reports it to LiveKit through its status, and dispatch avoids it. `/` fails only
on a failure a restart can fix, never on a drain or a dropped connection, so a LiveKit
outage never restarts workers. The host refreshes a heartbeat every 500 ms and the probes
read it without waiting on the host. `/ready` fails as soon as draining starts, so the pod
leaves rotation before it stops; set `terminationGracePeriodSeconds` above
`shutdown_timeout`.

```yaml
ports:
  - { name: health, containerPort: 8081 }
startupProbe:   { httpGet: { path: /, port: health }, periodSeconds: 1, failureThreshold: 300 }
livenessProbe:  { httpGet: { path: /, port: health }, periodSeconds: 1, failureThreshold: 5 }
readinessProbe: { httpGet: { path: /ready, port: health } }
```

LiveKit Cloud runs the startup and liveness probes above and routes dispatch through
registration, so it sets no readiness probe.

`/worker` uses the protocol's generated protobuf JSON serialization: camelCase keys and
omitted default values. These operational
endpoints accept `GET` and `HEAD`; other methods receive `405` with an `Allow` header.
`HEAD` reports the corresponding `GET` content length without sending its body.

The health and metrics listeners each admit at most 64 connections, accepts request heads up to 16 KiB and
64 headers, and serves one request per connection. Request headers must arrive within
5 seconds; the whole exchange, including probe callbacks and response writes, has a
10-second deadline. Requests with bodies, transfer encoding, invalid hosts or ambiguous
framing are rejected before host callbacks run. Shutdown cancels and joins active
connections. HTTP parsing, response headers and framing are handled by Hyper.

## Launching and shutdown

```bash
python main.py start --shutdown-timeout 55 --drain-timeout 45
python -m livekit.agents start main.py --shutdown-timeout 55 --drain-timeout 45
```

Both forms execute the entrypoint's `if __name__ == "__main__"` block, which calls
`cli.run_app(server)`. The module form is a Python-only adapter: it forwards the
command arguments and makes sibling imports available as a direct script launch
would. It does not search the module's globals for a server. The Go CLI can use
either form without changing the shared Rust CLI.

`start` accepts `--url`, `--api-key`,
`--api-secret`, `--log-level`, `--dev`, `--simulation`, `--cli-addr`,
`--shutdown-timeout`, and `--drain-timeout`. CLI timeout values override server
configuration. `console` accepts required `--connect-addr` and `--record`. Both commands
operate on the server supplied to `cli.run_app(server)`. Help and invalid
arguments are handled before starting that server. The launcher has no `download-files` command.

The process emits JSON logs in every mode. The Go CLI owns colors and layout.
For compatibility with existing Go launchers, `--reload-addr` aliases
`--cli-addr`, and the hidden `--log-format` option is accepted without changing
JSON output. Neither option adds another rendering or execution path.

On SIGTERM, SIGINT, or SIGHUP, the native signal listener starts one monotonic
shutdown deadline and notifies the host. The host stops accepting work through the
worker's drain operation, waits for active work, closes remaining resources, and
finalizes its event loop. The drain phase reserves
`shutdown_timeout - drain_timeout` seconds for closure and cleanup. Time spent
waiting for the host to respond consumes that same budget. Early drain completion
makes its unused time available for cleanup. Repeated SIGTERM never restarts the
deadline; repeated Ctrl+C also keeps the original deadline.

The CLI owns process signal handling for its lifetime; applications must not replace
these handlers. Only one launcher can listen at a time. Registration failures abort
startup before the worker accepts work. A worker with agent endpoints or HTTP routes
registers over WebTransport (QUIC over UDP): when the server never answers the handshake, a
server without WebTransport (LiveKit Cloud production today) or a network that drops UDP,
startup fails after the connection timeout and says so. Every other failed connection attempt
is logged with its error and the next retry. The native listener starts the deadline
even when the host event loop is blocked before receiving the shutdown notification.

A native deadline remains responsive while host tasks, callbacks, executor
threads, or log output are blocked. If the deadline expires, the process exits
nonzero. Successful cleanup disarms it only after the host loop closes. It does
not reset or add a new cleanup timeout after draining.

Configure the application budget below the deployment's termination grace period,
leaving room for any `preStop` hook and an operational margin. A signal carries
no Kubernetes grace-period value; the framework does not query Kubernetes or
assume its default. Forced termination can still preempt application cleanup.

Signal capture, parsing, normalized log records, the remote CLI connection, and
deadline enforcement are shared host-neutral Rust services. Each language host supplies its already
constructed server; the shared Rust CLI never searches for or imports application modules.
Python logging conversion and asyncio ownership remain in the Python adapter.

The agent is named once, `AgentServer(agent_name=)`: the lane every door of
the server writes, and the first segment of every door's address,
`/agents/{agent_name}/{deployment}/{endpoint}`. `LIVEKIT_AGENT_NAME` is the
default and `LIVEKIT_AGENT_NAME_OVERRIDE` a launcher's force. The voice door,
`@server.rtc_session(endpoint=)`, is named the same way, the agent's name when
left empty.

The endpoint is served at `/agents/<agent>/<deployment>/<endpoint>`, with
`default` as the deployment of a worker registered without one. Where it is served
follows the server the worker registers with: LiveKit Cloud when the worker is hosted there
or its URL is a Cloud project's, decided before the job processes start so they delegate
alike. The registration's edition is checked against it, and a mismatch is logged:

| Worker | Served by | Authentication |
| --- | --- | --- |
| registered with LiveKit Cloud | Cloud, at `https://<project>.livekit.cloud/agents/...` | Cloud verifies the bearer token |
| registered with a self-hosted LiveKit server | the worker's public listener | the worker verifies the bearer token with the project's API secret |
| `lk agent console` (not registered) | the worker's public listener on `127.0.0.1` only | **none**: every caller holds every thread permission, new threads in the `console` scope; the worker logs that its endpoints are unauthenticated |

Every
endpoint publishes an agent card at `.well-known/agent-card.json`, so a
stock A2A client works with no LiveKit code: it sends a first message, gets
a `contextId` back, and sends it from then on. The card declares a bearer
scheme for a LiveKit project token or thread grant, always listed and
required.

The endpoint serves the conversation and nothing else. A thread's history
and tasks are read from outside through the platform's threads service;
see [clients.md](clients.md).

## HTTP routes

`server.http` is a Starlette application for routes served alongside the
agent, such as a token endpoint for a web client:

```python
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

@server.http.get("/token")
async def token(request: Request) -> Response:
    return JSONResponse({"token": await mint_browser_token(request)})
```

`get`, `post`, `put`, `patch`, `delete`, `route` and `websocket` register
handlers, and `mount(path, app)` mounts any ASGI application. Routes are served
at `/agents/{agent_name}/{deployment}/{path}`, where the worker's endpoints are
(see [Launching and shutdown](#launching-and-shutdown)); a handler reads the request's LiveKit
preamble as `request.state.lk_request`. Read the client host, without a port, from
`request.state.lk_request.client_addr`. The ASGI `client` value is `None`: through
Cloud, the worker does not know the original client socket address.

On Cloud, the worker registers the route manifest and Cloud delivers each incoming
request as metadata and a bidirectional byte stream after the node preamble. The public
listener produces the same metadata and stream for each connection it accepts, so
routes and endpoints run unchanged on either path.
It does not parse HTTP, implement WebSocket framing or expose an ASGI contract.
Each language binding supplies its application server. Python serves these
streams through ASGI using h11 and wsproto, including request limits, response
head deadlines, completion trailers and WebSocket draining.

The worker handoff is in [`lk-agents-worker/src/http.rs`](../rust/crates/lk-agents-worker/src/http.rs),
with its shared contract in [`lk-agents-abi/src/worker/http/mod.rs`](../rust/crates/lk-agents-abi/src/worker/http/mod.rs).
Python route authoring and ASGI serving are together in
[`livekit/agents/http.py`](../livekit-agents-next/livekit/agents/http.py).
`AgentRoutes` defines the routes; its `_AsgiServer` owns the ASGI app, stream
serving and draining. The server itself is passed to Rust's handler contract,
without a separate adapter object.

A bounded inbox owns queued WebSocket messages, their byte reservations and
wakeups. Receiving transfers a message and releases its reservation without
suspending; cancellation only interrupts the wait. Connection cleanup joins the
reader and shared close operation before releasing the connection's capacity.

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
once: each endpoint says what happens, per channel. A message with
`livekit/kind: delegation` selects `delegation`, even when `livekit/channel`
is set. Otherwise the channel is `livekit/channel`, or `a2a` when absent.

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
`tasks/list` and `tasks/cancel` it from any worker at
once. Each of these finds the thread from `contextId` in its params: A2A's
task methods carry only the task id, so a stock client that sends just the
id gets `TaskNotFoundError`. Responses stream live on their original request.
After a disconnect, use `tasks/get` for the saved task outcome. Response events
are not retained; `tasks/resubscribe` is unsupported.

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
for short. The metadata keys need no extension negotiation; the card's
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
