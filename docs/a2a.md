# The A2A wire

A thread endpoint speaks the Agent2Agent protocol, version 1.0, plus the
LiveKit LiveKit agents extension. Any A2A client works with no LiveKit code.
This is the reference for a client, a partner agent, or a system that
answers a task.

## The agent card

Every endpoint publishes `<endpoint>/.well-known/agent-card.json`:

| Field            | Value |
| ---------------- | ----- |
| `name`           | the endpoint's name |
| interface        | one, `JSONRPC` (JSON-RPC 2.0 over HTTP, server-sent events for the streams) at `https://<project>.livekit.cloud/agents/<agent>/<deployment>/<endpoint>/v1` |
| `version`        | the deployment's, or `a2a_session(version=)` |
| security schemes | `livekit`, a bearer JWT (a LiveKit project token or a thread grant), always listed; marked required only when `public=False` |
| skills           | one, `delegate`; its description is the endpoint's `description=`, what another agent reads to decide whether to call |
| extensions       | `https://livekit.io/a2a/ext/agents/v1`, not required |

Endpoints are public by default: reachable without a LiveKit token. A
`public=False` endpoint answers only callers holding a project token:
backends and delegating agents. `auth=` is the developer's own check: a
function of the request, sync or async, run on every request; `False`
refuses it with 401. Both apply together.

## Threads and messages

The thread is A2A's `contextId`, and the message body is the only place a
thread is named. A first message with no `contextId` starts a new thread,
and the `Task` that answers it carries the id; send it on every message
after. A `contextId` the threads service does not know ends the task `failed` with
`livekit/reason: thread-not-found`.

A thread is found by its id and nothing else. A message never carries a
key; the platform's own ingress finding a thread by phone number is
planned ([planned-channels.md](planned-channels.md)).

Every message carries a `messageId`; a resent message with the same id
returns the task the first one created and runs nothing.

Every LiveKit client sends `A2A-Version: 1.0`.

## Tasks

Every message is a task in the thread's context. States: `submitted`,
`working`, `input-required`, `completed`, `failed`, `canceled`, `rejected`;
the last four are terminal. On the wire they are `TASK_STATE_SUBMITTED` and
so on.

| Method (1.0 name, 0.3 name)                    | What it does here |
| ---------------------------------------------- | ----------------- |
| `SendMessage`, `message/send`                  | a turn; the response is `{"task": ...}` once it ends |
| `SendStreamingMessage`, `message/stream`       | the same, as server-sent events, in the order below |
| a message with `taskId`                        | continues that task while it is not terminal; see "Answering a task" |
| `GetTask`, `tasks/get`                         | the task, with `historyLength` |
| `ListTasks`, `tasks/list`                      | every task of the thread named by `contextId`, by last update, filtered by `status`, paged by `pageSize` and `pageToken` |
| `CancelTask`, `tasks/cancel`                   | `canceled` if not terminal, else `TaskNotCancelableError`; a running turn stops at its next safe point |
| `SubscribeToTask`, `tasks/resubscribe`         | the current `Task`, then the task's stream from its start, followed live while the task runs; from any worker |

`GetTask`, `CancelTask` and `SubscribeToTask` find the thread from
`contextId` in their params, beside `id`. A2A's task methods carry only the
task id, so a stock client that sends just the id gets `TaskNotFoundError`.
A resubscribe resumes after an offset, the SSE `Last-Event-ID` header or
`offset` in the params.

A stream is, in order:

1. the `Task` in state `submitted`, sent once the task's row is durable, so a
   resubscribe on another worker finds it;
2. zero or more `statusUpdate` events in state `working`, each with a text
   message: what a handler or a tool sent with `ctx.update()` or `ctx.say()`;
3. zero or more `artifactUpdate` chunks of the one artifact named `answer`,
   the last with `lastChunk: true`;
4. the final `statusUpdate`, with the metadata keys that apply.

```json
{"jsonrpc": "2.0", "id": 1, "result": {"task": {"id": "AST_1", "contextId": "AT_x", "status": {"state": "TASK_STATE_SUBMITTED"}}}}
{"jsonrpc": "2.0", "id": 1, "result": {"statusUpdate": {"taskId": "AST_1", "contextId": "AT_x", "status": {"state": "TASK_STATE_WORKING", "message": {"messageId": "msg_1", "role": "ROLE_AGENT", "parts": [{"text": "Checking the schedule."}]}}}}}
{"jsonrpc": "2.0", "id": 1, "result": {"artifactUpdate": {"taskId": "AST_1", "contextId": "AT_x", "artifact": {"artifactId": "art_1", "name": "answer", "parts": [{"text": "Tuesday at 10 is free."}]}, "lastChunk": true}}}
{"jsonrpc": "2.0", "id": 1, "result": {"statusUpdate": {"taskId": "AST_1", "contextId": "AT_x", "status": {"state": "TASK_STATE_COMPLETED"}}}}
```

Timestamps are omitted above. A message with a `taskId` on a terminal task
is an `UnsupportedOperationError`: a follow-up is a new task in the same
context, with `referenceTaskIds` if it builds on an earlier one.

## Answering a task

A turn whose tool returned `InputRequired(question)` ends `input-required`.
The question is the turn's reply, as the status message; no model call
writes it. The pending call stays in the history with a placeholder output,
`waiting for the person's answer to: <question>`.

The answer continues the waiting task, on whatever worker takes it: the
same task and stream, with the message added to its history. A message with
the task's `taskId` continues it, and so does the person's message with none
(the oldest open question of the lane); a delegation with none, or with the
`taskId` of a running task, is a new task.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "m2",
      "role": "ROLE_USER",
      "contextId": "AT_x",
      "taskId": "AST_2",
      "parts": [{ "text": "Walgreens on Main Street" }]
    }
  }
}
```

## Metadata keys

The extension puts its fields in `metadata` under keys prefixed by its URI,
`https://livekit.io/a2a/ext/agents/v1/<name>`, written
`livekit/<name>` below. [a2a-live-ext.md](a2a-live-ext.md) is the
specification; this is the summary. A stock client that did not activate
the extension still receives the server-set keys and may ignore them.

| Key                          | On | Value |
| ---------------------------- | -- | ----- |
| `livekit/channel`            | a message | the channel it came from; absent means `a2a` |
| `livekit/kind`               | a message, or a data part | `delegation` when another agent asks on its user's behalf; on a data part, `chat_ctx` for a caller's conversation |
| `livekit/interrupted`        | the final status of a `canceled` task | `true` when the server ended the turn for a newer message |
| `livekit/interruptionReason` | the same | `new-message` |
| `livekit/mergedInto`         | the final status of a `completed` task | the task whose turn answered this message too |
| `livekit/verbatim`           | a text part | `true`: the part reaches the person exactly as written; see [a2a-live-ext.md](a2a-live-ext.md#exact-text) |
| `livekit/reason`             | a `tasks/cancel` request and the final status of the task it canceled; the status message of a `failed` task | why: the client's words on a cancel, else the reasons below |

A turn ended because the person sent a second message while it ran:

```json
{
  "statusUpdate": {
    "taskId": "AST_1",
    "contextId": "AT_x",
    "status": { "state": "TASK_STATE_CANCELED" },
    "metadata": {
      "https://livekit.io/a2a/ext/agents/v1/interrupted": true,
      "https://livekit.io/a2a/ext/agents/v1/interruptionReason": "new-message"
    }
  }
}
```

A text answered together with the one that followed it:

```json
{
  "statusUpdate": {
    "taskId": "AST_1",
    "contextId": "AT_x",
    "status": { "state": "TASK_STATE_COMPLETED" },
    "metadata": {
      "https://livekit.io/a2a/ext/agents/v1/mergedInto": "AST_2"
    }
  }
}
```

## Tokens

Two credentials reach a thread endpoint. The project credential, any
token the project's key signed that carries no thread grant, is unscoped.
A thread grant, minted by the application's backend for a browser or a
device the way a room token is ([backend-api.md](backend-api.md)), scopes
everything to one scope, the subject the threads belong to:

```json
{"agentThread": {"scope": "user:42",
                 "create": true, "write": true, "read": true}}
```

`threadIds` in place of `scope` scopes the token to listed threads; a
grant with neither is refused. A thread belongs to the project, not to one
agent: any agent of the project continues it, and the grant names no
endpoint. Which endpoints a token may call is the `agentEndpoint` grant's;
a token with only `agentThread` reaches public endpoints.

| Flag     | Allows |
| -------- | ------ |
| `create` | a first message with no `contextId`: a new thread in the grant's scope, which the request never carries; a grant with no scope cannot create |
| `write`  | a message, `tasks/cancel` |
| `read`   | `tasks/get`, `tasks/list`, `tasks/resubscribe`, and the thread's history and tasks from the platform's threads API ([backend-api.md](backend-api.md)) |
| `list`, `delete` | the threads service's listing of the scope and deletion, for a browser ([planned-backend-api.md](planned-backend-api.md)) |

A thread of another scope is refused.

Through LiveKit Cloud the node verifies the project credential and strips
it, and the worker reads its verdict from the request's preamble; a thread
grant is a LiveKit bearer the node strips too, so in v0 it reaches the
platform's threads API, not an endpoint through the node. A worker reached
directly verifies the token itself, from `Authorization: Bearer`, an
`X-LiveKit-Thread-Grant` header, or an `access_token` query parameter for
a browser `EventSource`, which cannot set headers.

A refusal is an HTTP status with `{"error": "..."}`, not a JSON-RPC error:
401 for no token where one is needed, a token the project did not sign, or
`auth=` returning `False`; 403 for a grant that does not cover the call.

## The extension

A LiveKit client activates it only toward a card that offers it:
`AgentClient`, `A2ADelegate` and a `LiveSession` delegating to an endpoint
then send `A2A-Extensions: https://livekit.io/a2a/ext/agents/v1`
without the developer doing anything, and the server echoes it. With it,
a delegation is marked `livekit/kind: delegation`, and the caller's thread
id goes as the `contextId`, so the expert works on its own lane of the
caller's thread. A stock A2A client that omits the header gets the same
endpoint as a plain text agent, which is what `required: false` on the card
means. The keys the server sets, such as `livekit/interrupted` and
`livekit/mergedInto`, are sent either way, since `metadata` is part of core
A2A.

## Writing an expert in another framework

An endpoint delegates to any A2A server. What the server implements decides
what it gets:

| The expert implements | What the caller sends and does |
|---|---|
| A2A only, over JSON-RPC with SSE | the request as text, with no `contextId` and nothing of the extension; its `working` status messages are the progress; the artifact text is the result; `failed` is a failure |
| the extension, listed on its card | `livekit/kind: delegation` and the caller's thread id as `contextId`; with no thread to share, the caller's conversation as a `chat_ctx` data part |

The caller never depends on a `livekit/` key coming back: with or without
the extension, the working statuses are the progress, the artifact is the
answer, `input-required` is a question for the person, and the terminal
states end the task.

## Errors

| Error                       | When |
| --------------------------- | ---- |
| `InvalidParamsError`        | a message with no `messageId`, params that are not an object |
| `TaskNotFoundError`         | an unknown task id; a `taskId` not in the message's thread; `tasks/*` with no `contextId` |
| `TaskNotCancelableError`    | `tasks/cancel` on a terminal task |
| `UnsupportedOperationError` | a message to a terminal task; a message to a thread whose turn runs, on an endpoint whose policy is `reject` |
| `MethodNotFoundError`, `InvalidRequestError`, `JSONParseError` | a malformed request; a batch request |

JSON-RPC errors are for malformed requests. Everything else ends the task
`failed`, with `livekit/reason` on its status message:

| Reason                | When |
| --------------------- | ---- |
| `thread-not-found`   | a `contextId` the threads service does not know |
| `worker-lost`         | the worker holding the turn died; resend the message |
| `lease-timeout`       | on an `interrupt` or `queue` endpoint, the thread's lane was not free within 60 seconds |
| `storage-unavailable` | the turn's save could not be written; resend the message |

See `examples/use-cases/03_insurance_eligibility.py` for a delegation over
A2A and `04_a_claim_with_a_photo.py` for a task answered later.
