# A2A protocol

LiveKit endpoints use A2A 1.0 over JSON-RPC 2.0 and HTTP server-sent events.
The [upstream A2A definitions](https://a2a-protocol.org/latest/definitions/)
are the reference for messages, tasks, parts, artifacts, agent cards, and
protocol errors. This page defines LiveKit's endpoint behavior and additions.

LiveKit adds thread addressing, channel and delegation metadata, interruption
and merge information, an optional typed conversation payload, and thread-scoped
authorization. These use A2A metadata and data parts, plus the additional task
request parameters listed below. They do not add task states or part types.

LiveKit fields apply directly to LiveKit endpoints. No `A2A-Extensions` request
or response header, or `capabilities.extensions` card entry, is required.
Clients may ignore unknown metadata and still consume text, progress, artifacts,
and task states. Task lookup requires the additional `contextId` parameter;
clients must supply it as described below.

## The agent card

Every endpoint publishes `<endpoint>/.well-known/agent-card.json`:

| Field            | Value |
| ---------------- | ----- |
| `name`           | the endpoint's name |
| interface        | one, `JSONRPC` (JSON-RPC 2.0 over HTTP, server-sent events for the streams) at `https://<project>.livekit.cloud/agents/<agent>/<deployment>/<endpoint>/v1` |
| `version`        | `a2a_session(version=)`, else the deployment the worker registered under (`LIVEKIT_AGENT_DEPLOYMENT`); a local worker registered under none uses `LIVEKIT_AGENT_VERSION`, else `1.0.0` |
| security schemes | `livekit`, a bearer JWT (a LiveKit project token or a thread grant), always listed and required |
| skills           | one, `delegate`; its description is the endpoint's `description=`, what another agent reads to decide whether to call |
| `capabilities.extensions` | optional: `https://livekit.io/a2a/ext/agents/v1`, a capability hint that the endpoint reads LiveKit fields; it grants nothing |

The extension entry may carry `params` declaring the tool a caller's model
sees for the endpoint; `RemoteAgent(url).as_tool()` reads them:

| Param | Type and contract |
| --- | --- |
| `toolName` | string: the tool's name; absent, the card's `name` |
| `inputSchema` | JSON Schema object for the tool's arguments; absent, `{"request": string}` (required), sent as the message's text. With it, the arguments go as the message's first part, a `DataPart` |

A LiveKit endpoint lists the entry without `params`. Spelling and case are
significant; other `params` keys are ignored.

Every endpoint requires a token: a project token or a thread grant. `auth=`
is the developer's own check on top: a function of the request, sync or async,
run on every request; `False` refuses it with 401.

## Threads and messages

The thread is A2A's `contextId`, in messages. Task lookup, cancellation, and subscription also carry
the thread ID in their request parameters. A first message with no `contextId` starts a new thread,
and the `Task` that answers it carries the id; send it on every message
after. A `contextId` the threads service does not know ends the task `failed` with
`livekit/reason: thread-not-found`.

An A2A message addresses a thread by ID. The platform resolves phone numbers
to thread IDs before dispatch; see [channel routing](../channels.md#channel-routing).

Every message carries a `messageId`; a resent message with the same id
returns the task the first one created and runs nothing. A retried first
message, the same `messageId` with no `contextId`, returns the same thread id
and task, so the client always learns the thread.

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
| `CancelTask`, `tasks/cancel`                   | `canceled` if not terminal, else `TaskNotCancelableError`; a running turn stops at its next safe point; canceling a reply that continued an `input-required` task returns that task to `input-required`, its question still pending |
| `SubscribeToTask`, `tasks/resubscribe`         | the current `Task`, then the task's stream from its start, followed live while the task runs; from any worker |

`GetTask`, `CancelTask` and `SubscribeToTask` find the thread from
`contextId` in their params, beside `id`. A2A's task methods carry only the
task id, so a stock client that sends just the id gets `TaskNotFoundError`.
A resubscribe resumes after an offset, the SSE `Last-Event-ID` header or
`offset` in the params. The framework mints the task id independently of any
delivery stream; the task records durable references to its streams, so a
resubscribe resolves the task's full ordered history.

Replay retention is configurable. The deployment documents its minimum replay
window; an expired cursor or stream returns an explicit expired outcome rather
than an empty successful replay. Deleting the thread also removes its delivery
history.

A stream is, in order:

1. the `Task` in state `submitted`, sent once the task's row is durable, so a
   resubscribe on another worker finds it;
2. zero or more `statusUpdate` events in state `working`, each with a text
   message: the progress a handler or a tool sent with `ctx.update()`;
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

A Python run result separates framework status from application output.
`status="input_required"` maps to A2A `input-required`, with `result.question`
as the status message. `status="completed"` maps to `completed`, with the
accepted `result.output` delivered as an artifact. The application's
`output_type` describes that accepted output. See
[run results](../agents.md#running-a-turn).

A turn whose tool returned `InputRequired(question)` ends `input-required`.
The question is the turn's reply, as the status message. A model-driven agent
facing the person may phrase it once under the
[wording contract](../tools.md#wording-and-verbatim); the endpoint sends that
final text without another model call. A handler without a model sends the
question as written. The pending call stays in the history with a placeholder output,
`waiting for the person's answer to: <question>`.

The next message in the thread answers it: its text replaces the
placeholder, and the model continues from the history, on whatever worker
takes the message.

- With the waiting task's `taskId`, the answer comes back on that task.
- Without one, the answer is a new task, and the waiting task ends
  `completed` with `livekit/mergedInto` naming it.

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

On the wire, each LiveKit key is the full string
`https://livekit.io/a2a/ext/agents/v1/<name>`. Tables use `livekit/<name>`
as shorthand; that shorthand is not a wire key. The URI is a metadata
namespace, not a requirement to negotiate an A2A extension. Values are JSON;
key spelling and case are significant. Unknown keys have no LiveKit semantics
and must not be interpreted as instructions or authorization.

| Key | Exact location | Writer | Type and meaning |
| --- | --- | --- | --- |
| `livekit/channel` | `Message.metadata` | ingress/client | String channel name; absent means `a2a`. Used to select the endpoint's request policy. |
| `livekit/kind` | `Message.metadata` | delegating caller | String `delegation`: an agent is asking on a person's behalf. Set on every delegated message, never a person's direct message. Overrides channel selection to `delegation`. |
| `livekit/kind` | `Part.metadata` on the conversation data part | delegating caller | String `chat_ctx`; identifies the payload defined below. |
| `livekit/reason` | `CancelTask` request `params.metadata` | client | Free-text cancellation reason; `caller-canceled` when a delegating caller cancels. |
| `livekit/reason` | `Task.metadata` and final `TaskStatusUpdateEvent.metadata` after cancellation | server | String cancellation reason supplied by the client. |
| `livekit/reason` | failed task's `status.message.metadata` | server | String failure code from the errors table below. |
| `livekit/interrupted` | `Task.metadata` and final canceled `TaskStatusUpdateEvent.metadata` | server | Boolean `true` when a newer message interrupted this turn. Absent on ordinary cancellation. |
| `livekit/interruptionReason` | same locations as `interrupted` | server | String `new-message`. |
| `livekit/mergedInto` | `Task.metadata` and final completed `TaskStatusUpdateEvent.metadata` | server | String task ID of the turn that answered this message too. |
| `livekit/verbatim` | `Task.metadata` and final completed or input-required `TaskStatusUpdateEvent.metadata` | server | String prefix of the answer/question to deliver exactly as written; the frontend may phrase the remaining text. |

Server-owned fields cannot be set by a caller to alter task outcomes. An absent
field makes no claim: in particular, `completed` with `mergedInto` means the
answer belongs to the referenced task, rather than an empty successful answer.
For interrupted turns, earlier artifact chunks are partial; the replacement
answer belongs to the newer task. Terminal task records do not change.

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

Two credentials reach an endpoint. The project credential is a token the
project's key signed carrying the `agentEndpoint` call grant
(`Credentials.project_token()`). It invokes the endpoint and reads its card,
but grants no thread access: a message or a task method under it alone is
refused 403. A token the project signed with neither grant, such as a room
join token, is refused 403. A thread grant, minted by the application's
backend for a browser or a device the way a room token is
([browser thread access](../clients.md#from-the-browser)), scopes everything
to one scope, the subject the threads belong to:

```json
{"agentThread": {"scope": "user:42",
                 "create": true, "write": true, "read": true}}
```

`threadIds` in place of `scope` scopes the token to listed threads; a
grant with neither is refused unless `all` is true. A thread belongs to the project, not to one
agent: any agent of the project continues it, and the grant names no
endpoint. A token with only `agentThread` may call every endpoint of the
project: its card, and messages and task methods on the threads the grant
covers, as its flags allow. An `agentEndpoint` grant naming an agent or a
deployment limits where the project credential is admitted; it does not
narrow a token that also carries `agentThread`.

The `agentThread` claim is a JSON object. `scope` is a string, `threadIds`
is an array of string IDs, and `all` is a boolean (default false) extending
access to every thread and scope in the project. Permission flags are booleans
and default to false; `all` does not enable missing permissions. These are
LiveKit JWT claims, not A2A metadata. `agentEndpoint.call` is a separate boolean
permission to invoke an endpoint; it does not grant access to thread data.

| Flag     | Allows |
| -------- | ------ |
| `create` | a first message with no `contextId`: a new thread in the grant's scope, which the request never carries; a grant with no scope cannot create |
| `write`  | a message, `tasks/cancel` |
| `read`   | `tasks/get`, `tasks/list`, `tasks/resubscribe`, and the thread's history and tasks from the platform's threads API ([browser thread access](../clients.md#from-the-browser)) |
| `list`, `delete` | the threads service's listing of the scope and deletion, for a browser ([browser thread access](../clients.md#from-the-browser)) |

A thread of another scope is refused.

Through LiveKit Cloud, the node verifies and strips bearer credentials,
and passes the authorization verdict to the worker. A worker reached
directly verifies the token itself, from an `Authorization: Bearer` or an
`X-LiveKit-Thread-Grant` header. A browser streams with `fetch`, a
JSON-RPC `POST` carrying the bearer header.

A refusal is an HTTP status with `{"error": "..."}`, not a JSON-RPC error:
401 for no token, a token the project did not sign, or
`auth=` returning `False`; 403 for a grant that does not cover the call.

## Delegation and conversation data

A caller shares its thread with an endpoint only under explicit authority: a
same-project grant token the endpoint verifies as covering that thread, or trust
configured on the caller and verified locally. The destination uses its
own lane and application context. A metadata key, an agent card or its
`capabilities.extensions` entry alone does not establish project membership or
grant access to history.

For an arbitrary A2A server, the first request contains text and `messageId`,
without the caller's thread ID or LiveKit-specific data. Retain the remote
service's returned `contextId` and pending task ID for subsequent replies.
`working` messages provide progress, artifacts provide results, and
`input-required` provides a question. These states work without LiveKit metadata.

### Conversation data part

A delegating caller with no thread to share may attach its conversation
so far as one `DataPart` on the message, tagged `livekit/kind: chat_ctx`, to
an endpoint whose card lists the LiveKit extension; other servers receive text
only:

```json
{
  "metadata": {
    "https://livekit.io/a2a/ext/agents/v1/kind": "chat_ctx"
  },
  "data": {
    "items": [
      {
        "id": "item_1",
        "type": "message",
        "role": "user",
        "content": ["Is my visit covered?"],
        "created_at": 1791322277.4
      },
      {
        "id": "item_2",
        "type": "message",
        "role": "assistant",
        "content": ["Let me check."],
        "created_at": 1791322279.1
      }
    ]
  }
}
```

`data` is the `livekit-agents` chat context JSON: `items`, each with
`id`, `type` (`message`, `function_call`, `function_call_output`), the fields of that type, and `created_at` in seconds.
The endpoint reads it as the caller's history and must not store it as its
own; the message's text parts are the turn. The part is omitted when the
caller shares its thread, since the endpoint then reads the caller's
history from it.

The text-only conversation payload has this schema. `items` is required and
ordered; timestamps are Unix seconds. Optional fields may be omitted.

| Item type | Required fields beyond `id`, `type`, `created_at` | Optional fields |
| --- | --- | --- |
| `message` | `role` (`system`, `developer`, `user`, or `assistant`), `content` (array of strings) | `interrupted` (boolean, default false), `extra` (JSON object) |
| `function_call` | `call_id`, `name`, `arguments` (string containing a JSON argument object) | `group_id` (string or null), `extra` (JSON object) |
| `function_call_output` | `call_id`, `output` (string), `is_error` (boolean) | `name` (string), `reply_required` (boolean, default true) |

Only permitted conversation items may be shared. Keep tool calls and their
outputs paired, and exclude the unresolved delegation call. Runtime objects,
private credentials, provider-only payloads, and application context objects are
not part of this payload. A receiver validates the payload before using it as
model history. This contract does not define an audio/image transfer format.

## Additional request fields and attributes

| Field | Location | Type and contract |
| --- | --- | --- |
| `contextId` | `GetTask`, `CancelTask`, `SubscribeToTask` JSON-RPC `params` | Required string thread ID, alongside the standard `id`. This is a LiveKit addition to these requests. |
| `offset` | `SubscribeToTask` JSON-RPC `params` | Optional opaque string stream cursor; resumes after that event. Takes precedence over `Last-Event-ID`. Never construct or increment it. |
| `Last-Event-ID` | subscription HTTP request header | Optional opaque SSE cursor when `offset` is absent. |
| `A2A-Version` | HTTP request header | Standard A2A version selection; LiveKit clients send `1.0`. |
| `X-LiveKit-Thread-Grant` | direct-worker HTTP request header | Alternative bearer JWT carrier for the thread grant described above. |
| `attributes` | LiveKit threads service, not an A2A message/task field | Application-defined string-to-string map for titles, previews, and flags. Read and updated through the thread API; never grants authorization. |

`contextId`, `taskId`, `messageId`, `referenceTaskIds`, `historyLength`, and
pagination fields retain their A2A meanings except where this page explicitly
adds a requirement. `answer` is LiveKit's artifact naming convention, not a new
artifact field. Thread IDs (`AT_…`) and task IDs (`AST_…`) are opaque strings.

Thread `scope`, expiry, attributes, saved application fields, and TextSession
routing state are stored through LiveKit storage APIs; they are not implicit
A2A metadata. Internal chat-item `extra` fields used for delegation correlation
or approval bookkeeping are not peer-facing A2A protocol fields. Applications
do not need to manufacture them. See [threads](../threads.md) and
[context persistence](../context.md).

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

A save whose acknowledgment was lost is not a failure: the task stays
`working` while the runtime checks the database for that run, the request
returns the current task after a bounded wait, and `tasks/get` or a
resubscribe later shows the real outcome.
