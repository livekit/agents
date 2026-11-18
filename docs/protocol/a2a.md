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
| `capabilities.extensions` | `https://livekit.io/a2a/ext/agents/v1` with `required: false`, always listed by a LiveKit endpoint and optional for others: a capability hint that the endpoint reads LiveKit fields; it grants nothing |

The extension entry may carry `params` declaring the tool a caller's model
sees for the endpoint; `RemoteAgent(url).as_tool()` reads them on first use, unless
`name=` or `description=` given in code override them:

| Param | Type and contract |
| --- | --- |
| `toolName` | string: the tool's name; absent, the card's `name` |
| `inputSchema` | JSON Schema object for the tool's arguments; absent, `{"request": string}` (required), sent as the message's text. With it, the arguments go as the message's first part, a `DataPart` |

A LiveKit endpoint lists the entry without `params`. Spelling and case are
significant; other `params` keys are ignored.

Every endpoint requires a token: a project token or a thread grant.

### Selecting work through a remote tool

`RemoteAgent(...).as_tool()` adds an optional, nullable `task_id` argument to the tool's
schema. For example, `{"request": "Why do you need the date?", "task_id": "booking-42"}`
continues that remote task for clarification. Saved conversation state supplies its
`contextId`; the model does not supply an input-occurrence ID. A task can also receive a
correction or an answer. Selecting it does not by itself answer its current question.

The task must be known on that configured remote in the current conversation and must
still be nonterminal. Invalid selection returns tool feedback before dispatch, without
creating new work. Omitted or null `task_id` leaves selection to the conversation owner.
Bind `Agent.run`, `TextSession`, or `VoiceSession` to the conversation's context store to
retain that state across executions; saved addresses do not restore credentials.

The framework removes `task_id` before sending the remote's own business input, then sets
the A2A message's `taskId` and saved `contextId`. A custom card schema keeps its other
arguments as a data part. Declaring a business argument named `task_id` is a setup error.
This tool argument is a caller API, not an extra A2A wire field.

With a configured routing model, the owner separately recognizes human responses and routes
proposed tool calls. The latter can continue working tasks, clarify a pending question, or
start independent work. A failed or ambiguous decision produces tool feedback. Disabling
automatic routing makes an explicit `task_id` continue known work and an omitted ID start new
work. See [response routing](../text-session.md#responses-to-requests-for-input).

A live task has one native observer. Further calls admitted to it wait for message admission;
progress, questions and the final result stay with the originating call. The observer closes
its old stream before opening another exchange, and a peer merging concurrent requests cannot
create a second result owner. Request and message identities survive recovery of a prepared
operation. Ending local observation leaves remote business work running.

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
| `SubscribeToTask`, `tasks/resubscribe`         | `UnsupportedOperationError` |

`GetTask` and `CancelTask` find the thread from
`contextId` in their params, beside `id`. A2A's task methods carry only the
task id, so a stock client that sends just the id gets `TaskNotFoundError`.
Responses stream live on the original request. Events are not retained and have no
resume cursor. After a disconnect, use `GetTask` to retrieve the current task and
its saved answer; it does not replay progress or answer chunks.

A stream is, in order:

1. the `Task` in state `submitted`, sent once the task's row is durable;
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

The next message reaches the conversation owner on whichever worker takes the message.
Task correlation alone does not prove that this message answers the question. The owner
recognizes responses to presented input requests, keeps side questions and new topics with
the current agent, and retains unresolved requests. The tool's recorded outcome stays intact;
transport admission does not replace it with arbitrary human text or invoke an answer tool.
See [response routing](../text-session.md#responses-to-requests-for-input).

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
| `livekit/interrupted` | `Task.metadata` and final canceled `TaskStatusUpdateEvent.metadata` | server | Boolean `true` when the handler reports an interrupted result. Absent on ordinary cancellation. |
| `livekit/mergedInto` | `Task.metadata` and final completed `TaskStatusUpdateEvent.metadata` | server | String task ID of the turn that answered this message too. |
| `livekit/progress` | Working `status.message.metadata`, `Task.metadata` or `TaskStatusUpdateEvent.metadata` | server | Boolean `true` explicitly offers the text for conversational progress. `ctx.update()` sets it. |
| `livekit/verbatim` | `Task.metadata` and final completed or input-required `TaskStatusUpdateEvent.metadata` | server | String prefix of the answer/question to deliver exactly as written; the frontend may phrase the remaining text. |

Server-owned fields cannot be set by a caller to alter task outcomes. An absent
field makes no claim: in particular, `completed` with `mergedInto` means the
answer belongs to the referenced task, rather than an empty successful answer.
For interrupted turns, earlier artifact chunks are partial. A newer request does not
interrupt an active turn; it follows the endpoint’s queue or batch policy.

Working text alone does not request an announcement: a peer can put a draft answer or internal
status there. A remote tool offers only explicitly flagged working text for presentation.
`TaskUpdate.progress` accepts a boolean `true` on the message or its enclosing task/update;
other values do not set it.
Third-party peers can use the namespaced flag without advertising a LiveKit extension.

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

The [authentication contract](../auth.md) owns credential authority, permission
flags, node admission and worker verification. Requests through a LiveKit node
need `agentEndpoint.call` for reach and the appropriate `agentThread` permission
for thread data. `Credentials.thread_grant_token()` includes both grants;
[client examples](../clients.md#tokens) show how to mint them.

These permissions are JWT claims, not A2A metadata. For example:

```json
{
  "agentEndpoint": {"call": true},
  "agentThread": {"scope": "user:42", "create": true, "write": true, "read": true}
}
```

`agentThread.scope` is a string, `threadIds` an array of string IDs, and `all`
a boolean. Permission flags are booleans and default to false; `all` does not
enable missing flags. A subject is required: `scope`, `threadIds` or `all`.
A first message with no `contextId` gets its scope from the authorized claim,
never from the request. See [thread grants](../auth.md#the-thread-grant) for
which subject forms permit creation and listing.

Send the token as `Authorization: Bearer <token>`. An authorization refusal is
an HTTP response outside JSON-RPC; [endpoint authentication](../auth.md#endpoints)
defines its status and error shape. Browser SSE delivery uses a JSON-RPC POST
through `fetch`, with the bearer header.

## Delegation and conversation data

A caller shares its thread with an endpoint only under explicit authority: a
same-project grant token the endpoint verifies as covering that thread, or trust
configured on the caller and verified locally. The destination uses its
own partition and application context. A metadata key, an agent card or its
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
| `contextId` | `GetTask`, `CancelTask`, `ListTasks`, `SubscribeToTask` JSON-RPC `params` | Required string thread ID, alongside the standard `id`. This is a LiveKit addition to these requests: a stock client that omits it gets `TaskNotFoundError`. |
| `A2A-Version` | HTTP request header | Standard A2A version selection; LiveKit clients send `1.0`. |
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
| `InvalidParamsError`        | a message with no `messageId`, params that are not an object; a body over the transport's payload budget |
| `TaskNotFoundError`         | an unknown task id; a `taskId` not in the message's thread; `tasks/*` with no `contextId` |
| `TaskNotCancelableError`    | `tasks/cancel` on a terminal task |
| `UnsupportedOperationError` | a message to a terminal task; a message to a thread whose turn runs, on an endpoint whose policy is `reject`; `SubscribeToTask` (use `GetTask`); `GetExtendedAgentCard` |
| `PushNotificationNotSupportedError` | any push notification config method |
| `VersionNotSupportedError`  | an `A2A-Version` header other than `1.0` |
| `InternalError`             | the endpoint's request budget is exhausted |
| `MethodNotFoundError`, `InvalidRequestError`, `JSONParseError` | a malformed request; a batch request |

JSON-RPC errors are for malformed requests. Everything else ends the task
`failed`, with `livekit/reason` on its status message:

| Reason                | When |
| --------------------- | ---- |
| `thread-not-found`   | a `contextId` the threads service does not know |
| `worker-lost`         | the worker holding the turn died; resend the message |
| `lease-timeout`       | on a `queue` endpoint, the thread's partition was not free within 60 seconds |
| `storage-unavailable` | the turn's save could not be written; resend the message |

A failure no task row can carry (a message on a `reject` endpoint whose row
was never written) is answered with the database's error instead of a
failed task; resending the message runs it.

A save whose acknowledgment was lost is not a failure: the task stays
`working` while the runtime checks the database for that run, the request
returns the current task after a bounded wait, and `tasks/get` later shows the real outcome.
