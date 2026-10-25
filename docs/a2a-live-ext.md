# The LiveKit agents extension to A2A

**URI:** `https://livekit.io/a2a/ext/agents/v1`
**Baseline:** A2A 1.0, JSON-RPC 2.0 over HTTP with server-sent events
**Status:** v1, implemented by `livekit-agents` 2.0 (server, client, delegate)

This document is the specification. The SDK is written from it; where
they disagree, the SDK is wrong.

## 1. Scope

The extension lets an A2A client and a LiveKit thread endpoint say what
core A2A has no field for: that a `contextId` is a thread the caller
shares, which channel a message came from, that another agent is asking on
a person's behalf and from which lane, whether a message stops the reply
the agent is producing, that a task was interrupted, merged, or goes on in
the background, that a question waits behind another, and why a task was
canceled or failed. It adds no method, no task state, no binding and no
part type. The behaviour these keys carry is
[delegation.md](delegation.md). Every field is a `metadata` entry under the extension's
URI, or a `DataPart` tagged with one.

An endpoint works without it. A client that never sends the header gets
a plain A2A agent and still receives the server-set keys of section 5,
which it may ignore. A delegating client MUST NOT require a key from the
endpoint it calls: with the extension inactive, the working statuses are
the progress, the artifact is the answer, `input-required` is a question,
and the terminal states end the task.

The key words MUST, MUST NOT, SHOULD and MAY are as in RFC 2119.

## 2. Activation

The endpoint's agent card lists the extension under
`capabilities.extensions`, not required:

```json
{ "uri": "https://livekit.io/a2a/ext/agents/v1" }
```

A client activates it per request with the header
`A2A-Extensions: https://livekit.io/a2a/ext/agents/v1`. The server
echoes the same header on the response when it activated it. A client
MUST send the header only to a card that lists the URI. A client that
cannot read response headers reads the keys of section 5 and ignores the
rest.

With the extension inactive the server MUST NOT send the data part of
section 7. It MUST still send the keys of section 5.

The card always lists the `livekit` bearer security scheme (a LiveKit
project token or a thread grant), marked required only on an endpoint that
is not public.

## 3. Threads

The thread is the `contextId`, carried in the message body and nowhere
else. A message with none starts a new thread; the `Task` that answers it
carries the id. A message never names a thread by anything but its id.

A delegating client that holds a thread MUST send its thread id as the
message's `contextId` when the expert's card lists the extension, and MUST
NOT send it otherwise. An endpoint of the same project that receives a
thread id of another agent works on its own lane of that thread: its own
history and state, beside the caller's. A stock server gets no id and keeps
a context of its own.

## 4. Keys

On the wire a key is `https://livekit.io/a2a/ext/agents/v1/<name>`.
This document writes `livekit/<name>`. Names are camelCase. Values are
JSON. A key not listed here is not part of the extension and MUST be
ignored.

| Key                          | On | Set by | Value |
| ---------------------------- | -- | ------ | ----- |
| `livekit/channel`            | a message | the client | string: `a2a` (the default when absent), or the channel the platform's ingress received it on |
| `livekit/kind`               | a message | a delegating agent | string: `delegation` |
| `livekit/interrupt`          | a message | the client | boolean: whether the message stops the reply the agent is producing; absent means `true`; a delegation sends `false` |
| `livekit/callerLane`         | a delegation in the caller's thread | a delegating agent | string: the caller's lane, whose history holds the person's own words |
| `livekit/heard`              | a message that answers an expert's question | a delegating agent | string: the person's message cut the question off, and this is what they saw or heard of it, often empty; the message text leads with one line saying the same |
| `livekit/kind`               | a `DataPart` | a delegating agent | string: `chat_ctx` |
| `livekit/reason`             | the `CancelTask` request's metadata | the client | string: why |
| `livekit/reason`             | the final `TaskStatusUpdateEvent` of a task a client canceled; the status message of a `failed` task | the server | string: the client's reason, or section 5's |
| `livekit/interrupted`        | the final `TaskStatusUpdateEvent` of a `canceled` task | the server | `true` |
| `livekit/interruptionReason` | the same event | the server | string: `new-message` |
| `livekit/mergedInto`         | the final `TaskStatusUpdateEvent` of a `completed` task | the server | string: a task id |
| `livekit/background`         | a `working` `TaskStatusUpdateEvent`; the task's metadata | the server | `true` while the task's turn has ended and its work goes on in the background; `false` once it left the background |
| `livekit/queued`             | an `input-required` task's metadata and final status | the server | boolean: the question waits, unasked, behind the one in front of the person |
| `livekit/answeredWith`       | the final `TaskStatusUpdateEvent` of a task; the task's metadata | the server | string: the task whose turn gave the one reply that answered this task too |
| `livekit/cancelRequested`    | a `working` task's metadata | the server | `true` between `CancelTask` and the task's end |
| `livekit/verbatim`           | the final `TaskStatusUpdateEvent` of a `completed` or `input-required` task | the server | string: the part of the answer or the question that reaches the person exactly as written; it leads the text |

## 5. Server-set keys, sent either way

These describe what the server did to a task. They ride on `metadata`,
which is core A2A, so a stock client receives them too. The examples are
the `result` of one server-sent event, timestamps omitted.

**`livekit/interrupted` and `livekit/interruptionReason`.** A thread
endpoint runs one turn at a time per agent. When a new message that
interrupts arrives while a reply is produced (the `interrupt` policy, or
`live` with `livekit/interrupt` absent or `true`), the running task ends
`canceled` with these two keys and no status message. Its history keeps
what ran; it has no artifact; the answer is on the new task. A task whose
work goes on in the background is not ended: it stays `working` and its
work answers it. A `canceled` task without `livekit/interrupted` was
canceled by a client.

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

**`livekit/mergedInto`.** Messages that waited together run as one turn
(the `live` and `batch` policies): every one but the last ends `completed`
with no artifact and this key naming the task whose turn answered them
all. A `completed` task with this key was answered; a dashboard counts one
reply. An answer to a question never uses it: the answer continues the
waiting task (section 8).

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

**`livekit/reason`.** On the final status of a task a client canceled, the
reason the client gave. On the status message of a `failed` task, why it
failed:

| Reason                | Meaning |
| --------------------- | ------- |
| `thread-not-found`   | the `contextId` names no thread |
| `worker-lost`         | the worker holding the turn died; the client resends |
| `lease-timeout`       | on an `interrupt` or `queue` endpoint, the thread's lane was not free within 60 seconds |
| `storage-unavailable` | the turn's save could not be written; the task fails under the id it was acknowledged with, and the client resends |

**`livekit/background`.** Under `live`, a tool that reports progress
lets the turn go on: the model replies, and the turn ends while the tool
runs. The task stays `working`; its last status of that turn carries the
reply as its message and this key. The answer comes later on the same
task, when the work ends. A reader SHOULD treat the text streamed before
this status as an acknowledgment, not as the answer. A caller that
continues a question reads this key as "the expert's task released the
floor".

**`livekit/answeredWith`.** Under `live`, work in the background that
ends together, for several tasks, is resumed in one turn, and the model
gives one reply. That reply is the answer of every one of those tasks, each
on its own stream; the earlier tasks carry this key, naming the last one.
Unlike `livekit/mergedInto`, the task had a turn and work of its own: its
answer is the reply, not a pointer to another task's.

**`livekit/queued`.** A lane has one question in front of the person. A
question asked while another is in front waits: its task is
`input-required` with this key `true`, and it is asked when the one in
front is answered or its task releases the floor. The key turns `false`
when the question is asked.

**`livekit/cancelRequested`.** `CancelTask` on a task that runs records the
request on the task; the endpoint stops the task's tools at their next
wait point and ends it `canceled`, with what they did. The key is `true`
in between.

```json
{
  "statusUpdate": {
    "taskId": "AST_3",
    "contextId": "AT_x",
    "status": {
      "state": "TASK_STATE_FAILED",
      "message": {
        "messageId": "msg_9",
        "role": "ROLE_AGENT",
        "parts": [{ "text": "the task failed: worker-lost" }],
        "metadata": { "https://livekit.io/a2a/ext/agents/v1/reason": "worker-lost" }
      }
    }
  }
}
```

## 6. Client-set keys

**`livekit/channel`.** Which channel the message came from. The endpoint
reads it to pick its policy for a message that arrives while a turn runs.
Absent means `a2a`. A message whose `livekit/kind` is `delegation` is on
the `delegation` channel whatever this key says.

**`livekit/kind: delegation`.** On a message: another agent is asking on
its user's behalf, inside a turn of its own. The endpoint SHOULD treat
the message as a request from a peer, not a person: never interrupt a
running reply with it, and answer it as an artifact the caller will
phrase. A delegation that names no `taskId` is a new task, whatever
questions are open. The caller MUST set it on every delegated message and
MUST NOT set it on a person's own message.

**`livekit/interrupt`.** Whether the message stops the reply the agent is
producing. The sender knows; the server does not. Absent means `true`. A
delegation sends `false`. Only the `live` policy reads it; under the other
policies the policy decides.

**`livekit/callerLane`.** On a delegation in the caller's thread: the
caller's lane. An endpoint of the same project MAY read that lane's
history to see the person's own words; the caller writes the person's
message there before it sends. A delegation whose `livekit/callerLane` is
the endpoint's own lane is refused at once (`UnsupportedOperationError`):
it would wait for the lane its caller holds.

**`livekit/reason`.** On `CancelTask`, why. Free text, recorded on the
task and returned on its final status. A delegating caller does not cancel
when it gives up (a barge-in, a closed call): the expert's work goes on,
and only an explicit `CancelTask` stops it.

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "CancelTask",
  "params": {
    "id": "AST_4",
    "contextId": "AT_x",
    "metadata": { "https://livekit.io/a2a/ext/agents/v1/reason": "the person asked to stop" }
  }
}
```

## 7. The `chat_ctx` data part

A delegating caller with no thread to share MAY attach its conversation
so far as one `DataPart` on the message, tagged `livekit/kind: chat_ctx`:

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
`id`, `type` (`message`, `function_call`, `function_call_output`,
`agent_handoff`), the fields of that type, and `created_at` in seconds.
The endpoint reads it as the caller's history and MUST NOT store it as its
own; the message's text parts are the turn. The part is omitted when the
caller shares its thread, since the endpoint then reads the caller's
history from it.

## 8. Tasks and the keys together

One message is one task; the task id is the id of the stream a client
resubscribes to. A message carrying the `taskId` of a task waiting
`input-required` continues that task: it runs under that id, the question
is answered, and no task of its own is made. A message whose `taskId`
names a task that runs is a new task; one whose `taskId` names a terminal
task is refused, unless it is a resend. A message without a `taskId` is a
new task in the same context and leaves open questions open, with one
exception on an endpoint that faces the person: a message that is not a
delegation and names no task goes to the question in front of the person,
when there is one. Terminal tasks never change; a result that lands after
its task ended is dropped. A resend of a `messageId` returns the task it
made, also while it runs: the response is that task's stream from its
first frame, under its id, with its answer.

A task that waits `input-required` keeps its answer stream open: the
continuation appends to it, so `SubscribeToTask` replays the question and
the answer as one stream. A reader of an `input-required` task gets what
the stream holds and ends; it follows live only a task that runs.

`GetTask`, `CancelTask` and `SubscribeToTask` find the thread from
`contextId` in their params, beside the task `id`. A2A's task methods carry
only the task id, so a stock client that sends just the id gets
`TaskNotFoundError`.

The order of events on a stream is: the `Task` (state `submitted`), zero
or more `working` status events each carrying a text message (progress),
zero or more chunks of the artifact named `answer` (the last with
`lastChunk: true`), then the final status event with the keys of section 5
that apply. A task that went on in the background has a `working` status
with `livekit/background` between its acknowledgment and its answer. The
answer is the `answer` artifact whose last chunk arrived last; a whole
answer sent at the end replaces what was streamed under another id. An
`input-required` question is the status message.

## 9. What a stock client sees

Over the same endpoint, with no header: the `Task`, text progress on
`working` statuses, the answer as an artifact, `input-required` with the
question as the status message, the terminal states, and the keys of
section 5, which it may ignore. Nothing else.

What a LiveKit caller sends to a stock server: text and `messageId`,
nothing else. No header, no `contextId` of its own, no `livekit/` key, no
data part.
