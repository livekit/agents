# The LiveKit LiveKit agents extension to A2A

**URI:** `https://livekit.io/a2a/ext/agents/v1`
**Baseline:** A2A 1.0, JSON-RPC 2.0 over HTTP with server-sent events
**Status:** v1, implemented by `livekit-agents` 2.0 (server, client, delegate)

This document is the specification. The SDK is written from it; where
they disagree, the SDK is wrong.

## 1. Scope

The extension lets an A2A client and a LiveKit thread endpoint say what
core A2A has no field for: that a `contextId` is a thread the caller
shares, which channel a message came from, that another agent is asking on
a person's behalf, that a task was ended or merged by the server, and why a
task was canceled or failed. It adds no method, no task state, no binding
and no part type. Every field is a `metadata` entry under the extension's
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
| `livekit/callerLane`         | a delegation in the caller's thread | a delegating agent | string: the caller's lane, whose history holds the person's own words; the expert's handler may read it |
| `livekit/kind`               | a `DataPart` | a delegating agent | string: `chat_ctx` |
| `livekit/reason`             | the `CancelTask` request's metadata | the client | string: why |
| `livekit/reason`             | the final `TaskStatusUpdateEvent` of a task a client canceled; the status message of a `failed` task | the server | string: the client's reason, or section 5's |
| `livekit/interrupted`        | the final `TaskStatusUpdateEvent` of a `canceled` task | the server | `true` |
| `livekit/interruptionReason` | the same event | the server | string: `new-message` |
| `livekit/mergedInto`         | the final `TaskStatusUpdateEvent` of a `completed` task | the server | string: a task id |
| `livekit/verbatim`           | a text `Part` of a `working` status message, of the answer artifact, or of an `input-required` status message | the server | `true`: this part reaches the person exactly as written; the agent facing them words every other part |

### Exact text

What happens during a task is a status message, and the answer is the
artifact. Exact text uses the same split, and the flag on the part is its
only encoding. A `working` status whose text part is flagged is said to the
person as written, at once; one without the flag is progress for the agent
facing the person to word. The answer artifact is flagged only when the
task ends on exact text, one flagged part for each exact result the task
ended on, in order; a reader says each part as written. A question is
flagged only when it is asked as written.

A tool says a notice, then the model answers in its own words:

```json
{"statusUpdate": {"taskId": "AST_1", "contextId": "AT_x", "status": {"state": "TASK_STATE_WORKING", "message": {"role": "ROLE_AGENT", "parts": [{"text": "Checking the schedule."}]}}}}
{"statusUpdate": {"taskId": "AST_1", "contextId": "AT_x", "status": {"state": "TASK_STATE_WORKING", "message": {"role": "ROLE_AGENT", "parts": [{"text": "Calls are recorded for quality.", "metadata": {"https://livekit.io/a2a/ext/agents/v1/verbatim": true}}]}}}}
{"artifactUpdate": {"taskId": "AST_1", "contextId": "AT_x", "artifact": {"artifactId": "art_1", "name": "answer", "parts": [{"text": "Tuesday at 10 is free."}]}, "lastChunk": true}}
{"statusUpdate": {"taskId": "AST_1", "contextId": "AT_x", "status": {"state": "TASK_STATE_COMPLETED"}}}
```

The task ends on exact text, a confirmation code read as written:

```json
{"artifactUpdate": {"taskId": "AST_2", "contextId": "AT_x", "artifact": {"artifactId": "art_2", "name": "answer", "parts": [{"text": "Your code is HTL-7A3K.", "metadata": {"https://livekit.io/a2a/ext/agents/v1/verbatim": true}}]}, "lastChunk": true}}
{"statusUpdate": {"taskId": "AST_2", "contextId": "AT_x", "status": {"state": "TASK_STATE_COMPLETED"}}}
```

The task ends on two exact results of one step, each its own part, never joined:

```json
{"artifactUpdate": {"taskId": "AST_4", "contextId": "AT_x", "artifact": {"artifactId": "art_4", "name": "answer", "parts": [{"text": "Your code is HTL-7A3K.", "metadata": {"https://livekit.io/a2a/ext/agents/v1/verbatim": true}}, {"text": "Calls are recorded for quality.", "metadata": {"https://livekit.io/a2a/ext/agents/v1/verbatim": true}}]}, "lastChunk": true}}
```

A question asked as written:

```json
{"statusUpdate": {"taskId": "AST_3", "contextId": "AT_x", "status": {"state": "TASK_STATE_INPUT_REQUIRED", "message": {"role": "ROLE_AGENT", "parts": [{"text": "Which payer, and the member id?", "metadata": {"https://livekit.io/a2a/ext/agents/v1/verbatim": true}}]}}}}
```

A stock client reads the same text and ignores the flag.

## 5. Server-set keys, sent either way

These describe what the server did to a task. They ride on `metadata`,
which is core A2A, so a stock client receives them too. The examples are
the `result` of one server-sent event, timestamps omitted.

**`livekit/interrupted` and `livekit/interruptionReason`.** A thread
endpoint runs one turn at a time per agent. When a new message arrives
for a thread whose turn is running and the endpoint's policy is to
interrupt, the running task ends `canceled` with these two keys and no
status message. Its history keeps what ran; it has no artifact; the answer
is on the new task. A `canceled` task without `livekit/interrupted` was
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

**`livekit/mergedInto`.** When the endpoint's policy batches messages
that arrive while a turn runs, every batched task but the last ends
`completed` with no artifact and this key naming the task whose turn
answered them all. The same key marks a task that was waiting
`input-required` when a later message, sent without its `taskId`, answered
it: the later message's task is the one named. A `completed` task with
this key was answered; a dashboard counts one reply.

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
| `storage-unavailable` | the turn's save could not be written; the client resends |

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
the message as a request from a peer, not a person: queue it behind a
running turn rather than interrupt, and answer it as an artifact the
caller will phrase. The caller MUST set it on every delegated message
and MUST NOT set it on a person's own message.

**`livekit/reason`.** On `CancelTask`, why. Free text, recorded on the
task and returned on its final status; `caller-canceled` when a delegating
caller gave up, as when the person spoke over a call.

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "CancelTask",
  "params": {
    "id": "AST_4",
    "contextId": "AT_x",
    "metadata": { "https://livekit.io/a2a/ext/agents/v1/reason": "caller-canceled" }
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
resubscribes to. A message carrying a `taskId` continues that task while it
is not terminal; a message without one is a new task in the same context,
and closes a waiting task with `livekit/mergedInto`. Terminal tasks never
change. A resend of a `messageId` returns the task it made.

`GetTask`, `CancelTask` and `SubscribeToTask` find the thread from
`contextId` in their params, beside the task `id`. A2A's task methods carry
only the task id, so a stock client that sends just the id gets
`TaskNotFoundError`.

The order of events on a stream is: the `Task` (state `submitted`), zero
or more `working` status events each carrying a text message (progress, or
exact text when its part is flagged),
zero or more chunks of the artifact named `answer` (the last with
`lastChunk: true`), then the final status event with the keys of section 5
that apply.

## 9. What a stock client sees

Over the same endpoint, with no header: the `Task`, text progress on
`working` statuses, the answer as an artifact, `input-required` with the
question as the status message, the terminal states, and the keys of
section 5, which it may ignore. Nothing else.

What a LiveKit caller sends to a stock server: text and `messageId`,
nothing else. No header, no `contextId` of its own, no `livekit/` key, no
data part.
