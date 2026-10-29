# Channels

A thread endpoint receives messages with channel metadata. Transport policy
selects delivery and concurrency behavior; application context keeps the same
explicit persistence contract across channels. The platform adapts SMS,
WhatsApp, and email into endpoint requests and delivers their replies.

| Channel                      | How it reaches the thread                                          | Which thread                                                      | The reply                               |
| ---------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------- |
| web chat, API, partner agent | an A2A client calls the endpoint directly                           | `contextId` on every message after the first                       | the task's answer, streamed or returned |
| another agent, delegating    | the same, with `livekit/kind: delegation`                           | the caller's thread under [explicit authority](protocol/a2a.md#delegation-and-conversation-data), else its own | the task's answer                       |
| phone call                   | a SIP rule dispatches a job; the call runs on a `VoiceSession`       | the thread the dispatch names, else a new one                      | speech                                  |

Every A2A message may carry `livekit/channel` with the channel's name, and
the handler reads the resolved value as `ctx.messages[i].channel`.
`livekit/kind: delegation` selects `delegation` even with an explicit channel;
otherwise it uses `livekit/channel`, or `a2a` when absent. The channel picks
the endpoint's policy and debounce for a message that arrives while a turn
runs; see
[server.md](server.md). Delegations default to `queue`, web
chat and API clients to `interrupt`.

Messages address a thread by ID; see [threads](threads.md). A routing key
selects that ID before the message reaches the endpoint.

## Channel routing

### Ingress and egress

The platform's ingress turns a text, a WhatsApp message or an email into an
A2A `message/send` on the right thread, and its egress turns the answer
back into a reply on that channel.

| Channel             | How it reaches the thread                                                                 | The key                                                                              | The reply                                          |
| ------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| SMS                 | the platform receives the carrier webhook and posts `message/send`                         | `phone:+E164`, derived from the sender                                               | the answer artifact, sent as a text                |
| WhatsApp            | the same                                                                                   | the same `phone:` key, so a text and a WhatsApp from one number are one conversation | a WhatsApp message                                 |
| email | the platform receives the inbound email and posts the body, subject and attachments        | `email:<address>`, lower-cased                                                       | a reply in the same thread, rendered from markdown |

The key never travels on the A2A wire and is never shown by the public
API. The ingress resolves it with the threads service's internal
`ResolveThread`, get-or-create, and posts the message
with the thread's id as `contextId` and the channel as `livekit/channel`
(`sms`, `whatsapp`, `email`). The endpoint's default policy for these
channels is `batch`: a burst of texts is answered together in one turn, as
one reply, after a two second debounce.

Within a configured routing namespace, SMS, WhatsApp and calls can share the
`phone:` key to continue one conversation. A routing key identifies that
conversation; it does not establish the person's authenticated identity.
An email address is its own key, and email and phone conversations stay separate. A thread has at most one key,
set when the platform creates it and never changed.

### Calls by number

A SIP call's dispatch names the thread the platform resolved by the
caller's `phone:` key, so a call continues the texts and calls before it.
See [voice-session.md](voice-session.md) for dispatch metadata.

### Routing an unknown sender

An unknown number gets a fresh thread. Routing a number to an existing
thread, for example by a CRM lookup, belongs to the platform's Universal
Dispatch, not to this framework.

## Shared conversation data

A voice door and text endpoint configured for the same lane share transport
history. Application code explicitly reloads marked context fields to observe
changes made by another activation. Saved private `TextSessionState` histories
remain separate from that transport view; lane sharing does not merge them.

A reply follows the channel of its incoming request. A text received during a
voice turn waits under the endpoint's lane policy and is answered by text;
its arrival does not silently take over the live call. Subsequent voice turns
can observe newly committed transport history and explicitly loaded context.

## Continuing between text and voice

Incoming texts use their configured endpoint and reply on their original
channel by default. Sharing a thread database does not inject a text into
an active voice session or consume a pending voice-task answer.

An application can explicitly route an incoming message into a live call.
That integration must identify the active frontend conversation, serialize
its input with the voice floor, and select a single reply destination. It
must not deliver one human answer to both a VoiceTask and a backend text
request. The frontend expert-event queue carries expert updates; it is not
this channel ingress mechanism.

A question asked through `TextSession` retains its session/frame and pending
call references. Continuing it by voice requires an explicit authorized
continuation of that same interaction through the backend endpoint. Seeing a
question in shared transport history does not by itself restore its private
frame or complete its waiting task. See [session state](text-session.md#loading-and-saving-session-state)
and [frontend expert events](voice-session.md#conversation-owned-inbox).

## Email

Email adds three things to the shape above:

- The ingress retains the root email `Message-ID` for reply threading.
  Its configured routing key selects the conversation: per correspondent
  by default, or per email thread when the application chooses that policy.
  These delivery references do not introduce an unregistered A2A extension key.
- Quoted history is stripped before the message is posted; the subject and
  the attachments are kept.
- Threading routes answers to tasks by itself: a question a task sends by
  email carries a `Message-ID`, and the reply's `In-Reply-To` is enough for
  the platform to post it with that task's ID. The agent need not interpret
  those headers to route the answer.

A reply the agent writes is markdown, rendered to HTML on the way out.
This reply path preserves the inbound email's addressing and threading.
It does not originate a new subject or send outbound attachments.

## Attachments

An image or a document arrives as an A2A file part whose `url` is an
application reference; the bytes are not stored with the thread, and a part
with inline bytes is refused. The model reads the attachment as a text content
item in the message, at the part's position, holding its typed record:

```json
{"type": "attachment", "ref": "https://files.example/r/7f3a", "name": "box.jpg", "media_type": "image/jpeg"}
```

`name` and `media_type` are present when the part carries them. These are the
fields of the typed attachment content a later release delivers natively, in
the same position, without a protocol change for clients. A tool takes the
reference as a typed argument:

```python
class Photo(BaseModel):
    ref: str
    name: str

@support.tool
async def start_return(ctx: RunContext[ReturnContext], order_id: str, reason: str, photos: list[Photo]) -> str: ...
```

The attachment reference is application-defined data, not a framework download
or authorization grant. Resolve it through the application's attachment service
before performing an action that requires the uploaded content.

## Delivery boundary

A reply follows an incoming message's channel and task correlation. Proactive
messages, scheduled reminders, and cross-channel delivery require separate
application egress; channel ingress does not add a scheduler or a durable task
runtime. A live frontend speaks through `ctx.live_session.say()` while its
`VoiceSession` remains active.
