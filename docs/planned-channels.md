# Planned: channels

Not built. This extends [channels.md](channels.md) with SMS, WhatsApp and
email, the dispatch handler, and typed attachments. Everything below reaches
the agent through the same thread endpoint and the same policies as the
built channels.

## Ingress and egress

The platform's ingress turns a text, a WhatsApp message or an email into an
A2A `message/send` on the right thread, and its egress turns the answer
back into a reply on that channel. Email is an experiment.

| Channel             | How it reaches the thread                                                                 | The key                                                                              | The reply                                          |
| ------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------------------- |
| SMS                 | the platform receives the carrier webhook and posts `message/send`                         | `phone:+E164`, derived from the sender                                               | the answer artifact, sent as a text                |
| WhatsApp            | the same                                                                                   | the same `phone:` key, so a text and a WhatsApp from one number are one conversation | a WhatsApp message                                 |
| email, experimental | the platform receives the inbound email and posts the body, subject and attachments        | `email:<address>`, lower-cased                                                       | a reply in the same thread, rendered from markdown |

The key never travels on the A2A wire and is never shown by the public
API. The ingress resolves it with the threads service's internal
`ResolveThread`, get-or-create, and posts the message
with the thread's id as `contextId` and the channel as `livekit/channel`
(`sms`, `whatsapp`, `email`). The endpoint's default policy for these
channels is `batch`: a burst of texts is answered together in one turn, as
one reply, after a two second debounce.

SMS, WhatsApp and calls share the `phone:` key: the same number is one
person and one thread. An email address is its own key, and an email
thread and a phone thread stay separate. A thread has at most one key,
set when the platform creates it and never changed.

## Calls by number

A SIP call's dispatch names the thread the platform resolved by the
caller's `phone:` key, so a call continues the texts and calls before it.
Built today, a call continues only the thread its dispatch names in its
metadata, else a new one; see [voice.md](voice.md).

## Who is texting

An unknown number gets a fresh thread. Routing a number to an existing
thread, for example by a CRM lookup, belongs to the platform's Universal
Dispatch, not to this framework.

## A text during a call, heard live

Built today, a text during a call is answered by text and the call learns
of it on its next turn. When the platform routes a thread to one worker,
the text lands on the call's worker and `LiveSession.on_inbox_message`
decides what happens: spoken as a user turn of the call by default,
answered by text, or left for after the call.

## A call during a waiting text

A question the agent asked by text is still pending when the person calls.
The call's agent sees it in the history and can take the answer by voice;
the pending text question completes with the call's turn.

## Email

Email adds three things to the shape above:

- The thread's root `Message-ID` rides as `livekit/thread` on the message.
  The thread is per correspondent, so one address is one conversation
  across threads; an ingress that wants a thread per thread can key on it
  instead.
- Quoted history is stripped before the message is posted; the subject and
  the attachments are kept.
- Threading routes answers to tasks by itself: a question a task sends by
  email carries a `Message-ID`, and the reply's `In-Reply-To` is enough for
  the platform to post it with that task's id. The main agent never sees it.

A reply the agent writes is markdown, rendered to HTML on the way out.
Attachments out and a subject on a new thread are not part of the design
yet.

## Attachments

An image or a document on a message becomes typed chat content the model
sees natively, with a reference the application resolves; the bytes are not
stored with the thread. A tool takes it as a typed argument:

```python
class Photo(BaseModel):
    ref: str
    name: str

@support.tool
async def start_return(ctx: RunContext[Deps, Return], order_id: str, reason: str, photos: list[Photo]) -> str: ...
```

Attachments out, a label or a statement as a file, are not part of the
design yet.

## Messages the agent starts

A reply to an incoming message is the handler's return value, on the
channel the message came from. A message the agent starts on its own, a
link sent mid-call, a reminder, a background task's question, is an open
design point, decided with channel egress; no verb is designed for it. Only
the live call speaks, with `ctx.live_session.say()`.

See `examples/use-cases/planned-09_speed_to_lead.py`,
`planned-12_many_chats.py` and `planned-14_returns_by_email.py`.
