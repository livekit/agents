# Channels

A thread endpoint takes messages from every channel the same way, with channel metadata available to the handler. Transport policy selects
delivery and concurrency behavior; application context retains the same
explicit persistence contract across channels. Web chat, API clients, partner
agents and phone calls reach the thread directly. SMS, WhatsApp and email
use the platform's ingress and egress; see
[planned-channels.md](planned-channels.md).

| Channel                      | How it reaches the thread                                          | Which thread                                                      | The reply                               |
| ---------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------- |
| web chat, API, partner agent | an A2A client calls the endpoint directly                           | `contextId` on every message after the first                       | the task's answer, streamed or returned |
| another agent, delegating    | the same, with `livekit/kind: delegation`                           | the caller's thread when the expert shares it, else its own       | the task's answer                       |
| phone call                   | a SIP rule dispatches a job; the call runs on a `LiveSession`       | the thread the dispatch names, else a new one                      | speech                                  |

Every A2A message may carry `livekit/channel` with the channel's name, and
the handler reads it per message as `ctx.messages[i].channel`: the
`livekit/channel` value, `delegation` for a message whose `livekit/kind` is
`delegation`, `a2a` otherwise. The channel picks the endpoint's policy and
debounce for a message that arrives while a turn runs; see
[text-endpoints.md](text-endpoints.md). Delegations default to `queue`, web
chat and API clients to `interrupt`.

Messages address a thread by id; see [threads.md](threads.md). The platform
resolves phone numbers to thread IDs; see [channel routing](planned-channels.md).

## Mixing channels

The agent's voice and text turns share one lane; see [lanes.md](lanes.md).

- **A reply goes back on the channel the message came from.** A text is
  answered by text, a call in speech. The handler returns a value and never
  chooses.
- **A text during a call** is answered by text, on whichever worker takes
  the lane after the call's current voice turn. The call knows about it from
  its next turn on: that turn reloads transport history; application code reloads persisted
  context fields before using them.
- **A call after a text** is answered from the same history: the call's
  agent sees what was said by text.

## Attachments

A file part on a message reaches the model as part of the user message.
See [typed attachments](planned-channels.md) for attachment handling.

See `examples/use-cases/01_appointments_by_phone_and_text.py` and
`04_a_claim_with_a_photo.py`.
