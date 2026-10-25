# Channels

A thread endpoint takes messages from every channel the same way, and the
handler never knows which one sent them. One thread, one history, one
state, whatever the channel of the moment. Web chat, API clients, partner
agents and phone calls are built; SMS, WhatsApp and email come with the
platform's ingress and egress and are planned, see
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
[text-endpoints.md](text-endpoints.md). Web chat, API clients and
delegations default to `live` ([delegation.md](delegation.md)); SMS,
WhatsApp and email to `batch`.

| Channel      | Default policy | Debounce |
| ------------ | -------------- | -------- |
| `a2a`        | `live`         | none     |
| `delegation` | `live`         | none     |
| `sms`        | `batch`        | 2 s      |
| `whatsapp`   | `batch`        | 2 s      |
| `email`      | `batch`        | 2 s      |
| any other    | `live`         | none     |

A thread is found by its id only; see [threads.md](threads.md). Finding
a call's thread by the caller's number is planned; see
[planned-channels.md](planned-channels.md).

## Mixing channels

The agent's voice and text turns share one lane; see [lanes.md](lanes.md).

- **A reply goes back on the channel the message came from.** A text is
  answered by text, a call in speech. The handler returns a value and never
  chooses.
- **A text during a call** goes to the call when the lane's text door is
  `live`, or the lane has none: the call holds its lane for the whole call
  and drains the lane's inbox itself. The text is a turn of the call, the
  model reads it at once (on a realtime model, in its own context), and the
  reply, spoken, is also the text's answer on its task: the full reply the
  call generated, also what an interruption cut off. A text that answers the
  question in front of the person goes to the task that asked. When the
  text door has another inbox type, the call takes the lane per voice turn:
  the text is answered by text, on whichever worker takes the lane, and the
  call reloads the lane's history and state on its next turn.
- **An open question belongs to the lane, not the door.** A question asked
  by text is in front of the person when they call, and the spoken answer
  continues the task that asked; a question still open when a call ends
  stays on the lane for a text to answer.
- **A call after a text** is answered from the same history: the call's
  agent sees what was said by text.

## Attachments

A file part on a message reaches the model as part of the user message.
Typed attachments are planned; see [planned-channels.md](planned-channels.md).

See `examples/use-cases/01_appointments_by_phone_and_text.py` and
`04_a_claim_with_a_photo.py`.
