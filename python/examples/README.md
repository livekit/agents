# Fast brain, slow brain

A front desk talks to the person, by text and on the phone; its one tool delegates over A2A to a
back office that holds the system and does the work.

| Example | What |
| --- | --- |
| [`frontdesk/`](frontdesk/README.md) | A dental practice's scheduling: two tools and a booking that asks for what it is missing. |
| [`hotel-receptionist/`](hotel-receptionist/README.md) | A hotel's bookings, lookups and cancellations: guided flows, verification, approval on writes. |
| [`airline-support/`](airline-support/README.md) | An airline's customer care: a back office that asks for what is missing, EU261 in code, payment after a reviewed reply. |

## The homepage agent

[`livekit-expert/`](livekit-expert/README.md) is the voice agent on livekit.io's homepage: a
single agent, with no back office or storage, that answers questions about LiveKit products
from a Markdown knowledge base it looks up on demand. Unlike the examples above, it has its own
`pyproject.toml` so its directory deploys on its own; in this repository it is a uv workspace
member and runs in the root `.venv`. Its README covers running, evals and deploying.

## Running them

Follow the [Getting started](../../README.md#getting-started): install from PyPI, set the staging
project's keys, then run `lk agent dev` in an example's directory. Use `lk agent console`
for a terminal conversation or an [A2A client](../../docs/clients.md) for text.

## Storage and conversation state

Each `main.py` supplies `AgentDBStorage()` to `AgentServer`. For local SQLite storage,
replace it with `SQLiteStorage()` from `livekit.agents.storage`; it stores databases under
`~/.livekit/agent-threads/`. Both use the same context and session APIs.

One thread can contain these independent persistence partitions:

| Entry point | `partition_key` | Stored conversation |
| --- | --- | --- |
| Phone | Omitted: `DEFAULT_PARTITION_KEY` | Frontend voice history and session state |
| `frontend` A2A endpoint | `"frontend"` | Frontend text history and pending delegations |
| `backend` A2A endpoint | `"backend"` | Backend text history, pending approvals and business fields |

The framework's `DEFAULT_PARTITION_KEY` constant is `"default"`; the examples leave it
implicit on the phone entrypoint. Voice and text have separate frontend partitions because
their saved session states cannot resume each other. Both delegate to the backend partition
within the same thread. Reusing a thread therefore also reuses its backend business context.
A2A `contextId` selects the thread; a call joins one through its dispatch's `thread_id`.
A new thread starts independent conversations.

Every backend application context declares a `SessionState` field with
`persisted(default_factory=SessionState)` alongside its persisted business fields. Its handler
restores those fields and explicitly binds execution:

```python
app = AppContext(services=cal)
await ctx.store.load_context(app)
binding = ctx.store.bind(app)
turn = await TextSession(agent=backend).run(ctx.message, context=binding)
await ctx.store.save_context(app)
return turn.result
```

Binding keeps the same `app.session` object current and incrementally checkpoints accepted
conversation changes, including input-required and approval waits. Business fields still
require `save_context`. Runtime services such as `cal` are supplied by the application and
are not serialized. `turn.state` and `await ctx.store.load_session()` are detached snapshots.

The frontends have no application fields: both `VoiceSession` and `TextSession` use
`context=ctx.store.bind()` so the store restores and checkpoints their internal state.
`Agent.run()` remains the direct execution API for the airline's classifier, writer and
reviewer; these calls are individual computations within the backend conversation.

The examples' seeded calendars, hotels and airline systems are in-memory fakes. Persisting
conversation state does not make those external business effects durable; use a durable
service with idempotent writes in a real application. See [context persistence](../../docs/context.md)
and [threads](../../docs/threads.md) for checkpoint and ownership details.


## Testing text and voice

Start with the deterministic checks from the repository root. They use scripted models;
no provider calls are needed:

```sh
make test TEST_ARGS="python/tests/test_example_simulation.py python/tests/test_airline_support_example.py python/tests/test_session_binding.py python/tests/test_text_session.py python/tests/test_voice_storage.py python/tests/test_a2a_delegation.py --unit"
```

For a manual conversation, use the same staging credentials as above and run from
`python/examples/frontdesk`. These commands each start their own agent:

```sh
# Typed conversation through the voice frontend, without microphone/speaker I/O.
FRONTDESK_NOW=2026-06-12T09:00 lk agent console --text

# Microphone, speaker output and the realtime voice model.
FRONTDESK_NOW=2026-06-12T09:00 lk agent console --record
```

Ask for an opening, provide an email, reject one booking and approve another. Check that
no appointment is booked before approval and that the saved email is reused.

To test the A2A text handler, start `FRONTDESK_NOW=2026-06-12T09:00 lk agent dev`
and use [AgentClient](../../docs/clients.md#a-client-for-an-endpoint) against the `frontend`
endpoint. Keep the returned `thread_id` across messages and pass the waiting `task_id` when
answering an input-required result. Console text mode does not exercise this handler.

For repeatable model and audio checks against that running worker:

```sh
lk agent simulate text --scenarios scenarios/booking.yaml --agent-name dentist
lk agent simulate audio --scenarios scenarios/booking.yaml --agent-name dentist
```

These simulations dispatch the voice entrypoint in text or audio mode; they do not replace
the A2A text-handler checks. Restart the worker between the two runs to reset the seeded
calendar, since the scenarios reserve specific slots.

Test recovery separately through the actual A2A endpoint or a dispatched call with a
stable `thread_id`: stop the worker after an accepted turn, restart it, and continue the
same thread. Verify history, pending approval and saved application fields. Start a new
thread to verify isolation. Console restarts create fresh conversations and do not prove
durable recovery. The examples' in-memory business services reset on restart, so a saved
booking reference alone does not prove that the booking itself survived.
