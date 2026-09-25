# Delegation over A2A

A voice agent that talks, and an expert that thinks. Two processes on one machine, speaking
[A2A](https://a2a-protocol.org) with the LiveKit agent session extension.

Northwind Air's support line splits in half. `voice.py` is a realtime model that owns the
conversation and knows nothing about fares, seats or rules. `expert.py` is the fare desk: a
text model with twelve tools, a timetable, live seat inventory and a page of fare policy,
which never speaks to the caller. It works out what is true and hands back facts; the phone
agent says them in its own words.

## Running it

Two terminals. The desk first, because the phone agent connects to it:

```bash
python expert.py dev      # serves http://localhost:8321/fare-desk
python voice.py console   # call in
```

Ask something real. The seeded airline has an interesting case waiting:

| caller              | what makes them worth asking about                                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `dana@example.com`  | Gold, and her Tokyo flight tomorrow is delayed 245 minutes — our fault, so the change fee is waived and her seat moves for nothing |
| `ortiz@example.com` | on a BASIC fare, which cannot be changed or refunded at all                                                                        |
| `raman@example.com` | holds 120 USD of travel credit, which a new booking spends                                                                         |

_"My flight to Tokyo tomorrow is delayed — what else can you put me on?"_ makes the desk
check the weather at both ends, find the evening flight, and work out that the delay waives
both the fee and the fare difference.

## The two lanes

Each process logs its own half, so the hand-off reads across two terminals:

```
voice.py     ▶ delegated: their Tokyo flight tomorrow is delayed, find something else
expert.py         → check_weather({"airport": "HND", ...})
expert.py         ← done: {'conditions': 'typhoon warning', ...}
expert.py         → rebook({"booking_ref": "NW7Q2K", ...})
expert.py         … rebook: holding a seat on NW812
voice.py     … relayed: holding a seat on NW812
voice.py     ◀ answered: moved to NW812, the delay waived the fee
```

## What to look at

- **`@server.a2a_session(endpoint="fare-desk")`** serves an `AgentSession` over A2A on the
  agent server's own HTTP app. The handler runs once per context, builds its session and
  hands it over; every later request on that `contextId` is a turn of the same session, so
  the desk remembers who it is talking to.
- **`ctx.update()`** in `rebook` and `book_flight` reports while the seat is being held and
  releases the turn, so the caller hears progress instead of silence. The report is relayed
  as the tool wrote it, not handed to a model to restate.
- **`ctx.request`** in `end_of_call` is how the desk tells whether a caller is waiting on an
  answer. When one is, it sets a directive that rides back with the answer; in an ordinary
  session there is nobody to advise.
- **`collect_email` lives on the voice side**, because spelling an address back is a
  back-and-forth and the desk is not on the phone. The desk asks for one in its answer.
- **`delegate=A2ADelegate(url)`** is the whole of the voice side's delegation code. The
  session closes the delegate when the call ends, which is what tells the desk to drop the
  context.

## Persistence

Persistence lets a session be loaded again after it closed. A text session cannot stay alive for days between messages, and a voice session ends with the call. A session that ends with a defined error (an LLM, STT or TTS failure) still closes gracefully, so it still saves. Persistence is not designed to survive a server crash: nothing written since the last save is recovered, and no mechanism in the framework exists for that case.

A session given `persist=` is saved once, when it closes: the items its history and its agents' contexts gained, changed or lost since the last save, then its current agent, the agents that current agent returns to, its userdata (the mock airline included) and any durable tool's frame, in one batch. One conversation is one agent-db database, so its id is the database id. The front session, the phone agent's, takes the conversation id as its own, so every call on the conversation resumes it; each desk context it talked to is a row in the same database, under the caller's, with the context id as its id.

The store belongs to the agent server, which hands it to each job as `ctx.store` and to each desk context as `ctx.persisted`:

```python
server = AgentServer(store=store.AgentDB())   # LIVEKIT_AGENTDB_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET

@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await session.start(agent=Receptionist(), room=ctx.room, persist=ctx.store.session(conversation_id))

@server.a2a_session(endpoint="fare-desk", description="Answers fare questions.")
async def fare_desk(ctx: A2ASessionContext) -> None:
    await session.start(agent=FareDesk(), persist=ctx.persisted)   # None when the caller named no conversation
```

Start agent-db locally, from `agents-private/agent-db`, and leave it running:

```bash
mage build && mage devLocal   # management :7780, data plane ws://localhost:7781/db
```

Point the processes at it. devLocal serves its data plane on a port of its own, and takes its own key, which the example passes when the URL is on localhost:

```bash
export LIVEKIT_AGENTDB_URL=http://localhost:7780
export LIVEKIT_AGENTDB_WS_URL=ws://localhost:7781/db
```

### The desk drill, with no microphone

`chat.py` is a text client over A2A. It mints a conversation and a context, prints both, and sends each line as a person's turn. Ending its input sends the goodbye (`lk/kind = close`), which closes the desk's context, and so saves it.

```bash
python expert.py dev   # terminal 1
python chat.py         # terminal 2: prints conversation DB_... and context chat-...
```

1. Ask two things that build on each other: _"Hi, I'm dana@example.com. What's the status of my flight to Tokyo tomorrow?"_, then _"What other flights could you put me on that day, and what would the change cost me?"_ The desk quotes the change and keeps the quote on the booking.
2. End the input with Ctrl-D. The desk closes the context and saves it.
3. Open it again from a fresh client: `python chat.py --conversation DB_... --context chat-...`, and ask a follow-up that only makes sense with what came before: _"OK, go ahead and move me onto that evening flight you just quoted."_ The desk logs `↺ rehydrated chat-...: N messages back` and rebooks from the quote it made in the first session.

`--delegate` sends lines as instructions, the way the phone agent asks.

### The phone agent drill: a durable tool

`collect_email` on the phone agent is a durable tool: it awaits `EffectCall(GetEmailTask(...))`, so a session closed while the email task runs saves the tool's frame, and the next session resumes the task where the caller left off. `voice_drill.py` runs the phone agent over text with the same delegate and persistence; ending its input closes the session.

```bash
python expert.py dev        # terminal 1
python voice_drill.py       # terminal 2: prints conversation DB_...
```

1. Ask for something that needs an address: _"Hi, my flight to Tokyo tomorrow is delayed. Can you move me onto the evening flight?"_ The desk asks for the caller's email, and the phone agent hands over to the email task, which asks for it.
2. End the input with Ctrl-D while the task is waiting. The session saves with the email task current.
3. Start it again on the same conversation: `python voice_drill.py --conversation DB_...`. It logs `the AgentTask was awaited from a durable tool, so it resumes`, and the email task is the current agent again, without asking twice.
4. Give the address: _"It's dana@example.com"_. The task hands back to the restored `collect_email`, which records the caller and returns, and the phone agent delegates the change.

### Reading the rows

`agentdb-console` in `agents-private/agent-db` reads the database directly; the tables are the contract a dashboard reads:

```bash
alias adb='./bin/agentdb-console -database DB_...'
adb -q "SELECT session_id, parent_session_id, endpoint, current_agent_id, closed_at FROM sessions"
adb -q "SELECT json_extract(item,'$.role') AS role, substr(json_extract(item,'$.content[0]'),1,80) AS text
        FROM chat_items WHERE owner = 'session' AND json_extract(item,'$.type') = 'message' ORDER BY created_at"
adb -q "SELECT agent_id, parent_agent_id, length(durable_state) AS frame_bytes FROM agents"
adb -q "SELECT json_extract(item,'$.call_id') AS call_id, json_extract(item,'$.extra.\"lk.task_id\"') AS task_id
        FROM chat_items WHERE owner = 'session' AND json_extract(item,'$.name') = 'lk_agents_delegate'
        AND json_extract(item,'$.type') = 'function_call_output'"
```

A desk session names its caller in `parent_session_id`. `lk.task_id` is the A2A task on both sides: the output of each delegate call names the desk task that answered it, and each call and reply of the desk names the task that produced it, so a dashboard joins the two sides through `chat_items`. `durable_state` is pickled Python, the one column only this framework reads.

### The voice half

`voice.py` persists too once it is given a conversation, into the conversation's front session:

```bash
CONVERSATION=DB_... python voice.py console
```

Hang up and run it again on the same `CONVERSATION`: the call resumes with what was said before, and its delegations reach the same desk context, which the desk loads again on the first one. A durable tool resumes only on a pipeline model; `voice.py` runs a realtime one.

## Talking to the desk without a voice agent

It is plain A2A, so anything that speaks it can drive the desk:

```bash
curl localhost:8321/fare-desk/.well-known/agent-card.json

curl -X POST localhost:8321/fare-desk/v1/message:stream -N \
     -H 'content-type: application/json' -H 'A2A-Version: 1.0' \
     -d '{"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text":
          "what flights are there from SFO to Tokyo next Monday?"}]}}'
```

`A2A-Version: 1.0` is required by the binding; both SDK clients set it for you.

A client that does not know the extension sees text and task states and works fine. One that
does also gets the typed chat items, verbatim text and directives.
