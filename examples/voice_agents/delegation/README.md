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
  agent server's own HTTP app. The handler runs once per conversation, builds its session and
  hands it over; every later request on that `contextId` is a turn of the same session, so
  the desk remembers who it is talking to.
- **`ctx.update()`** in `rebook` and `book_flight` reports while the seat is being held and
  releases the turn, so the caller hears progress instead of silence. The report is relayed
  as the tool wrote it, not handed to a model to restate.
- **`ctx.request`** in `end_of_call` is how the desk tells whether a caller is waiting on an
  answer. When one is, it sets a directive that rides back with the answer; in an ordinary
  session there is nobody to advise.
- **`collect_email` lives on the voice side**, because spelling an address back is a
  conversation and the desk is not on the phone. The desk asks for one in its answer.
- **`delegate=A2ADelegate(url)`** is the whole of the voice side's delegation code. The
  session closes the delegate when the call ends, which is what tells the desk to drop the
  conversation.

## Persistence

With agent-db configured, every conversation persists as it goes: the desk writes each item as it lands and checkpoints its mutable state (the mock airline included) when a turn ends, so a desk killed mid-conversation and restarted picks up where it was. One conversation is one agent-db database; the phone agent's session and each desk context it talked to are rows in it, the desk's under the caller's.

Start agent-db locally, from `agents-private/agent-db`, and leave it running:

```bash
mage build && mage devLocal   # management :7780, data plane ws://localhost:7781/db
```

Point both processes at it. The key and secret are devLocal's own, apart from the LiveKit project's:

```bash
export LIVEKIT_AGENTDB_URL=http://localhost:7780
export LIVEKIT_AGENTDB_WS_URL=ws://localhost:7781/db
export LIVEKIT_AGENTDB_API_KEY=devkey LIVEKIT_AGENTDB_API_SECRET=secret
```

### The crash drill, with no microphone

`chat.py` is a text client over A2A. It creates a conversation database and a context, prints both, and sends each line as a person's turn.

```bash
python expert.py dev   # terminal 1
python chat.py         # terminal 2: prints conversation DB_... and context chat-...
```

1. Ask two things that build on each other: _"Hi, I'm dana@example.com. What's the status of my flight to Tokyo tomorrow?"_, then _"What other flights could you put me on that day, and what would the change cost me?"_ The desk quotes the change and keeps the quote on the booking.
2. Kill the desk hard: `kill -9 $(lsof -ti tcp:8321 -sTCP:LISTEN)`.
3. Restart it: `python expert.py dev`. A turn that was mid-call when it died is still marked `running`; the next start marks it `interrupted` and tells the model the outcome is unknown. A restart inside the dead desk's 10 s lease waits the rest of it out first.
4. In the same `chat.py`, ask a follow-up that only makes sense with what came before: _"OK, go ahead and move me onto that evening flight you just quoted."_ The desk logs `↺ rehydrated chat-...: N messages back` and rebooks from the quote it made before the crash.

`chat.py --conversation DB_... --context chat-...` picks the same conversation up from a fresh client; `--delegate` sends lines as instructions, the way the phone agent asks.

### Reading the rows

`agentdb-console` in `agents-private/agent-db` reads the database directly; the tables are the contract a dashboard reads:

```bash
alias adb='./bin/agentdb-console -database DB_...'
adb -q "SELECT session_id, parent_session_id, kind, current_agent_id, lease_owner FROM sessions"
adb -q "SELECT json_extract(item_json,'$.role') AS role, substr(json_extract(item_json,'$.content[0]'),1,80) AS text
        FROM chat_items WHERE owner = 'session' AND json_extract(item_json,'$.type') = 'message' ORDER BY created_at"
adb -q "SELECT call_id, name, status, is_error, substr(output,1,60) AS output FROM tasks ORDER BY started_at"
adb -q "SELECT session_id, call_id, child_session_id, task_id, status FROM delegations ORDER BY created_at"
```

`delegations` fills from the phone agent's side: each row says which desk context and task answered which delegate call.

### The voice half

`voice.py` persists too once it is given a conversation. The session id `voice` is the app's choice, stable across calls:

```bash
CONVERSATION=DB_... python voice.py console
```

Hang up and run it again on the same `CONVERSATION`: the call resumes with what was said before, and its delegations reach the same desk context, which rehydrates on the first one.

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
