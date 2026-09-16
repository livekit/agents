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

| caller | what makes them worth asking about |
|---|---|
| `dana@example.com` | Gold, and her Tokyo flight tomorrow is delayed 245 minutes — our fault, so the change fee is waived and her seat moves for nothing |
| `ortiz@example.com` | on a BASIC fare, which cannot be changed or refunded at all |
| `raman@example.com` | holds 120 USD of travel credit, which a new booking spends |

*"My flight to Tokyo tomorrow is delayed — what else can you put me on?"* makes the desk
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

- **`@server.text_session(endpoint="fare-desk")`** serves an `AgentSession` over A2A on the
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
  session closes the delegate when the call ends, which tells the desk to drop the
  conversation rather than wait for it to go idle.

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
