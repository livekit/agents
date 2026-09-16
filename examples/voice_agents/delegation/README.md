# Delegation over A2A

A voice agent that talks, and an expert that thinks. Two processes on one machine, speaking
[A2A](https://a2a-protocol.org) with the LiveKit agent session extension.

The voice agent has no tools of its own beyond the one delegation gives it, and knows nothing
about fares. It keeps the caller company, sends every real question to the fare desk, and
phrases whatever comes back. The fare desk never speaks to the caller: it returns facts.

## Running it

Two terminals. The expert first, because the voice agent connects to it:

```bash
python expert.py dev      # serves http://localhost:8321/fare-desk
python voice.py console   # talk to it
```

Ask something that needs a lookup, such as *"what would it cost to move my Monday flight to
Tuesday, and is there space?"*. You should hear the agent acknowledge, then report what the
expert is doing while it does it, then answer.

## What to look at

- **`expert.py`** — `@server.text_session(endpoint="fare-desk")` serves an `AgentSession` over
  A2A on the agent server's own HTTP app. The handler builds the session, starts it, and hands
  it over; each incoming request becomes a turn of it.
- **`ctx.update()`** inside `check_availability` and `hold_seat` reports progress while the
  work is still running, and releases the turn. That report is relayed to the voice agent as
  the tool wrote it, not handed to a model to restate.
- **`ctx.session.request`** in `end_of_call` is how the expert tells whether a caller is
  waiting on an answer. When one is, it sets a directive that rides back with the answer; when
  nobody is, there is nothing to advise.
- **`voice.py`** — `delegate=A2ADelegate(url)` is the whole of the voice side. The session
  closes the delegate when the call ends, and `delegation_directive` is where the directive
  arrives, after the answer has been spoken.

## Talking to it without a voice agent

The endpoint is plain A2A, so anything that speaks it can use the expert:

```bash
curl localhost:8321/fare-desk/.well-known/agent-card.json

curl -X POST localhost:8321/fare-desk/v1/message:stream -N \
     -H 'content-type: application/json' \
     -d '{"message": {"messageId": "m1", "role": "ROLE_USER",
          "parts": [{"text": "what does it cost to move Monday to Tuesday?"}]}}'
```

A client that does not know the extension sees text and task states and works fine. One that
does also gets the typed chat items, verbatim text and directives.
