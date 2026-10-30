# Vehicle service

The vehicle service demo from the LiveKit homepage. Nick works the service desk at Bayside BMW, a
made-up dealership, to show how a voice agent books a service visit and calls a customer back about
a repair.

## The calls

- **Inbound, booking a visit:** Nick finds the caller's file or opens a new one, picks the car,
  and asks about the problem one question at a time. Then he adds the right service, finds a time,
  sets up the lounge, the shuttle, or a loaner, and reads the visit back before booking it. He can
  also move or cancel a visit that's already booked, and tells the caller not to drive a car that
  isn't safe.
- **Outbound, repair callback:** Nick calls once the technician has found the cause. He goes over
  each repair with its price and when the car would be ready, records what the customer wants done
  now or next time, and arranges how they'll get the car back.

The job metadata picks the call. `"direction": "outbound"` makes it the repair callback, and
`"dial"`, with `"trunk"` and `"from"`, rings a real phone. `"today"` or `"timezone"` sets the date
the calendar starts from. Otherwise the agent waits for someone to join the room.

## How it's built

- `src/agent.py` is the entrypoint for the `vehicle-service` agent. It sets up the session and
  starts the right call.
- `src/advisor.py` holds Nick's two roles, `ServiceAdvisor` and `RepairCallback`, with their
  instructions and tools.
- `src/service_desk.py` is the service desk for one call: the customer's file, the visit being put
  together, the calendar, and the car in the shop on the callback. Every rule lives here, and it
  publishes each change as a `service.*` attribute for the demo page.
- `src/bayside.json` seeds the dealership, its customers, service menu, recalls, calendar, and the
  repair the technician finds.
- `src/conversation.py` reads the session history to tell whether the caller actually heard
  something, like the read-back, and has answered since.
- `src/simulation.py` grades a simulated call on what the desk ended up with, against the
  scenario's `expected` userdata.
- `src/suggested_replies.py` suggests what the caller could say next after each of Nick's turns,
  for the page's "try saying" line.
- `src/phone.py` rings a real phone for an outbound call.

The voice pipeline runs on LiveKit Inference: `assemblyai/universal-3-6-pro` for speech to text,
Gemma 4 31B (`google/gemma-4-31b-it`) as the LLM, Rime Coda (`rime/coda`) for the voice, and the
LiveKit turn detector. Suggested replies also use Gemma 4 31B, and ai-coustics cleans up the
caller's audio.

The demo page reads the `service.*` attributes, so changing their shape needs a matching change to
the page.

## Run it locally

```console
uv sync
lk app env --write --destination .env.local
lk agent dev
```

Then talk to it in the [Agent Console](https://docs.livekit.io/testing/agent-console/), or run
`lk agent console` to talk to it in your terminal. To try it one text turn at a time, use the
[agent debugger](https://docs.livekit.io/testing/debugger/):

```console
lk agent debugger start
lk agent debugger say "Hi, I need an oil change on my X3."
lk agent debugger stop
```

## Tests and simulations

```console
uv run pytest
lk agent simulate text --scenarios scenarios.yaml
```

`tests/test_agent.py` runs turns against the real LLM, so it needs the credentials in `.env.local`.
Each scenario's `expected` userdata says what the visit should end up as, and `src/simulation.py`
fails the run if it doesn't.

## Deploy

```console
lk agent deploy .
```

This needs a `livekit.toml` that names the agent. It's gitignored, and the repository's
[`Deploy Homepage Examples`](../../../.github/workflows/deploy-homepage-examples.yml) workflow
writes one for each homepage demo.
