# Restaurant ordering

The restaurant ordering demo from the LiveKit homepage. Gabby answers the phone at Zappi's
Pizzeria, a made-up neighborhood pizzeria in San Francisco, to show how a voice agent takes an
order, reads it back, and sends it to the kitchen.

## The calls

- **Inbound, taking an order:** Gabby takes the order from the menu, answers allergen questions,
  sets up pickup or delivery, takes the caller's name and number, and reads the order back with
  its total before it goes to the kitchen.
- **Outbound, delivery update:** Gabby calls about order #214, about to go out with the driver,
  because one item ran out. The customer can swap it for something at the same price or take it
  off, add breadsticks, leave a note for the driver, or cancel.

The job metadata picks the call. `"direction": "outbound"` makes it the delivery update, with
`"name"` for the customer's name, and `"dial"`, with `"trunk"` and `"from"`, rings a real phone.
Otherwise the agent waits for someone to join the room.

## How it's built

- `src/agent.py` is the entrypoint for the `restaurant-ordering` agent. It sets up the session and
  starts the right call.
- `src/gabby.py` holds Gabby's two roles, `TakeOrder` and `DeliveryUpdate`, with their
  instructions and tools.
- `src/pizzeria.py` is the point of sale: the menu, the order being taken, and order #214. It
  checks every change against the restaurant's rules and publishes it as a `restaurant.*`
  attribute for the demo page.
- `src/zappis.json` seeds the menu, prices, fees, hours, and allergens.
- `src/conversation.py` reads the session history to tell whether the caller actually heard
  something, like the read-back, and has answered since.
- `src/simulation.py` grades a simulated call on what the point of sale ended up with, against the
  scenario's `expected` userdata.
- `src/suggested_replies.py` suggests what the caller could say next after each of Gabby's turns,
  for the page's "try saying" line.
- `src/phone.py` rings a real phone for an outbound call.

The voice pipeline runs on LiveKit Inference: `assemblyai/universal-3-6-pro` for speech to text,
Gemma 4 31B (`google/gemma-4-31b-it`) as the LLM, Fish Audio S2.1 Pro (`fishaudio/s2.1-pro`) for
the voice, and the LiveKit turn detector. Suggested replies also use Gemma 4 31B, and ai-coustics
cleans up the caller's audio.

The demo page reads the `restaurant.*` attributes, so changing their shape needs a matching change
to the page.

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
lk agent debugger say "Hi, can I get a large pepperoni for pickup?"
lk agent debugger stop
```

## Tests and simulations

```console
uv run pytest
lk agent simulate text --scenarios scenarios.yaml
```

`tests/test_agent.py` runs turns against the real LLM, so it needs the credentials in `.env.local`.
Each scenario's `expected` userdata says what the order should end up as, and `src/simulation.py`
fails the run if it doesn't.

## Deploy

```console
lk agent deploy .
```

This needs a `livekit.toml` that names the agent. It's gitignored, and the repository's
[`Deploy Homepage Examples`](../../../.github/workflows/deploy-homepage-examples.yml) workflow
writes one for each homepage demo.
