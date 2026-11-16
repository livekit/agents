# Card verification

The card verification demo from the LiveKit homepage. Avery plays a card services agent at
Kestrel Bank, a made-up bank, to show how a voice agent verifies a caller before it touches their
account.

Nothing is real. Any name and birthday work, and the card's last four digits decide the check:
`1111` passes and anything else, like `0000`, fails. Three failed checks lock the caller out for
the rest of the call.

## The calls

- **Inbound, card services:** Avery verifies the caller, then looks into a charge they don't
  recognize, locks a lost card, or sets a travel notice. A charge they didn't make locks the card
  and opens a loss prevention case. Anything else goes to a card specialist.
- **Outbound, fraud alert:** Avery calls about a flagged purchase abroad, confirms the cardholder
  by name and birthday, and asks whether they made it. If they didn't, the card is locked and the
  case goes to loss prevention.

The job metadata picks the call. `"direction": "outbound"` makes it the fraud alert, and `"dial"`,
with `"trunk"` and `"from"`, rings a real phone. Otherwise the agent waits for someone to join the
room.

## How it's built

- `src/agent.py` is the entrypoint for the `card-verification` agent. It sets up the session and
  starts the right call.
- `src/card_services.py` holds Avery's two roles, `CardServices` and `FraudAlert`, with their
  instructions and tools.
- `src/bank.py` is the made-up bank: the identity check, the lockout, the flagged purchase, the
  lock, and the travel notice. It publishes each change as an `auth.*` attribute for the demo page.
- `src/simulation.py` grades a simulated call on what the bank ended up with, against the
  scenario's `expected` userdata.
- `src/suggested_replies.py` suggests what the caller could say next after each of Avery's turns,
  for the page's "try saying" line.
- `src/phone.py` rings a real phone for an outbound call.

The voice pipeline runs on LiveKit Inference: `assemblyai/universal-3-6-pro` for speech to text,
Gemma 4 31B (`google/gemma-4-31b-it`) as the LLM, Rime Coda (`rime/coda`) for the voice, and the
LiveKit turn detector. Suggested replies also use Gemma 4 31B, and ai-coustics cleans up the
caller's audio.

The demo page reads the `auth.*` attributes, so changing their shape needs a matching change to the
page.

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
lk agent debugger say "Hi, I need to lock my card."
lk agent debugger stop
```

## Tests and simulations

```console
uv run pytest
lk agent simulate text --scenarios scenarios.yaml
```

Each scenario's `expected` userdata says what the bank should end up with, and
`src/simulation.py` fails the run if it doesn't. The quiet-caller scenario needs real silence, so
it's in `scenarios-audio.yaml` and runs with `lk agent simulate audio --scenarios scenarios-audio.yaml`.

## Deploy

```console
lk agent deploy .
```

This needs a `livekit.toml` that names the agent. It's gitignored, and the repository's
[`Deploy Homepage Examples`](../../../.github/workflows/deploy-homepage-examples.yml) workflow
writes one for each homepage demo.
