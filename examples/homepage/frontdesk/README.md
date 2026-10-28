# Front desk

The front desk demo from the LiveKit homepage. Riley answers the phone at Juniper Health, Dr. Maya
Okafor's practice, to show how a voice agent checks who's calling before it opens a patient's
records. The clinic, its patients, and its calendar are made up.

## The calls

- **Inbound, front desk:** Riley finds the caller's chart by their date of birth and the spelling of
  their name, or registers them as a new patient. Then it books an appointment or moves one they
  already have.
- **Outbound, confirmation call:** Riley calls a patient about their next appointment, verifies
  them, and confirms it. It takes a note for the doctor, or moves the appointment if the time no
  longer works.

The job metadata picks the call. `"direction": "outbound"` makes it the confirmation call, and
`"dial"`, with `"trunk"` and `"from"`, rings a real phone. Otherwise the agent waits for someone to
join the room.

## How it's built

- `src/agent.py` is the entrypoint for the `homepage-front-desk` agent. It sets up the session and
  starts the right call.
- `src/receptionist.py` holds Riley's two roles, `Receptionist` and `ConfirmationCall`, with their
  instructions and tools.
- `src/clinic.py` is the made-up clinic: its charts, calendar, and appointments. It publishes each
  change as a `clinic.*` attribute for the demo page.
- `src/schedule.json` seeds the clinic, its patients, open times, and appointments.
- `src/suggested_replies.py` suggests what the caller could say next after each of Riley's turns,
  for the page's "try saying" line.
- `src/phone.py` rings a real phone for an outbound call.

Riley talks through OpenAI GPT-Live (`openai/gpt-live-1`) on LiveKit Inference. GPT-Live hands
every tool call to a backend model (`openai/gpt-5.6-luna`), so the clinic's records only change when
the backend calls a tool. Suggested replies use Gemma 4 31B, and ai-coustics cleans up the caller's
audio.

The demo page reads the `clinic.*` attributes, so changing their shape needs a matching change to
the page.

## Run it locally

```console
uv sync
lk app env --write --destination .env.local
lk agent dev
```

Then talk to it in the [Agent Console](https://docs.livekit.io/testing/agent-console/), or run
`lk agent console` to talk to it in your terminal. GPT-Live only takes audio, so start the
[agent debugger](https://docs.livekit.io/testing/debugger/) with `lk agent debugger start --audio`.

## Tests and simulations

```console
uv run pytest
lk agent simulate audio --scenarios scenarios.yaml
```

Run the simulations in audio: GPT-Live doesn't work in text simulations.

## Deploy

```console
lk agent deploy .
```

This needs a `livekit.toml` that names the agent. It's gitignored, and the repository's
[`Deploy Homepage Examples`](../../../.github/workflows/deploy-homepage-examples.yml) workflow
writes one for each homepage demo.
