# Expressive agent

A free-form voice agent that demonstrates [Expressive Mode](https://docs.livekit.io/agents/build/expressive/).
There is no task and no tool: you talk to it like a friend, and it matches your
register. Tell it good news and it gets excited; tell it something went wrong
and it drops the energy.

Expressive Mode is the single `expressive=True` flag on `AgentSession`. With it
enabled the framework injects the TTS provider's markup guide into the LLM
prompt, so the model emits inline delivery tags (emotion, pacing, non-verbal
sounds) that the TTS renders and the transcript never shows.

## Architecture

- `agent.py` is the composition root: session setup and the server entrypoint.
- `prompt.md` holds the persona only. It steers *what* the agent says, and
  expressive mode owns *how* it sounds, so the two never restate each other.
- `protocol.py` is the whole frontend contract: the dispatch metadata shape, the
  attributes echoed back, and the voice table those metadata values name.

The pipeline uses LiveKit Inference with Gemma 4 31B, Deepgram Nova-3, Fish
Audio S2.1 Pro, and the LiveKit turn detector.

## Run locally

Provide LiveKit Cloud credentials in `../.env` or the environment, then:

```bash
uv sync --all-extras --dev   # from the repository root
uv run agent.py console
```

Use `uv run agent.py dev` to connect the agent to LiveKit Cloud for a frontend
session.

## Configuration

The agent reads its dispatch metadata, so a frontend can pick the pipeline at
connect time without a redeploy. `protocol.py` is the contract, in both
directions:

```json
{ "expressive": true, "tts": "fishaudio" }
```

- `expressive` (default `true`) toggles Expressive Mode.
- `tts` selects a voice from `protocol.py`: `fishaudio`, `inworld`, `cartesia`, or `xai`.

Both values are echoed back as participant attributes (`expressive`,
`tts_provider`, `tts_label`) so the frontend can display the active pipeline.

Note that xAI steers delivery through prosody and sound tags but has no
expression tag, so it publishes no `lk.expression`. Its speech is expressive;
a frontend mood indicator just has nothing to read. See `protocol.py`.

## Trying it with and without expressive

The comparison is the point of the demo. Run it once with `expressive=True` and
once with `expressive=False`, and say the same thing to each. The words come out
much the same; the delivery does not.

Expressive Mode requires a `livekit.agents.inference.TTS` model that declares a
markup dialect. Fish Audio, Inworld TTS 2, Cartesia Sonic 3, and xAI qualify;
providers without a dialect synthesize normally and the flag stays inert.
