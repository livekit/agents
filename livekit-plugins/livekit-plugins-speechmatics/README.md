# Speechmatics STT plugin for LiveKit Agents

Support for Speechmatics STT.

See [https://docs.livekit.io/agents/integrations/stt/speechmatics/](https://docs.livekit.io/agents/integrations/stt/speechmatics/) for more information.

## Installation

```bash
pip install livekit-plugins-speechmatics
```

## Model

`model` selects the transcription model and defaults to `linden-1`, currently the only Agent STT
model:

```python
stt = speechmatics.STT(model="linden-1")
```

`operating_point` is a deprecated alias for `model` and warns when used. The RT operating points
`enhanced` and `standard` are not Agent STT models and are rejected by the service.

## Turn detection modes

The `turn_detection_mode` parameter controls how end-of-turn (endpointing) is detected:

- `EXTERNAL` (default) — Speechmatics does not endpoint on its own. Turns close when the caller
  calls `finalize()`. In practice you pass a `vad` to the plugin and its end-of-speech drives
  `finalize()`; LiveKit does **not** call `finalize()` for you, and no VAD is auto-loaded. Without a
  `vad` (and without calling `finalize()` yourself) turns never close, so nothing is finalized.
- `VAD` — Speechmatics runs its own VAD and closes turns itself (service-side endpointing). No `vad`
  is required. Pair it with `turn_detection="stt"` on the `AgentSession`, otherwise the session's own
  turn detector decides and Speechmatics' end-of-turn is ignored.

## Usage — service-side endpointing (`VAD`)

Let Speechmatics detect turns and tell the session to act on them:

```python
from livekit.agents import AgentSession
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        turn_detection_mode=speechmatics.TurnDetectionMode.VAD,
    ),
    turn_detection="stt",
    ...
)
```

## Usage — caller-driven endpointing (`EXTERNAL`, default)

Pass a `vad` to the plugin; its end-of-speech drives `finalize()`. `AgentSession` loads its own VAD
when none is given, so pass the same instance to both and a single VAD serves the session and the
plugin:

```python
from livekit.agents import AgentSession, inference
from livekit.plugins import speechmatics

vad = inference.VAD()

agent = AgentSession(
    stt=speechmatics.STT(
        # EXTERNAL is the default; a VAD passed here drives finalize() on end-of-speech.
        vad=vad,
        speaker_active_format="[Speaker {speaker_id}] {text}",
    ),
    vad=vad,
    ...
)
```

## Diarization

Speechmatics attributes each transcript segment to a speaker. Diarization is enabled by default
(`enable_diarization=True`); the segment is the unit of attribution, so each result carries a single
`speaker_id` and there is no per-word speaker data. To fold the speaker label into the transcript
text, set `speaker_active_format` using the `{speaker_id}` and `{text}` placeholders:

- `speaker_active_format="<{speaker_id}>{text}</{speaker_id}>"` -> `<S1>Hello</S1>`
- `speaker_active_format="[Speaker {speaker_id}] {text}"` -> `[Speaker S1] Hello`

Adjust your system instructions to inform the LLM of this format so it can attribute speakers.

```python
from livekit.agents import AgentSession
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        enable_diarization=True,
        max_speakers=4,
        speaker_active_format="[Speaker {speaker_id}] {text}",
        additional_vocab=[
            speechmatics.AdditionalVocabEntry(
                content="LiveKit",
                sounds_like=["live kit"],
            ),
        ],
    ),
    ...
)
```

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable
`SPEECHMATICS_API_KEY` or in a `.env.local` file.
