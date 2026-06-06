# Speechmatics STT plugin for LiveKit Agents

Support for Speechmatics STT.

See [https://docs.livekit.io/agents/integrations/stt/speechmatics/](https://docs.livekit.io/agents/integrations/stt/speechmatics/) for more information.

## Installation

```bash
pip install livekit-plugins-speechmatics
```

## Diarization

Speechmatics STT engine can be configured to emit information about individual speakers in a conversation. This needs to be enabled using `enable_diarization=True`. The text output of the transcription can be configured to include this information using the macros `speaker_id` and `text`, as shown in the examples below.

- `<{speaker_id}>{text}</{speaker_id}>` -> `<S1>Hello</S1>`
- `[Speaker {speaker_id}] {text}` -> `[Speaker S1] Hello`

You should adjust your system instructions to inform the LLM of this format for speaker identification.

## Turn detection modes

The `turn_detection_mode` parameter controls how end-of-turn is detected:

- `EXTERNAL` (default) — Speechmatics does not endpoint on its own; turn boundaries are driven by an external VAD or by calling `finalize()`. If no `vad` is passed, Silero is auto-loaded (requires `livekit-plugins-silero`). Pass `vad=None` to opt out and drive `finalize()` yourself.
- `ADAPTIVE` — Speechmatics controls end of turn using its own VAD and the pace of speech.
- `SMART_TURN` — Speechmatics ML-based endpointing.
- `FIXED` — Endpoints after a fixed silence duration set by `end_of_utterance_silence_trigger`.

## Usage (LiveKit Turn Detection)

The default `EXTERNAL` mode pairs naturally with LiveKit's turn detector. The format for the output text needs to be adjusted to not include any extra content at the end of the utterance. Using `[Speaker S1] ...` as the `speaker_active_format` should work well. You may need to adjust your system instructions to inform the LLM of this format for speaker identification. You must also include the listener for when the VAD has detected the end of speech.

The `end_of_utterance_silence_trigger` parameter controls the amount of silence before the end of turn detection is triggered. The default is `0.5` seconds.

Usage:

```python
from livekit.agents import AgentSession, inference
from livekit.agents.inference import TurnDetector
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        end_of_utterance_silence_trigger=0.2,
        speaker_active_format="[Speaker {speaker_id}] {text}",
        speaker_passive_format="[Speaker {speaker_id} *PASSIVE*] {text}",
    ),
    vad=inference.VAD(),
    turn_detection=TurnDetector(),
    min_endpointing_delay=0.3,
    max_endpointing_delay=5.0,
    ...
)
```

## Usage (Speechmatics end of utterance detection and speaker ID)

To delegate end-of-turn detection to Speechmatics, set `turn_detection_mode=TurnDetectionMode.ADAPTIVE` (or `SMART_TURN` / `FIXED`) and pair it with `turn_detection="stt"` on the `AgentSession`.

```python
from livekit.agents import AgentSession
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        turn_detection_mode=speechmatics.TurnDetectionMode.ADAPTIVE,
        speaker_active_format="[Speaker {speaker_id}] {text}",
        speaker_passive_format="[Speaker {speaker_id} *PASSIVE*] {text}",
        additional_vocab=[
            speechmatics.AdditionalVocabEntry(
                content="LiveKit",
                sounds_like=["live kit"],
            ),
        ],
    ),
    turn_detection="stt",
    ...
)
```

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable `SPEECHMATICS_API_KEY` or `.env.local` file.
