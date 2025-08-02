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

## Usage (Speechmatics end of utterance detection and speaker ID)

To use the Speechmatics end of utterance detection and speaker ID, you can use the following configuration:

```python
from livekit.agents import AgentSession
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        end_of_utterance_silence_trigger=0.5,
        enable_diarization=True,
        speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
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

Note: Using the `end_of_utterance_silence_trigger` parameter will tell the STT engine to wait for this period of time from the last detected speech and then emit the full utterance to LiveKit. This may conflict with LiveKit's end of turn detection, so you may need to adjust the `min_endpointing_delay` and `max_endpointing_delay` parameters accordingly.

## Usage (LiveKit Turn Detection)

To use the LiveKit end of turn detection, the format for the output text needs to be adjusted to not include any extra content at the end of the utterance. Using `[Speaker S1] ...` as the `speaker_active_format` should work well. You may need to adjust your system instructions to inform the LLM of this format for speaker identification.

Usage:

```python
from livekit.agents import AgentSession
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        enable_diarization=True,
        end_of_utterance_mode=speechmatics.EndOfUtteranceMode.NONE,
        speaker_active_format="[Speaker {speaker_id}] {text}",
    ),
    turn_detector=EnglishModel(),
    min_endpointing_delay=0.5,
    max_endpointing_delay=5.0,
    ...
)
```

Note: The plugin was built with LiveKit's [end-of-turn detection feature](https://docs.livekit.io/agents/v1/build/turn-detection/) in mind, and it doesn't implement phrase endpointing. `AddTranscript` and `AddPartialTranscript` events are emitted as soon as they’re received from the Speechmatics STT engine.

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable `SPEECHMATICS_API_KEY` or `.env.local` file.
