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

To use the Speechmatics end of utterance detection and speaker ID, you can use the following configuration.

Note: The `turn_detection_mode` parameter tells the plugin to control the end of turn detection. The default is `FIXED`, which means that the plugin will emit finalized words after a period of silence (controlled by the `end_of_utterance_silence_trigger` value). In this example we use `ADAPTIVE` mode, which means that the plugin will control the end of turn detection using the plugin's own VAD detection and the pace of speech. The `turn_detection="stt"` parameter tells the plugin to use the STT engine's end of turn detection.

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

## Usage (LiveKit Turn Detection)

To use the LiveKit end of turn detection, the format for the output text needs to be adjusted to not include any extra content at the end of the utterance. Using `[Speaker S1] ...` as the `speaker_active_format` should work well. You may need to adjust your system instructions to inform the LLM of this format for speaker identification. You must also include the listener for when the VAD has detected the end of speech.

The `end_of_utterance_silence_trigger` parameter controls the amount of silence before the end of turn detection is triggered. The default is `0.5` seconds.

Usage:

```python
from livekit.agents import AgentSession
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import speechmatics, silero

agent = AgentSession(
    stt=speechmatics.STT(
        turn_detection_mode=speechmatics.TurnDetectionMode.EXTERNAL,
        speaker_active_format="[Speaker {speaker_id}] {text}",
        speaker_passive_format="[Speaker {speaker_id} *PASSIVE*] {text}",
    ),
    vad=silero.VAD.load(),
    turn_detection=EnglishModel(),
    min_endpointing_delay=0.3,
    max_endpointing_delay=5.0,
    ...
)

@agent.on("user_state_changed")
def on_user_state(state):
    if state.new_state == "listening" and state.old_state == "speaking":
        stt.finalize()

```

Note: The plugin was built with LiveKit's [end-of-turn detection feature](https://docs.livekit.io/agents/v1/build/turn-detection/) in mind, and it doesn't implement phrase endpointing. `AddTranscript` and `AddPartialTranscript` events are emitted as soon as they’re received from the Speechmatics STT engine.

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable `SPEECHMATICS_API_KEY` or `.env.local` file.
