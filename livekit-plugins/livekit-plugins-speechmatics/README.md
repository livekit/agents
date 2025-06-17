# Speechmatics STT plugin for LiveKit Agents

Support for Speechmatics STT.

See [https://docs.livekit.io/agents/integrations/stt/speechmatics/](https://docs.livekit.io/agents/integrations/stt/speechmatics/) for more information.

## Installation

```bash
pip install livekit-plugins-speechmatics
```

## Usage (Speechmatics end of utterance detection and speaker ID)

To use the Speechmatics end of utterance detection and speaker ID, you can use the following configuration:

```python
from livekit.agents import AgentSession
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(
        transcription_config=speechmatics.types.TranscriptionConfig(
            max_delay=2.0,
            max_delay_mode="fixed",
            diarization="speaker",
            speaker_diarization_config=speechmatics.types.RTSpeakerDiarizationConfig(max_speakers=10),
            conversation_config=speechmatics.types.RTConversationConfig(end_of_utterance_silence_trigger=0.7),
        ),
    ),
    ...
)
```

Note: Using the `end_of_utterance_silence_trigger` parameter will tell the STT engine to wait for this period of time from the last detected speech and then emit the full utterance to LiveKit. This may conflict with LiveKit's end of turn detection, so you may need to adjust the `min_endpointing_delay` and `max_endpointing_delay` parameters accordingly.

## Usage (LiveKit Turn Detection)

Usage:

```python
from livekit.agents import AgentSession
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins import speechmatics

agent = AgentSession(
    stt=speechmatics.STT(),
    turn_detector=EnglishModel(),
    min_endpointing_delay=0.5,
    max_endpointing_delay=5.0,
    ...
)
```

Note: The plugin was built with LiveKit's [end-of-turn detection feature](https://docs.livekit.io/agents/v1/build/turn-detection/) in mind, and it doesn't implement phrase endpointing. `AddTranscript` and `AddPartialTranscript` events are emitted as soon as they’re received from the Speechmatics STT engine.

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable `SPEECHMATICS_API_KEY` or
`.env.local` file.
