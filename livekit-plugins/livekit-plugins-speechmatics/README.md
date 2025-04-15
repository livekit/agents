# LiveKit Plugins Speechmatics

Agent Framework plugin for Speechmatics.

## Installation

```bash
pip install livekit-plugins-speechmatics
```

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

Note: The plugin was built with
LiveKit's [end-of-turn detection feature](https://docs.livekit.io/agents/v1/build/turn-detection/) in mind,
and it doesn't implement phrase endpointing. `AddTranscript` and `AddPartialTranscript` events are emitted as soon
as they’re received from the Speechmatics STT engine.

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable `SPEECHMATICS_API_KEY` or
`.env.local` file.
