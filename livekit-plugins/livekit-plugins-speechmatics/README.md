# LiveKit Plugins Speechmatics

Agent Framework plugin for Speechmatics.

## Installation

```bash
pip install livekit-plugins-speechmatics
```

Usage:

```python
agent = VoicePipelineAgent(
    stt=speechmatics.STT(),
    turn_detector=turn_detector.EOUModel(),
    min_endpointing_delay=0.5,
    max_endpointing_delay=5.0,
    ...
)
```

Note: The plugin was built with
LiveKit's [end-of-turn detection feature](https://github.com/livekit/agents#in-house-phrase-endpointing-model) in mind,
and it doesn't implement phrase endpointing. `AddTranscript` and `AddPartialTranscript` events are emitted as soon
as they’re received from the Speechmatics STT engine. For the best user experience,
we recommend running the agent with end-of-turn detection enabled (
see [example](https://github.com/livekit-examples/voice-pipeline-agent-python/blob/main/agent.py)).

## Pre-requisites

You'll need to specify a Speechmatics API Key. It can be set as environment variable `SPEECHMATICS_API_KEY` or
`.env.local` file.
