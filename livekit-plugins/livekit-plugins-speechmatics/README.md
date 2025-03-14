# LiveKit Plugins Speechmatics

Agent Framework plugin for Speechmatics.

## Installation

```bash
pip install livekit-plugins-speechmatics
```

### Usage:

```python
agent = VoicePipelineAgent(
    stt=speechmatics.STT(),
    turn_detector=turn_detector.EOUModel(),
    min_endpointing_delay=0.5,
    max_endpointing_delay=5.0,
    ...
)
```

### Note on End-of-Turn Detection

This plugin was designed with LiveKit's [end-of-turn detection feature](https://github.com/livekit/agents#in-house-phrase-endpointing-model) in mind.
`AddTranscript` and `AddPartialTranscript` events are emitted as soon as they’re received from the STT engine.

For the best user experience, we recommend running the agent with end-of-turn detection enabled.
See [this example](https://github.com/livekit-examples/voice-pipeline-agent-python/blob/main/agent.py) for implementation details.

### End-of-Utterance Detection in Version `0.1.0`

Starting in version `0.1.0`, the plugin introduces an alternative end-of-utterance detection method
that doesn’t rely on LiveKit's built-in end-of-turn detection.

This method uses:
- A silence timeout (`min_endpointing_delay`) measured from the last non-empty `AddPartialTranscript` or `AddTranscript` message.
- The `is_eos` flag returned by the STT engine to determine when the user has stopped speaking.

Unlike the default behaviour, `AddTranscript` and `AddPartialTranscript` events are **not** emitted immediately.
Instead, they’re stored in an internal transcript buffer.
All buffered transcript messages are released when:
1. The configured `min_endpointing_delay` expires, and/or
2. The `is_eos` flag is set to `True`.


```python
agent = VoicePipelineAgent(
    stt=speechmatics.STT(min_endpointing_delay=0.8)
    ...
)
```

## Pre-requisites

You'll need to specify a Speechmatics API Key which you can get from https://portal.speechmatics.com.
It can be set as environment variable `SPEECHMATICS_API_KEY` or `.env.local` file.
