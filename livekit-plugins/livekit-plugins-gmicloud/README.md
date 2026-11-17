# Gmicloud plugin for LiveKit Agents

Support for [GMI Cloud](https://www.gmicloud.ai)-hosted speech models in LiveKit
Agents. One plugin covers the TTS model families behind GMI Cloud's
request-queue API: Inworld Realtime TTS, MiniMax speech synthesis, and Resemble
AI's Chatterbox.

## Installation

```bash
pip install livekit-plugins-gmicloud
```

## Pre-requisites

You'll need a GMI Cloud API key from the
[GMI Cloud console](https://console.gmicloud.ai/). It can be set as an
environment variable: `GMI_CLOUD_API_KEY`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins.gmicloud import TTS

session = AgentSession(
    tts=TTS(model="inworld-tts-2", voice="Dennis"),
    # ... llm, stt, turn_detection ...
)
```

Any GMI Cloud audio model id can be passed as `model`; the plugin selects the
request shape from the id prefix:

| Family | Model ids | Request shape |
| --- | --- | --- |
| Inworld Realtime TTS | `inworld-tts-*` | `voice_id`, `audio_encoding`, `speaking_rate`, `temperature`, `timestamp_type` |
| MiniMax speech synthesis | `minimax-tts-speech-*` | `voice_id`, `speed`, `format` |
| Resemble AI's Chatterbox | `Chatterbox-tts` | requires a Resemble `voice_uuid` as `voice` |

Ids that match none of the prefixes fall back to the Inworld request shape.
Provider-specific fields can be passed through with `extra_payload`.

Responses may arrive synchronously or as queued/dispatched generations that the
plugin resolves by polling. Queued generations are slow on GMI Cloud (the
plugin polls for up to 3 minutes by default, configurable via `poll_attempts`
and `poll_interval_secs`); prefer the Inworld family for latency-sensitive
agents. Output is WAV at the model's native rate; the plugin reads the real
sample rate and channel count from each response and emits audio frames at
that format. Synthesis is chunked (non-streaming).
