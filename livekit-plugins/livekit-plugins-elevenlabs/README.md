# ElevenLabs plugin for LiveKit Agents

Support for voice synthesis with [ElevenLabs](https://elevenlabs.io/).

See [https://docs.livekit.io/agents/integrations/tts/elevenlabs/](https://docs.livekit.io/agents/integrations/tts/elevenlabs/) for more information.

## Installation

```bash
pip install livekit-plugins-elevenlabs
```

## Pre-requisites

You'll need an API key from ElevenLabs. It can be set as an environment variable: `ELEVEN_API_KEY`

## Realtime speech-to-text audio chunks

Configure the outgoing audio chunk duration when creating the STT instance:

```python
from livekit.plugins import elevenlabs

stt = elevenlabs.STT(
    model="scribe_v2_realtime",
    audio_chunk_duration_ms=100,
)
```

`audio_chunk_duration_ms` accepts a positive integer in milliseconds and defaults
to 50. At 100 ms, continuous audio produces approximately 10 audio messages per
second instead of 20, at the cost of up to 50 ms additional buffering. Larger
chunks do not pace reconnect backlogs or guarantee that provider queue errors
are avoided. Flushes send any remaining partial chunk before the commit.

This option applies only to realtime STT and is set at construction time.

## Scribe v3 Live turn prediction

Enable server-side turn prediction with `scribe_v3_live`:

```python
stt = elevenlabs.STT(
    model="scribe_v3_live",
    commit_strategy="turn_prediction",
    server_vad={"vad_silence_threshold_secs": 2.0},
)
```

Set `turn_handling={"turn_detection": "stt"}` on `AgentSession` to use the
server's commits as turn boundaries. `server_vad` is optional and configures the
VAD fallback. Other models do not support `turn_prediction`.

The other commit strategies are `manual` and `vad`. If `commit_strategy` is
omitted, the plugin uses `vad` when `server_vad` is a dict and `manual` otherwise.
An explicit strategy takes precedence over `server_vad`.

Use `stt.update_options(commit_strategy=...)` to change strategies. This reconnects
active streams, so wait for the current turn's final transcript first.
