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
from livekit.plugins import elevenlabs, silero

stt = elevenlabs.STT(
    model="scribe_v3_live",
    commit_strategy="turn_prediction",
    server_vad={"vad_silence_threshold_secs": 2.0},
    vad=silero.VAD.load(),
)
```

Set `turn_handling={"turn_detection": "stt"}` on `AgentSession` to use the
server's commits as turn boundaries. `server_vad` is optional and configures the
VAD fallback. Other models do not support `turn_prediction`.

> **Warning:** With `scribe_v3_live` and `turn_prediction`, the server can force
> a segment commit after approximately 36 seconds of audio while the user is
> still speaking. Without a local `vad` passed to `elevenlabs.STT`, the plugin
> treats that commit as end of speech and can trigger an early agent response.

The optional local `vad` helps detect forced segment commits during long turns.
The plugin keeps the turn open when the last turn probability is below 0.9,
34–38 seconds of audio have been sent since the previous server commit, and
local VAD still detects speech. The plugin emits the final transcript immediately.
This is a heuristic because the provider does not report why it committed.

If no new segment text arrives, the plugin emits end of speech after the server
silence threshold plus 0.5 seconds (2.0 seconds with the server default).
When new segment text arrives, the plugin cancels this fallback and waits for
the segment's final transcript.
The timer can expire even if background audio keeps local VAD in the speaking
state. Without a local `vad`, automatic server commits continue to end speech.
The plugin logs deferrals at DEBUG and fallback timeouts at WARNING.
Before reconnecting, the plugin finishes any deferred turn.
If local VAD fails, the plugin warns and disables the guard for that connection
while transcription continues.

The other commit strategies are `manual` and `vad`. If `commit_strategy` is
omitted, the plugin uses `vad` when `server_vad` is a dict and `manual` otherwise.
An explicit strategy takes precedence over `server_vad`.

Use `stt.update_options(commit_strategy=...)` to change strategies. This reconnects
active streams, so change strategies between turns. A stream-level strategy
update applies to the owning STT and all its active streams.
