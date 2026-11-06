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

## Eleven Realtime (speech-to-speech)

Use an Eleven Realtime model as the `llm` of an `AgentSession`. Instructions and
function tools come from the LiveKit `Agent`:

```python
from livekit.agents import AgentSession
from livekit.plugins import elevenlabs

session = AgentSession(
    llm=elevenlabs.realtime.RealtimeModel(
        model="eleven_realtime_v1_mini",
        voice="brian",
        language="en",
    ),
)
```

The model, voice, language and turn detection lock once the conversation starts;
instructions and tools can still change. The server ends a session after
`max_duration_seconds` (600 by default, up to 7200).
