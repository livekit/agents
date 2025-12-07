# Inworld plugin for LiveKit Agents

Support for voice synthesis with [Inworld TTS](https://docs.inworld.ai/docs/tts/tts).

See [https://docs.livekit.io/agents/integrations/tts/inworld/](https://docs.livekit.io/agents/integrations/tts/inworld/) for more information.

## Installation

```bash
pip install livekit-plugins-inworld
```

## Authentication

Set `INWORLD_API_KEY` in your `.env` file ([get one here](https://platform.inworld.ai/login)).

## Usage

Use Inworld TTS within an `AgentSession` or as a standalone speech generator. For example,
you can use this TTS in the [Voice AI quickstart](/agents/start/voice-ai/).

```python
from livekit.plugins import inworld

tts = inworld.TTS()
```

Or with options:

```python
from livekit.plugins import inworld

tts = inworld.TTS(
    voice="Hades",                 # voice ID (default or custom cloned voice)
    model="inworld-tts-1",         # or "inworld-tts-1-max"
    encoding="OGG_OPUS",           # LINEAR16, MP3, OGG_OPUS, ALAW, MULAW, FLAC
    sample_rate=48000,             # 8000-48000 Hz
    bit_rate=64000,                # bits per second (for compressed formats)
    speaking_rate=1.0,             # 0.5-1.5
    temperature=1.1,               # 0-2
    timestamp_type="WORD",         # WORD, CHARACTER, or TIMESTAMP_TYPE_UNSPECIFIED
    text_normalization="OFF",      # ON, OFF, or APPLY_TEXT_NORMALIZATION_UNSPECIFIED
)
```

## Streaming

Inworld TTS supports WebSocket streaming for lower latency real-time synthesis. Use the
`stream()` method for streaming text as it's generated:

```python
from livekit.plugins import inworld

tts = inworld.TTS(
    voice="Hades",
    model="inworld-tts-1",
    buffer_char_threshold=100,     # chars before triggering synthesis (default: 100)
    max_buffer_delay_ms=3000,      # max buffer time in ms (default: 3000)
)

# Create a stream for real-time synthesis
stream = tts.stream()

# Push text incrementally
stream.push_text("Hello, ")
stream.push_text("how are you today?")
stream.flush()  # Flush any remaining buffered text
stream.end_input()  # Signal end of input

# Consume audio as it's generated
async for audio in stream:
    # Process audio frames
    pass
```
