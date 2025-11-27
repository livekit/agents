# Inworld plugin for LiveKit Agents

Support for voice synthesis with [Inworld TTS](https://docs.inworld.ai/docs/tts/tts).

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
