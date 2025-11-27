# Inworld plugin for LiveKit Agents

Support for voice synthesis with [Inworld TTS](https://docs.inworld.ai/docs/tts/tts).

## Installation

```bash
pip install livekit-plugins-inworld
```

## Authentication

Set `INWORLD_API_KEY` in your `.env` file ([get one here](https://platform.inworld.ai/login)).

## Usage

```python
from livekit.plugins import inworld

tts = inworld.TTS(
    voice="Ashley",
    timestamp_type="WORD",        # word-level timestamps for captions/lipsync
    text_normalization="OFF",     # read text exactly as written
)
```

## Features

- **Voice cloning**: Use custom voice IDs from [Inworld Portal](https://platform.inworld.ai/tts-playground)
- **Timestamps**: Word or character-level timing for karaoke, captions, lipsync
- **Custom pronunciation**: Inline IPA notation (e.g., `"Visit /kriːt/ today"`)
- **Text normalization**: Control expansion of numbers, dates, abbreviations
