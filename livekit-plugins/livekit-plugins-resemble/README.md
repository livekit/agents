# LiveKit Plugins Resemble

Agent Framework plugin for voice synthesis with the [Resemble AI](https://www.resemble.ai/) API, using their WebSocket streaming interface.

## Installation

```bash
pip install livekit-plugins-resemble
```

## Pre-requisites

You'll need an API key from Resemble AI. It can be set as an environment variable: `RESEMBLE_API_KEY`

Additionally, you'll need the voice UUID and project UUID from your Resemble AI account.

## Usage

```python
from livekit.plugins.resemble import TTS

# Initialize TTS with your credentials
tts = TTS(
    api_key="your_api_key",  # or set RESEMBLE_API_KEY environment variable
    voice_uuid="your_voice_uuid",
    project_uuid="your_project_uuid"
)

# Synthesize text to speech
audio_stream = tts.synthesize("Hello, world!")

# Or use streaming for real-time synthesis
stream = tts.stream()
await stream.synthesize_text("Hello, world!")
``` 