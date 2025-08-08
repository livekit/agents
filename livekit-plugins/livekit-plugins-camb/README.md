# Camb AI Plugin for LiveKit Agents

This plugin integrates [Camb AI](https://camb.ai/) text-to-speech capabilities with LiveKit Agents.

## Installation

```bash
pip install livekit-plugins-camb
```

## Usage

### Text-to-Speech

```python
from livekit.plugins import camb

# Initialize the TTS engine
tts_engine = camb.TTS(
    voice_id=1234,  # Replace with your chosen voice ID
    language=1,     # 1 for English
    gender=camb.Gender.MALE,
    age=30,
    api_key="your_camb_api_key"  # Or set CAMB_API_KEY environment variable
)

# List available voices
voices = await tts_engine.list_voices()
for voice in voices:
    print(f"Voice ID: {voice.id}, Name: {voice.voice_name}")

# Synthesize speech
audio_stream = tts_engine.synthesize("Hello, this is a test of the Camb AI text-to-speech system.")

# Process the audio stream
async for event in audio_stream:
    # Process audio frames
    pass
```

## API Key

You need a Camb AI API key to use this plugin. You can obtain one from the [Camb AI dashboard](https://camb.ai).

Set your API key either:
- As an argument when creating the TTS instance: `camb.TTS(api_key="your_api_key")`
- As an environment variable: `CAMB_API_KEY=your_api_key`

## Features

- High-quality text-to-speech synthesis
- Support for multiple voices and languages
- Customizable voice parameters (gender, age)

## License

Apache 2.0
