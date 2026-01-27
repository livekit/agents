# Camb.ai Plugin for LiveKit Agents

Text-to-Speech plugin for [Camb.ai](https://camb.ai) TTS API, powered by MARS technology.

## Features

- High-quality neural text-to-speech with MARS series models
- Multiple model variants (mars-flash, mars-pro)
- Enhanced pronunciation for names and places
- Support for 140+ languages
- Real-time HTTP streaming
- Pre-built voice library

## Installation

```bash
pip install livekit-plugins-camb
```

## Prerequisites

You'll need a Camb.ai API key. Set it as an environment variable:

```bash
export CAMB_API_KEY=your_api_key_here
```

Or obtain it from [Camb.ai Studio](https://studio.camb.ai/public/onboarding).

## Quick Start

```python
import asyncio
from livekit.plugins.camb import TTS

async def main():
    # Initialize TTS (uses CAMB_API_KEY env var)
    tts = TTS()

    # Synthesize speech
    stream = tts.synthesize("Hello from Camb.ai!")
    audio_frame = await stream.collect()

    # Save to file
    with open("output.wav", "wb") as f:
        f.write(audio_frame.to_wav_bytes())

asyncio.run(main())
```

## List Available Voices

```python
import asyncio
from livekit.plugins.camb import list_voices

async def main():
    voices = await list_voices()
    for voice in voices:
        print(f"{voice['name']} ({voice['id']}): {voice['gender']}, {voice['language']}")

asyncio.run(main())
```

## Select a Specific Voice

```python
tts = TTS(voice_id=147320)
stream = tts.synthesize("Using a specific voice!")
```

## Model Selection

Camb.ai offers multiple MARS models for different use cases:

```python
# Faster inference, 22050 Hz (default)
tts = TTS(model="mars-flash")

# Higher quality, 48000 Hz
tts = TTS(model="mars-pro")
```

## Advanced Configuration

```python
tts = TTS(
    api_key="your-api-key",  # Or use CAMB_API_KEY env var
    voice_id=147320,  # Voice ID from list-voices
    language="en-us",  # BCP-47 locale
    model="mars-pro",  # MARS model variant
    output_format="pcm_s16le",  # Audio format
    enhance_named_entities=True,  # Better pronunciation for names/places
)
```

## Usage with LiveKit Agents

```python
from livekit import agents
from livekit.plugins.camb import TTS

async def entrypoint(ctx: agents.JobContext):
    # Connect to room
    await ctx.connect()

    # Initialize TTS
    tts = TTS(language="en-us")

    # Synthesize and publish
    stream = tts.synthesize("Hello from LiveKit with Camb.ai!")
    audio_frame = await stream.collect()

    # Publish to room
    source = agents.AudioSource(tts.sample_rate, tts.num_channels)
    track = agents.LocalAudioTrack.create_audio_track("tts", source)
    await ctx.room.local_participant.publish_track(track)
    await source.capture_frame(audio_frame)
```

## Configuration Options

### TTS Constructor Parameters

- **api_key** (str | None): Camb.ai API key
- **voice_id** (int): Voice ID to use (default: 147320)
- **language** (str): BCP-47 locale (default: "en-us")
- **model** (SpeechModel): MARS model variant (default: "mars-flash")
- **output_format** (OutputFormat): Audio format (default: "pcm_s16le")
- **enhance_named_entities** (bool): Enhanced pronunciation (default: False)
- **sample_rate** (int | None): Audio sample rate (auto-detected from model if None)
- **base_url** (str): API base URL
- **http_session** (httpx.AsyncClient | None): Reusable HTTP session

### Available Models

- **mars-flash**: Faster inference, 22050 Hz (default)
- **mars-pro**: Higher quality synthesis, 48000 Hz

### Output Formats

- **pcm_s16le**: 16-bit PCM (recommended for streaming)
- **pcm_s32le**: 32-bit PCM (highest quality)
- **wav**: WAV with headers
- **flac**: Lossless compression
- **adts**: ADTS streaming format

## API Reference

### TTS Class

Main text-to-speech interface.

**Methods:**
- `synthesize(text: str) -> ChunkedStream`: Synthesize text to speech
- `update_options(**kwargs)`: Update voice settings dynamically
- `aclose()`: Clean up resources

**Properties:**
- `model` (str): Current MARS model name
- `provider` (str): Provider name ("Camb.ai")
- `sample_rate` (int): Audio sample rate (22050 or 48000 Hz depending on model)
- `num_channels` (int): Number of audio channels (1)

### list_voices Function

```python
async def list_voices(
    api_key: str | None = None,
    base_url: str = "https://client.camb.ai/apis",
) -> list[dict]
```

Returns list of voice dicts with: id, name, gender, age, language.

## Multi-Language Support

Camb.ai supports 140+ languages. Specify using BCP-47 locales:

```python
# French
tts = TTS(language="fr-fr", voice_id=...)

# Spanish
tts = TTS(language="es-es", voice_id=...)

# Japanese
tts = TTS(language="ja-jp", voice_id=...)
```

## Dynamic Options

Update TTS settings without recreating the instance:

```python
tts = TTS()

# Change voice
tts.update_options(voice_id=12345)

# Change model
tts.update_options(model="mars-pro")
```

## Error Handling

The plugin handles errors according to LiveKit conventions:

```python
from livekit.agents import APIStatusError, APIConnectionError, APITimeoutError

try:
    stream = tts.synthesize("Hello!")
    audio = await stream.collect()
except APIStatusError as e:
    print(f"API error: {e.status_code} - {e.message}")
except APIConnectionError as e:
    print(f"Connection error: {e}")
except APITimeoutError as e:
    print(f"Request timed out: {e}")
```

## Future Features

Coming soon:
- GCP Vertex AI integration
- Voice cloning via custom voice creation
- Voice generation from text descriptions
- WebSocket streaming for real-time applications

## Links

- [Camb.ai Documentation](https://docs.camb.ai/)
- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [GitHub Repository](https://github.com/livekit/agents)

## License

Apache License 2.0
