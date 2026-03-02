# LiveKit Gnani Plugin

This plugin provides Speech-to-Text functionality for LiveKit Agents using Gnani's (Vachana) multilingual STT engine with strong support for Indian languages.

## Installation

```bash
pip install livekit-plugins-gnani
```

## Features

- **Multilingual Support**: Transcribe audio in multiple Indian languages including Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati, Marathi, Malayalam, Punjabi, and English (Indian accent)
- **Code-Switching**: Support for mixed language conversations (e.g., English-Hindi)
- **High-Quality Transcription**: Optimized for Indian accents and languages
- **Simple REST API**: Easy to integrate non-streaming transcription

## Supported Languages

- `en-IN` - English (India)
- `hi-IN` - Hindi
- `gu-IN` - Gujarati
- `ta-IN` - Tamil
- `kn-IN` - Kannada
- `te-IN` - Telugu
- `mr-IN` - Marathi
- `bn-IN` - Bengali
- `ml-IN` - Malayalam
- `pa-IN` - Punjabi
- `en-IN,hi-IN` - English-Hindi (code-switching)

## Configuration

You'll need to set up your Gnani credentials. You can do this via environment variables:

```bash
export GNANI_API_KEY="your-api-key"
export GNANI_ORG_ID="your-organization-id"
export GNANI_USER_ID="your-user-id"
```

Or pass them directly when initializing the STT instance.

## Usage

### Basic Usage

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import gnani

async def entrypoint(ctx: JobContext):
    # Initialize Gnani STT
    stt_instance = gnani.STT(
        language="hi-IN",  # Hindi
    )
    
    # Use with voice assistant
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=stt_instance,
        llm=llm.openai.LLM(),
        tts=tts.openai.TTS(),
    )
    
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    assistant.start(ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### With Explicit Credentials

```python
from livekit.plugins import gnani

stt_instance = gnani.STT(
    language="en-IN",
    api_key="your-api-key",
    organization_id="your-org-id",
    user_id="your-user-id",
)
```

### Code-Switching (Multiple Languages)

```python
from livekit.plugins import gnani

# Support English-Hindi code-switching
stt_instance = gnani.STT(
    language="en-IN,hi-IN",
)
```

### Direct Recognition

```python
from livekit.plugins import gnani
from livekit import rtc

stt_instance = gnani.STT(language="ta-IN")  # Tamil

# Recognize from audio buffer
event = await stt_instance.recognize(buffer=audio_frames)
print(f"Transcript: {event.alternatives[0].text}")
```

## API Details

The Gnani STT plugin uses the Gnani Vachana API v3 endpoint:

- **Endpoint**: `https://api.vachana.ai/stt/v3`
- **Method**: POST (multipart/form-data)
- **Authentication**: Via API headers
- **Audio Formats**: WAV, MP3, FLAC, OGG, M4A
- **Sample Rates**: 8 kHz – 44.1 kHz (mono recommended)
- **Max Duration**: Up to 60 seconds per request

## Limitations

- **Non-Streaming**: This plugin uses the REST API which does not support streaming recognition. For streaming use cases, consider using `StreamAdapter` with a VAD.
- **Audio Duration**: Maximum 60 seconds per request
- **No Interim Results**: Only final transcripts are returned

## Links

- [LiveKit Documentation](https://docs.livekit.io)
- [Gnani Vachana Platform](https://vachana.ai)
- [GitHub Repository](https://github.com/livekit/agents)

## License

Apache License 2.0
