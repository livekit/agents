# Fish Audio plugin for LiveKit Agents

Support for voice synthesis with [Fish Audio](https://fish.audio/).

- Docs: `https://docs.fish.audio/`

## Installation

```bash
pip install livekit-plugins-fishaudio
```

## Prerequisites

Obtain an API key from Fish Audio.

Set the API key as an environment variable:

```
FISH_API_KEY=<your_api_key>
```

## Usage

### Real-time Streaming (WebSocket)

```python
from livekit.plugins import fishaudio

tts = fishaudio.TTS(
    streaming=True,          # Enable WebSocket streaming
    latency_mode="balanced", # "normal" (~500ms) or "balanced" (~300ms)
    output_format="pcm",     # PCM recommended for streaming
)
```

### Chunked Synthesis (Non-streaming)

```python
tts = fishaudio.TTS(
    streaming=False,
    output_format="mp3",  # MP3, WAV, or PCM
)
```

## Configuration Options

```python
tts = fishaudio.TTS(
    api_key="your_api_key",       # Or set FISH_API_KEY
    model="speech-1.6",           # See Fish Audio docs for available models
    reference_id="voice_id",      # Optional voice model
    sample_rate=24000,            # Audio sample rate (Hz)
    latency_mode="balanced",      # Streaming latency optimization
    temperature=0.7,              # Consistency vs expressiveness (0.1-1.0)
    top_p=0.9,                    # Output diversity (0.1-1.0)
)
```
