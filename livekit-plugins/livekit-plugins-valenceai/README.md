# ValenceAI Plugin for LiveKit Agents

Real-time emotion detection for audio using [ValenceAI](https://valencevibes.com/)'s streaming WebSocket API.

This plugin wraps an underlying STT provider (e.g., Deepgram) and enriches transcriptions with emotion tags, enabling your agent to understand the emotional context of user speech.

## Installation

```bash
pip install livekit-plugins-valenceai
```

For Deepgram STT support:

```bash
pip install livekit-plugins-valenceai[deepgram]
```

## Prerequisites

You'll need:
1. A ValenceAI API key - set as `VALENCE_API_KEY` environment variable or pass directly
2. An underlying STT provider (e.g., Deepgram, AssemblyAI)

## Features

- **Real-time streaming**: WebSocket-based emotion detection with low latency
- **STT wrapper pattern**: Wraps any LiveKit STT provider to add emotion awareness
- **Emotion tagging**: Enriches transcriptions with emotion tags like `[Happy]`, `[Angry]`, `[Sad]`
- **Configurable models**: Choose between 4-emotion or 7-emotion classification
- **Confidence filtering**: Set minimum confidence thresholds for emotion tags
- **Graceful degradation**: Falls back to plain transcription if Valence is unavailable

## Quick Start

```python
from livekit.agents import AgentSession
from livekit.plugins import valenceai, deepgram

# Create emotion-aware STT
emotion_stt = valenceai.STT(
    underlying_stt=deepgram.STT(),
    # api_key="your-api-key",  # or use VALENCE_API_KEY env var
)

# Use in your agent
session = AgentSession(
    stt=emotion_stt,
    llm=your_llm,
    tts=your_tts,
)
```

## Output Format

Transcriptions are enriched with emotion tags:

```
[Neutral] Hi there, I'm calling about my order.
[Angry] I've been waiting for two weeks and it still hasn't arrived!
[Sad] I'm really disappointed with this service.
[Happy] Oh great, thank you so much for resolving this!
```

## Configuration Options

```python
from livekit.plugins import valenceai, deepgram

stt = valenceai.STT(
    # Required: underlying STT provider
    underlying_stt=deepgram.STT(),

    # Optional: Valence API key (defaults to VALENCE_API_KEY env var)
    api_key="your-valence-api-key",

    # Optional: Valence API server URL
    server_url="https://qa.getvalenceai.com",

    # Optional: Emotion model - "4emotions" or "7emotions"
    model="4emotions",

    # Optional: Minimum confidence threshold (0.0-1.0)
    # Emotions below this threshold won't be tagged
    min_confidence=0.3,
)
```

### Emotion Models

**4emotions** (default):
- Neutral
- Happy
- Sad
- Angry

**7emotions**:
- Extended model with additional emotion categories

## Advanced Usage

### Using with AgentSession

```python
from livekit.agents import AgentSession
from livekit.plugins import valenceai, deepgram, openai

async def create_agent():
    # Create emotion-aware STT
    emotion_stt = valenceai.STT(
        underlying_stt=deepgram.STT(
            model="nova-2",
            language="en",
        ),
        model="4emotions",
        min_confidence=0.25,
    )

    # The LLM will receive emotion-tagged transcriptions
    # e.g., "[Angry] I'm frustrated with this issue!"
    session = AgentSession(
        stt=emotion_stt,
        llm=openai.LLM(model="gpt-4o"),
        tts=your_tts,
    )

    return session
```

### Accessing the WebSocket Client Directly

```python
from livekit.plugins.valenceai import ValenceWebSocketClient

# For advanced use cases
client = ValenceWebSocketClient(
    api_key="your-api-key",
    server_url="https://qa.getvalenceai.com",
    model="4emotions",
)

await client.connect()

# Process audio
emotions = await client.process_audio(audio_samples, sample_rate=48000)
print(f"Detected: {emotions['dominant']} ({emotions['confidence']:.1%})")

await client.disconnect()
```

## How It Works

1. **Audio Buffering**: Audio frames are buffered as they arrive from the user
2. **Parallel Processing**: Audio is forwarded to the underlying STT while being buffered
3. **Emotion Detection**: When a final transcript arrives, buffered audio is sent to Valence AI
4. **Enrichment**: The transcript is enriched with the detected emotion tag
5. **Delivery**: The emotion-tagged transcript is forwarded to your LLM

```
User Audio → [Buffer] → Underlying STT → Transcript
                ↓
           Valence AI → Emotion
                ↓
        [Emotion] + Transcript → LLM
```

## API Reference

### STT Class

```python
valenceai.STT(
    underlying_stt: stt.STT,
    api_key: str | None = None,
    server_url: str = "https://qa.getvalenceai.com",
    model: Literal["4emotions", "7emotions"] = "4emotions",
    min_confidence: float = 0.0,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `underlying_stt` | `stt.STT` | Required | The base STT provider to wrap |
| `api_key` | `str \| None` | `None` | Valence API key (falls back to `VALENCE_API_KEY` env var) |
| `server_url` | `str` | `"https://qa.getvalenceai.com"` | Valence API server URL |
| `model` | `"4emotions" \| "7emotions"` | `"4emotions"` | Emotion classification model |
| `min_confidence` | `float` | `0.0` | Minimum confidence to include emotion tags |

### ValenceWebSocketClient Class

```python
valenceai.ValenceWebSocketClient(
    api_key: str,
    server_url: str = "https://qa.getvalenceai.com",
    model: Literal["4emotions", "7emotions"] = "4emotions",
)
```

**Methods:**

- `connect()` - Connect to Valence WebSocket server (with retry logic)
- `disconnect()` - Disconnect from the server
- `process_audio(audio_samples, sample_rate, timeout)` - Send audio and get emotion prediction
- `is_connected` - Property indicating connection status
- `latest_emotion` - Property with the most recent emotion prediction

## Error Handling

The plugin handles errors gracefully:

- **No API key**: Logs warning, returns plain transcriptions without emotion tags
- **Connection failure**: Retries with exponential backoff (max 3 attempts)
- **Timeout**: Uses last known emotion, logs warning
- **Processing error**: Logs error, returns plain transcription

## Learn More

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [ValenceAI Documentation](https://valencevibes.com/)

## License

Apache License 2.0
