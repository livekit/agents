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
    server_url="https://api.getvalenceai.com",

    # Optional: Emotion model - "4emotions" or "7emotions"
    model="4emotions",

    # Optional: Minimum confidence threshold (0.0-1.0)
    # Predictions below this threshold are tagged [Neutral]
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
    server_url="https://api.getvalenceai.com",
    model="4emotions",
)

await client.connect()
await client.start_streaming()

# Stream audio frames as they arrive
await client.send_audio_chunk(audio_data, sample_rate=48000, samples_per_channel=960)

# Predictions arrive asynchronously (~every 5s of audio); read without blocking
emotions = await client.get_latest_emotion()
print(f"Detected: {emotions['dominant']} ({emotions['confidence']:.1%})")

await client.stop_streaming()
await client.disconnect()
```

## How It Works

1. **Continuous Streaming**: Each audio frame is forwarded to the underlying STT and, in parallel, streamed to Valence AI through an ordered sender
2. **Asynchronous Predictions**: Valence emits an emotion prediction roughly every 5 seconds of audio; predictions are stored locally with audio-position timestamps
3. **Enrichment**: When the underlying STT emits a final transcript, each sentence is tagged with the stored prediction closest to its audio time range — an instant local lookup, never a blocking network call
4. **Delivery**: The emotion-tagged transcript is forwarded to your LLM

```
User Audio ─┬─→ Underlying STT ──→ Final Transcript ─┐
            │                                        ├─→ [Emotion] Transcript → LLM
            └─→ Valence AI ──→ Timestamped ──────────┘
                               Predictions
```

## API Reference

### STT Class

```python
valenceai.STT(
    underlying_stt: stt.STT,
    api_key: str | None = None,
    server_url: str = "https://api.getvalenceai.com",
    model: Literal["4emotions", "7emotions"] = "4emotions",
    min_confidence: float = 0.0,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `underlying_stt` | `stt.STT` | Required | The base STT provider to wrap |
| `api_key` | `str \| None` | `None` | Valence API key (falls back to `VALENCE_API_KEY` env var) |
| `server_url` | `str` | `"https://api.getvalenceai.com"` | Valence API server URL |
| `model` | `"4emotions" \| "7emotions"` | `"4emotions"` | Emotion classification model |
| `min_confidence` | `float` | `0.0` | Minimum confidence for a specific emotion tag; below it, `[Neutral]` is used |

### ValenceWebSocketClient Class

```python
valenceai.ValenceWebSocketClient(
    api_key: str,
    server_url: str = "https://api.getvalenceai.com",
    model: Literal["4emotions", "7emotions"] = "4emotions",
)
```

**Methods:**

- `connect()` - Connect to Valence WebSocket server (with retry logic)
- `disconnect()` - Disconnect from the server
- `start_streaming()` / `stop_streaming()` - Begin/end a streaming session
- `send_audio_chunk(audio_data, sample_rate, samples_per_channel)` - Stream one audio frame
- `get_latest_emotion()` - Most recent prediction, non-blocking
- `get_emotion_for_timerange(start_ms, end_ms)` - Prediction closest to an audio time range, non-blocking
- `is_connected` - Property indicating connection status
- `latest_emotion` - Property with the most recent emotion prediction

## Error Handling

The plugin handles errors gracefully:

- **No API key**: Raises `ValueError` at initialization — set `VALENCE_API_KEY` env var or pass `api_key` directly
- **Connection failure**: Retries with exponential backoff (max 3 attempts)
- **No prediction yet**: Falls back to `[Neutral]` until the first prediction arrives
- **Processing error**: Logs error, returns plain transcription

Note: emotion enrichment applies to streaming recognition (`stream()`), which is
what `AgentSession` uses. Batch `recognize()` returns the underlying STT's
transcription unchanged, since the Valence API needs ~5s of accumulated audio
per prediction.

## Learn More

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [ValenceAI Documentation](https://valencevibes.com/)

## License

Apache License 2.0
