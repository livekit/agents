# Fish Audio plugin for LiveKit Agents

Support for voice synthesis with [Fish Audio](https://fish.audio/).

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for more information.

## Installation

```bash
pip install livekit-plugins-fishaudio
```

## Pre-requisites

You'll need an API key from Fish Audio. You can set the following environment variables:

- `FISH_API_KEY`: Your Fish Audio API key (required)
- `FISH_AUDIO_REFERENCE_ID`: Optional default reference voice model ID

## Usage

### Real-time Streaming (Recommended)

Fish Audio's WebSocket streaming provides the lowest latency for interactive voice applications:

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import fishaudio, deepgram, openai, silero

async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You are a friendly voice assistant.",
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=fishaudio.TTS(
            streaming=True,  # Enable WebSocket streaming (default)
            latency_mode="balanced",  # ~300ms latency
            output_format="pcm",  # PCM recommended for streaming
        ),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Chunked Synthesis (Non-streaming)

For use cases where latency is not critical:

```python
tts = fishaudio.TTS(
    streaming=False,  # Disable streaming
    output_format="mp3",  # MP3, WAV, or PCM
)

# Use synthesize() instead of stream()
stream = tts.synthesize("Hello, world!")
async for audio in stream:
    # Process complete audio chunks
    pass
```

### Voice Model Configuration

#### Using a Reference Voice Model

```python
tts = fishaudio.TTS(
    reference_id="your_voice_model_id",
    model="speech-1.6",
    streaming=True,
)
```

#### Using Custom Reference Audio

```python
from livekit.plugins import fishaudio

# Create reference audio from file
with open("reference.wav", "rb") as f:
    reference_audio = fishaudio.create_reference_audio(
        audio=f.read(),
        text="This is the reference audio text",
    )

# Note: Fish Audio SDK's TTSRequest supports reference audio
# You may need to extend the implementation for custom reference usage
```

### Advanced Configuration

#### Latency Optimization

```python
tts = fishaudio.TTS(
    latency_mode="balanced",  # "normal" (~500ms) or "balanced" (~300ms)
    streaming=True,
)
```

#### Synthesis Parameters

Control voice expressiveness and consistency:

```python
tts = fishaudio.TTS(
    temperature=0.7,  # 0.1-1.0: Lower = more consistent, Higher = more expressive
    top_p=0.9,        # 0.1-1.0: Controls diversity
    streaming=True,
)
```

### Available Models

- `speech-1.5`: Earlier version with good quality
- `speech-1.6`: Latest recommended version (default)
- `agent-x0`: Experimental agent-optimized model
- `s1`: Compact model for lower latency
- `s1-mini`: Smallest model for maximum speed

### Configuration Options

```python
tts = fishaudio.TTS(
    # Required
    api_key="your_api_key",              # Or set FISH_API_KEY env var

    # Voice Configuration
    model="speech-1.6",                  # TTS backend/model
    reference_id="voice_model_id",       # Optional reference voice

    # Audio Format
    output_format="pcm",                 # "pcm", "mp3", "wav"
    sample_rate=24000,                   # Sample rate in Hz
    num_channels=1,                      # Audio channels (mono)

    # Streaming Configuration
    streaming=True,                      # Enable real-time WebSocket streaming
    latency_mode="balanced",             # "normal" (~500ms) or "balanced" (~300ms)

    # Synthesis Parameters
    temperature=0.7,                     # Consistency vs expressiveness (0.1-1.0)
    top_p=0.9,                          # Output diversity (0.1-1.0)

    # Advanced
    base_url="https://api.fish.audio",  # Custom API endpoint
)
```

## Error Handling

The plugin uses LiveKit's standard exception framework:

```python
from livekit.agents import APIConnectionError, APITimeoutError, APIStatusError

try:
    stream = tts.stream()
    # ... use stream
except APIConnectionError as e:
    # Network or WebSocket connection failed
    print(f"Connection error: {e}")
except APITimeoutError as e:
    # Request or streaming timed out
    print(f"Timeout: {e}")
except APIStatusError as e:
    # Fish Audio API returned an error
    print(f"API error: {e}")
```

## API Reference

See the [Fish Audio API documentation](https://docs.fish.audio) for more details on available features and models.
