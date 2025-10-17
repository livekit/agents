# Fish Audio plugin for LiveKit Agents

Support for voice synthesis with [Fish Audio](https://fish.audio/).

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for more information.

## Installation

```bash
pip install livekit-plugins-fishaudio
```

## Pre-requisites

You'll need an API key from Fish Audio. It can be set as an environment variable: `FISH_API_KEY`

## Usage

### Basic Usage

```python
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import fishaudio

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
        tts=fishaudio.TTS(),
        chat_ctx=initial_ctx,
    )

    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Using a Specific Voice Model

```python
tts = fishaudio.TTS(
    reference_id="your_voice_model_id",
    model="speech-1.6",
)
```

### Using Custom Reference Audio

```python
from livekit.plugins import fishaudio

# Read reference audio file
with open("reference.wav", "rb") as f:
    reference_audio = fishaudio.create_reference_audio(
        audio=f.read(),
        text="This is the reference audio text",
    )

# Use in TTS request (requires custom implementation)
```

### Available Models

- `speech-1.5`: Earlier version
- `speech-1.6`: Latest recommended version (default)
- `agent-x0`: Experimental agent model
- `s1`: Compact model
- `s1-mini`: Smallest model

### Configuration Options

```python
tts = fishaudio.TTS(
    api_key="your_api_key",              # Optional, reads from FISH_API_KEY env var
    model="speech-1.6",                  # TTS backend/model
    reference_id="voice_model_id",       # Optional reference voice
    output_format="mp3",                 # Output format: "mp3", "wav", "pcm"
    sample_rate=24000,                   # Sample rate in Hz
    num_channels=1,                      # Audio channels (mono)
    base_url="https://custom.api.url",   # Optional custom API endpoint
)
```

### Dynamic Voice Switching

```python
# Change voice model during runtime
tts.update_options(
    model="s1",
    reference_id="different_voice_id",
)
```

### Listing Available Voice Models

```python
# Get all available models
models = await tts.list_models()
for model in models:
    print(f"Model: {model['title']} - ID: {model['_id']}")

# Get specific model info
model_info = await tts.get_model("your_model_id")
print(f"Model details: {model_info}")
```

## API Reference

See the [Fish Audio API documentation](https://docs.fish.audio) for more details on available features and models.
