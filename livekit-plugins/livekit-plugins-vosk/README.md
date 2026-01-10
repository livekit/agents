# Vosk Plugin for LiveKit Agents

Offline speech-to-text plugin using [Vosk](https://alphacephei.com/vosk/) for the LiveKit Agents framework.

## Features

- **Offline Processing**: Runs entirely locally without internet connection
- **No API Keys Required**: Completely free, no cloud service costs
- **Multi-language Support**: 20+ languages including English, Spanish, French, German, Chinese, Russian, and more
- **Streaming Recognition**: Real-time transcription with interim results
- **Word-level Timestamps**: Precise timing information for each word
- **Speaker Diarization**: Optional speaker identification (requires speaker model)
- **Privacy-focused**: All processing happens on your device

## Installation

```bash
pip install livekit-plugins-vosk
```

## Download Models

Vosk requires pre-downloaded models. Download from: https://alphacephei.com/vosk/models

### Quick Start - Small English Model (~40MB)

```bash
mkdir -p ~/.cache/vosk/models
cd ~/.cache/vosk/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### Available Models

- **English**: `vosk-model-en-us-0.22` (large), `vosk-model-small-en-us-0.15` (small)
- **Spanish**: `vosk-model-es-0.42`
- **French**: `vosk-model-fr-0.22`
- **German**: `vosk-model-de-0.21`
- **Chinese**: `vosk-model-cn-0.22`
- **Russian**: `vosk-model-ru-0.42`
- **And many more...**

See the [full model list](https://alphacephei.com/vosk/models).

## Usage

### Basic Example

```python
from livekit.agents import JobContext, cli, WorkerOptions
from livekit.plugins import vosk
import os

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    # Path to your downloaded Vosk model
    model_path = os.path.expanduser("~/.cache/vosk/models/vosk-model-small-en-us-0.15")
    
    # Create STT instance
    stt_instance = vosk.STT(
        model_path=model_path,
        language="en",
        sample_rate=16000,
        enable_words=True,
    )
    
    # Use in streaming mode
    stream = stt_instance.stream()
    
    # Process audio frames...
    # stream.push_frame(audio_frame)
    
    # Get transcription events
    async for event in stream:
        if event.type == "final_transcript":
            print(f"Final: {event.alternatives[0].text}")
        elif event.type == "interim_transcript":
            print(f"Interim: {event.alternatives[0].text}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### With AgentSession

```python
from livekit.agents import Agent, AgentSession, JobContext, cli, WorkerOptions
from livekit.plugins import vosk, silero
import os

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    
    model_path = os.path.expanduser("~/.cache/vosk/models/vosk-model-en-us-0.22")
    
    agent = Agent(
        instructions="You are a helpful voice assistant.",
    )
    
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=vosk.STT(
            model_path=model_path,
            language="en",
            enable_words=True,
        ),
        llm="openai/gpt-4o",
        tts="cartesia/sonic-2",
    )
    
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### With Speaker Diarization

```python
stt_instance = vosk.STT(
    model_path="/path/to/vosk-model-en-us-0.22",
    speaker_model_path="/path/to/vosk-model-spk-0.4",
    language="en",
    enable_words=True,
)
```

## Configuration Options

### STT Constructor

- **`model_path`** (required): Path to the Vosk model directory
- **`language`** (default: `"en"`): Language code for metadata
- **`sample_rate`** (default: `16000`): Audio sample rate in Hz
- **`enable_words`** (default: `True`): Include word-level timestamps
- **`max_alternatives`** (default: `0`): Number of alternative transcriptions (0 = disabled)
- **`speaker_model_path`** (optional): Path to speaker identification model

## Supported Languages

Vosk supports 20+ languages:

- English (US, Indian)
- Spanish
- French
- German
- Italian
- Portuguese
- Chinese
- Russian
- Japanese
- Turkish
- Vietnamese
- Dutch
- Catalan
- Arabic
- Greek
- Farsi
- Filipino
- Ukrainian
- Kazakh
- Swedish
- And more...

See https://alphacephei.com/vosk/models for the complete list.

## Performance Tips

1. **Model Size**: Smaller models (~50MB) are faster but less accurate. Larger models (~1GB) provide better accuracy.
2. **Sample Rate**: Vosk works best with 16kHz audio. The plugin automatically resamples if needed.
3. **CPU Usage**: Vosk runs on CPU. For production, use a server with adequate CPU resources.
4. **Memory**: Load models once and reuse them across multiple streams to save memory. The plugin automatically caches loaded models globally.
5. **Prewarming**: Call `stt_instance.prewarm()` at startup to load models into memory before the first request, reducing initial latency.

## Advantages

- ✅ **Privacy**: All processing is local, no data sent to cloud
- ✅ **Cost**: Completely free, no API fees
- ✅ **Latency**: Lower latency without network round-trip
- ✅ **Reliability**: Works offline, no internet dependency
- ✅ **Compliance**: Easier to meet data residency requirements

## Limitations

- Requires pre-downloaded models (50MB - 1GB)
- Accuracy may be lower than latest cloud models for some languages
- Requires local compute resources (CPU/memory)
- Model updates require manual download

## License

Apache 2.0

## Links

- [Vosk Website](https://alphacephei.com/vosk/)
- [Vosk GitHub](https://github.com/alphacep/vosk-api)
- [Vosk Models](https://alphacephei.com/vosk/models)
- [LiveKit Agents](https://docs.livekit.io/agents/)
