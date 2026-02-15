# Azure Plugin for LiveKit Agents

Complete Azure AI integration for LiveKit Agents, including Azure Speech Services and Azure Voice Live Realtime API.

**What's included:**
- **RealtimeModel** - Azure Voice Live for speech-to-speech
- **STT** - Powered by Azure Speech Services
- **TTS** - Powered by Azure Speech Services with neural voices

For Azure OpenAI LLM (non-realtime), see the [OpenAI plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai) which supports `LLM.with_azure()`.

See [https://docs.livekit.io/agents/integrations/azure/](https://docs.livekit.io/agents/integrations/azure/) for more information.




## Realtime Mode (Azure Voice Live)

### Azure Credentials

For the realtime speech-to-speech model:

```bash
export AZURE_VOICELIVE_ENDPOINT=https://<region>.api.cognitive.microsoft.com/
export AZURE_VOICELIVE_API_KEY=<your-speech-key>
export AZURE_VOICELIVE_MODEL=<model-name>  # e.g., gpt-4o, gpt-4o-mini, etc.
export AZURE_VOICELIVE_VOICE=en-US-AvaMultilingualNeural
```

To power the intelligence of your voice agent, you have flexibility and choice in the generative AI model between GPT-Realtime, GPT-5, GPT-4.1, Phi, and more options. For supported models and regions, see [Supported models and regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live#supported-models-and-regions).



### Quick Start
Azure Voice Live provides end-to-end speech-to-speech:

```python
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import azure

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful voice assistant powered by Azure."
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user and offer assistance"
        )

@agents.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=azure.realtime.RealtimeModel(
            voice="en-US-AvaMultilingualNeural",
        )
    )

    await session.start(room=ctx.room, agent=Assistant())

if __name__ == "__main__":
    agents.cli.run_app(agents.AgentServer())
```

### Advanced Configuration

For full control over Azure Voice Live settings:

```python
import os
from azure.ai.voicelive.models import AzureSemanticVadEn, AudioInputTranscriptionOptions
from livekit.agents import AgentSession
from livekit.plugins import azure

# Configure Semantic VAD for intelligent turn detection
# Options:
# - AzureSemanticVad: Default semantic VAD (multilingual)
# - AzureSemanticVadEn: English-only, optimized for English
# - AzureSemanticVadMultilingual: Explicit multilingual support
turn_detection = AzureSemanticVadEn(
    threshold=0.5,              # Voice activity detection threshold (0.0-1.0)
    silence_duration_ms=500,    # Silence duration before turn ends
    prefix_padding_ms=300,      # Audio padding before speech
    speech_duration_ms=200,     # Minimum speech duration to trigger detection
    remove_filler_words=True,   # Remove filler words like "um", "uh"
)

# Configure input audio transcription with language constraint
# This helps prevent language misidentification
input_audio_transcription = AudioInputTranscriptionOptions(
    model="whisper-1",
    language="en-US",  # Constrain to English for reliable detection
)

session = AgentSession(
    llm=azure.realtime.RealtimeModel(
        endpoint=os.getenv("AZURE_VOICELIVE_ENDPOINT"),
        api_key=os.getenv("AZURE_VOICELIVE_API_KEY"),
        model=os.getenv("AZURE_VOICELIVE_MODEL", "gpt-4o"),
        voice=os.getenv("AZURE_VOICELIVE_VOICE", "en-US-AvaMultilingualNeural"),
        input_audio_transcription=input_audio_transcription,
        turn_detection=turn_detection,
        tool_choice="auto",  # Enable function calling
    )
)
```

### RealtimeModel Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `endpoint` | Azure Voice Live endpoint URL | `AZURE_VOICELIVE_ENDPOINT` env var |
| `api_key` | Azure API key | `AZURE_VOICELIVE_API_KEY` env var |
| `model` | Model name | `AZURE_VOICELIVE_MODEL` env var |
| `voice` | Azure neural voice name | `AZURE_VOICELIVE_VOICE` env var |
| `input_audio_transcription` | Audio transcription config (model, language) | `whisper-1` with auto-detect |
| `turn_detection` | VAD configuration object | Server default |
| `tool_choice` | Function calling mode ("auto", "none", etc.) | "auto" |

### Turn Detection Options

| VAD Type | Description |
|----------|-------------|
| `AzureSemanticVad` | Default semantic VAD (multilingual) |
| `AzureSemanticVadEn` | English-only, optimized for English |
| `AzureSemanticVadMultilingual` | Explicit multilingual support |

### VAD Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `threshold` | Voice activity detection sensitivity | 0.0-1.0 |
| `silence_duration_ms` | Silence duration before turn ends | milliseconds |
| `prefix_padding_ms` | Audio padding before detected speech | milliseconds |
| `speech_duration_ms` | Minimum speech duration to trigger | milliseconds |
| `remove_filler_words` | Remove "um", "uh", etc. | boolean |

### Azure Voice Live Capabilities

- **Low-latency conversations** - Realtime bidirectional streaming
- **Multilingual support** - Works with Azure's multilingual neural voices
- **Function calling** - Built-in tool use for agentic workflows
- **Interruption handling** - Graceful handling of user interruptions

## Pipeline Mode (STT + LLM + TTS)


### Azure Credentials

**Azure Speech Services (for STT/TTS)**

```bash
export AZURE_SPEECH_KEY=<your-speech-key>
export AZURE_SPEECH_REGION=<your-region>  # e.g., eastus, westus2
```

**Azure OpenAI (for LLM)**

```bash
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<your-openai-key>
export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
export AZURE_OPENAI_API_VERSION=2024-10-01-preview
```

### Quick Start

For more control over individual components:

```python
from livekit.agents import Agent, AgentSession
from livekit.plugins import azure, openai, silero

session = AgentSession(
    stt=azure.STT(),
    llm=openai.LLM.with_azure(
        model="gpt-4o",
        azure_endpoint="https://your-resource.openai.azure.com",
        api_key="your-api-key",
        api_version="2024-10-01-preview",
    ),
    tts=azure.TTS(voice="en-US-JennyNeural"),
    vad=silero.VAD.load(),
)

await session.start(room=ctx.room, agent=Agent(instructions="You are helpful."))
```

## Voice Selection

Azure provides a wide variety of neural voices. Use multilingual voices for multi-language support:

```python
# Multilingual voices (recommended)
model = azure.realtime.RealtimeModel(
    voice="en-US-AvaMultilingualNeural"
)

# Or use TTS with specific voices
tts = azure.TTS(voice="en-US-JennyNeural")
```

**Popular voices:**
- `en-US-AvaMultilingualNeural` - Multilingual, feminine
- `en-US-AndrewMultilingualNeural` - Multilingual, masculine
- `en-US-JennyNeural` - English, feminine
- `en-US-GuyNeural` - English, masculine

See [Azure Voice Gallery](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts) for full list.

### HD Voices

For higher quality audio, use DragonHD voices:

```python
tts = azure.TTS(voice="en-US-Ava:DragonHDLatestNeural")
```

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit Azure Integration Guide](https://docs.livekit.io/agents/integrations/azure/)
- [Azure Voice Live Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live)
- [Azure Voice Live Supported Models and Regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live#supported-models-and-regions)
- [Azure Speech Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure Voice Gallery](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts)
- [Azure Portal](https://portal.azure.com)
- [OpenAI Plugin (for Azure OpenAI LLM)](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai)
