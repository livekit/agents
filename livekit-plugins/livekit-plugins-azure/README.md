# Azure Plugin for LiveKit Agents

Complete Azure AI integration for LiveKit Agents, including Azure Speech Services and Azure Voice Live Realtime API.

**What's included:**
- **RealtimeModel** - Azure Voice Live for speech-to-speech
- **STT** - Powered by Azure Speech Services
- **TTS** - Powered by Azure Speech Services with neural voices

For Azure OpenAI LLM (non-realtime), see the [OpenAI plugin](https://github.com/livekit/agents/tree/main/livekit-plugins/livekit-plugins-openai) which supports `LLM.with_azure()`.

See [https://docs.livekit.io/agents/integrations/azure/](https://docs.livekit.io/agents/integrations/azure/) for more information.

## Installation

```bash
pip install livekit-plugins-azure
```

## Realtime Mode (Azure Voice Live)

### Azure Credentials

```bash
export AZURE_VOICE_LIVE_ENDPOINT=https://<resource>.services.ai.azure.com
export AZURE_VOICE_LIVE_API_KEY=<your-key>
export AZURE_VOICE_LIVE_MODEL=<model-name>  # e.g., gpt-realtime, gpt-4.1
```

To authenticate with Microsoft Entra ID instead of an API key, pass `entra_token=` a token with the `https://ai.azure.com/.default` scope.

You can choose between GPT-Realtime, GPT-5, GPT-4.1, Phi and more. For supported models and regions, see [Supported models and regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live#supported-models-and-regions).

### Quick Start

```python
from livekit.agents import Agent, VoiceSession
from livekit.plugins import azure

session = VoiceSession(
    llm=azure.realtime.RealtimeModel(voice="en-US-AvaMultilingualNeural"),
)
```

### Advanced Configuration

Voice Live options are passed as the JSON the service documents:

```python
import os
from livekit.plugins import azure

model = azure.realtime.RealtimeModel(
    endpoint=os.getenv("AZURE_VOICE_LIVE_ENDPOINT"),
    api_key=os.getenv("AZURE_VOICE_LIVE_API_KEY"),
    model="gpt-4.1",
    voice="en-US-AvaMultilingualNeural",
    # azure-speech with text models such as gpt-4.1, whisper-1 with gpt-realtime
    input_audio_transcription={"model": "azure-speech", "language": "en-US"},
    turn_detection={
        "type": "azure_semantic_vad",
        "silence_duration_ms": 500,
        "remove_filler_words": True,
    },
    input_audio_noise_reduction={"type": "azure_deep_noise_suppression"},
)
```

### RealtimeModel Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `endpoint` | Azure Voice Live resource endpoint | `AZURE_VOICE_LIVE_ENDPOINT` env var |
| `api_key` | Azure API key | `AZURE_VOICE_LIVE_API_KEY` env var |
| `entra_token` | Microsoft Entra token, instead of an API key | None |
| `model` | Model name | `AZURE_VOICE_LIVE_MODEL` env var, or `gpt-realtime` |
| `voice` | Azure voice name, OpenAI voice name, or a Voice Live voice object | `en-US-AvaMultilingualNeural` |
| `modalities` | `["text"]` for text-only responses | audio |
| `input_audio_transcription` | Transcription config (model, language), `None` to disable | `azure-speech` for text models and `phi4-mm-realtime`, `whisper-1` otherwise |
| `turn_detection` | Turn detection config, e.g. `server_vad`, `azure_semantic_vad`, `azure_semantic_vad_multilingual` | `server_vad` |
| `input_audio_noise_reduction` | Noise suppression config | None |
| `tool_choice` | Function calling mode | `"auto"` |
| `max_output_tokens` | Maximum output tokens per response | `4096` |

See [How to use the Voice Live API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-how-to) for every turn detection, noise suppression and voice option.

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
