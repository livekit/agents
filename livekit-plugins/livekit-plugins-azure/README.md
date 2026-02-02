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

## Prerequisites

### Azure Credentials

You'll need Azure credentials for the services you want to use:

#### Azure Speech Services (STT/TTS)

Set these environment variables:

```bash
export AZURE_SPEECH_KEY=<your-speech-key>
export AZURE_SPEECH_REGION=<your-region>  # e.g., eastus, westus2
```

**Setup:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Create a **Cognitive Services** or **Speech Service** resource
3. Go to **Keys and Endpoint** to get your key and region

#### Azure Voice Live (Realtime)

For the realtime speech-to-speech model:

```bash
export AZURE_VOICELIVE_ENDPOINT=https://<region>.api.cognitive.microsoft.com/
export AZURE_VOICELIVE_API_KEY=<your-speech-key>
export AZURE_VOICELIVE_MODEL=gpt-4o
export AZURE_VOICELIVE_VOICE=en-US-AvaMultilingualNeural
```

#### Azure OpenAI (for pipeline mode LLM)

```bash
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
export AZURE_OPENAI_API_KEY=<your-openai-key>
export AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
export AZURE_OPENAI_API_VERSION=2024-10-01-preview
```

## Quick Start

### Realtime Mode (Azure Voice Live)

Azure Voice Live provides end-to-end speech-to-speech with GPT-4o:

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

### Pipeline Mode (STT + LLM + TTS)

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

## Features

### Azure Voice Live Capabilities

Azure Voice Live provides unified speech-to-speech:

- **Low-latency conversations** - Realtime bidirectional streaming
- **Multilingual support** - Works with Azure's multilingual neural voices
- **Function calling** - Built-in tool use for agentic workflows
- **Interruption handling** - Graceful handling of user interruptions

### Voice Selection

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
- [Azure Speech Service Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure Voice Gallery](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts)
