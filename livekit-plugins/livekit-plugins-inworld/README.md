# Inworld plugin for LiveKit Agents

Support for voice synthesis with [Inworld](https://beta.docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech).

See [https://docs.livekit.io/agents/integrations/tts/inworld/](https://docs.livekit.io/agents/integrations/tts/inworld/) for more information.

## Installation

```bash
pip install livekit-plugins-inworld
```

## Authentication

The Inworld plugin requires a [Inworld API key](https://studio.inworld.ai/login).

Set `INWORLD_API_KEY` in your `.env` file.

## Usage

Use Inworld TTS within an `AgentSession` or as a standalone speech generator. For example,
you can use this TTS in the [Voice AI quickstart](/agents/start/voice-ai/).

```python
from livekit.plugins import inworld

session = AgentSession(
   tts=inworld.TTS(voice="Olivia")
   # ... llm, stt, etc.
)
```
