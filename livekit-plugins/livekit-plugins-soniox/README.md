# Soniox plugin for LiveKit Agents

Support for Soniox Speech-to-Text [Soniox](https://soniox.com/) API, using WebSocket streaming interface.

See https://docs.livekit.io/agents/integrations/stt/soniox/ for more information.

## Installation

```bash
pip install livekit-plugins-soniox
```

## Pre-requisites

The Soniox plugin requires an API key to authenticate. You can get your Soniox API key [here](https://console.soniox.com/).

Set API key in your `.env` file:

```
SONIOX_API_KEY=<your_soniox_api_key>
```

## Usage

Use Soniox in an `AgentSession` or as a standalone transcription service:

```python
from livekit.plugins import soniox

session = AgentSession(
    stt = soniox.STT(),
    # ... llm, tts, etc.
)
```

Congratulations! You are now ready to use Soniox Speech-to-Text API in your LiveKit agents.

You can test Soniox Speech-to-Text API in the LiveKit's [Voice AI quickstart](https://docs.livekit.io/agents/start/voice-ai/).

## More information and reference

Explore integration details and find comprehensive examples in our [Soniox LiveKit integration guide](https://speechdev.soniox.com/docs/speech-to-text/integrations/livekit).
